import time
import uuid
import json
import os
import asyncio
import hashlib
import base64
from contextlib import asynccontextmanager
from typing import List, Dict, Any, AsyncGenerator, cast
import redis
import httpx
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Intenta usar orjson para un parseo JSON más rápido ---
try:
    import orjson
    def orjson_dumps(v, *, default): return orjson.dumps(v, default=default).decode()
    JSON_SERIALIZER, JSON_DESERIALIZER = orjson_dumps, orjson.loads
    print("Usando 'orjson' para un rendimiento JSON mejorado.")
except ImportError:
    JSON_SERIALIZER, JSON_DESERIALIZER = json.dumps, json.loads
    print("Usando 'json' estándar.")

# ==============================================================================
# --- 1. CONFIGURACIÓN Y CONSTANTES ---
# ==============================================================================

load_dotenv()
API_TITLE = "Qwen Web API Proxy (con Mapeo de Modelos)"
API_VERSION = "10.0.0" # Versión final con control de "thinking" por modelo

### INICIO DE LA CONFIGURACIÓN DE MODELOS
# --- CORRECCIÓN: Ahora cada modelo especifica si debe usar 'thinking' ---
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "qwen3-max": {
        "internal_model_id": "qwen3-max",
        "filter_phase": False,
        "enable_thinking": False
    },
    "qwen-final": {
        "internal_model_id": "qwen3-235b-a22b",
        "filter_phase": True,
        "enable_thinking": False
    },
    "qwen-thinking": {
        "internal_model_id": "qwen3-235b-a22b",
        "filter_phase": False,
        "enable_thinking": True # Este modelo está explícitamente configurado para pensar
    },
    "qwen-coder-plus": {
        "internal_model_id": "qwen3-coder-plus",
        "filter_phase": True,
        "enable_thinking": False
    },
    "qwen-coder-30b": {
        "internal_model_id": "qwen3-coder-30b-a3b-instruct",
        "filter_phase": True,
        "enable_thinking": False
    }
}
### FIN DE LA CONFIGURACIÓN DE MODELOS

QWEN_API_BASE_URL = "https://chat.qwen.ai/api/v2"
QWEN_AUTH_TOKEN_FALLBACK = os.getenv("QWEN_AUTH_TOKEN")
QWEN_COOKIES_JSON_B64 = os.getenv("QWEN_COOKIES_JSON_B64")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
if not UPSTASH_REDIS_URL: raise RuntimeError("UPSTASH_REDIS_URL debe estar definida.")

QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING = "", ""

def process_cookies_and_extract_token(b64_string: str | None):
    global QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING
    if not b64_string:
        print("[WARN] Cookies no definidas. Usando Auth Token de respaldo."); QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK or ""; return
    try:
        cookies_list = JSON_DESERIALIZER(base64.b64decode(b64_string))
        QWEN_COOKIE_STRING = "; ".join([f"{c['name']}={c['value']}" for c in cookies_list])
        found_token = next((c.get("value", "") for c in cookies_list if c.get("name") == "token"), "")
        QWEN_AUTH_TOKEN = f"Bearer {found_token}" if found_token else (QWEN_AUTH_TOKEN_FALLBACK or "")
        print("✅ Token de autorización y cookies procesados.")
    except Exception as e:
        print(f"[ERROR CRÍTICO] Procesando cookies: {e}."); QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING = QWEN_AUTH_TOKEN_FALLBACK or "", ""

process_cookies_and_extract_token(QWEN_COOKIES_JSON_B64)

try:
    redis_client = redis.from_url(UPSTASH_REDIS_URL, decode_responses=True); redis_client.ping(); print("✅ Conexión a Redis (Upstash) exitosa.")
except Exception as e: raise RuntimeError(f"❌ ERROR CRÍTICO AL CONECTAR CON REDIS: {e}") from e

MIN_CHAT_ID_POOL_SIZE, MAX_CHAT_ID_POOL_SIZE = 2, 5; REDIS_POOL_KEY = "qwen_chat_id_pool"; app_state: Dict[str, Any] = {}

# ==============================================================================
# --- 2. MODELOS DE DATOS ---
# ==============================================================================

class OpenAIMessage(BaseModel): role: str; content: str
class OpenAIChunkDelta(BaseModel): content: str | None = None; role: str | None = None
class OpenAIChunkChoice(BaseModel): index: int = 0; delta: OpenAIChunkDelta; finish_reason: str | None = None
class OpenAICompletionChunk(BaseModel): id: str; object: str = "chat.completion.chunk"; created: int; model: str; choices: List[OpenAIChunkChoice]
class OpenAIResponseChoice(BaseModel): index: int = 0; message: OpenAIMessage; finish_reason: str = "stop"
class OpenAIUsage(BaseModel): prompt_tokens: int = 0; completion_tokens: int = 0; total_tokens: int = 0
class OpenAIChatCompletion(BaseModel): id: str; object: str = "chat.completion"; created: int; model: str; choices: List[OpenAIResponseChoice]; usage: OpenAIUsage
class ConversationState(BaseModel): last_parent_id: str | None = None

# ==============================================================================
# --- 3. GESTIÓN DE ESTADO Y POOL ---
# ==============================================================================

def get_conversation_state(qwen_chat_id: str) -> ConversationState | None:
    state_json = redis_client.get(f"qwen_conv:{qwen_chat_id}"); return ConversationState.model_validate_json(state_json) if state_json else None
def save_conversation_state(qwen_chat_id: str, state: ConversationState):
    redis_client.set(f"qwen_conv:{qwen_chat_id}", state.model_dump_json(), ex=86400)

async def create_qwen_chat(client: httpx.AsyncClient, internal_model_id: str, headers: Dict[str, str]) -> str | None:
    url = f"{QWEN_API_BASE_URL}/chats/new"
    payload = {"title": "Proxy Pool Chat", "models": [internal_model_id], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
    try:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            chat_id = data.get("data", {}).get("id")
            print(f"✅ Nuevo Chat ID creado para el pool: {chat_id[:8]}...")
            return chat_id
        else:
            print(f"[ERROR] La API de Qwen no creó el chat. Respuesta: {data}")
            return None
    except httpx.HTTPStatusError as e:
        print(f"[ERROR-HTTP] Creando chat para el pool: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"[ERROR-GEN] Creando chat para el pool: {e}")
        return None

# ==============================================================================
# --- 4. LÓGICA DEL CLIENTE QWEN ---
# ==============================================================================

def get_live_qwen_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json", "Content-Type": "application/json; charset=UTF-8", "Authorization": QWEN_AUTH_TOKEN,
        "Cookie": QWEN_COOKIE_STRING, "Origin": "https://chat.qwen.ai", "Referer": "https://chat.qwen.ai/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "source": "web", "x-accel-buffering": "no",
    }

# --- CORRECCIÓN: PAYLOAD AHORA DEPENDE DE LA CONFIGURACIÓN DEL MODELO ---
def _build_qwen_completion_payload(chat_id: str, message: OpenAIMessage, parent_id: str | None, model_config: Dict[str, Any]) -> Dict[str, Any]:
    current_timestamp = int(time.time()); user_message_fid = str(uuid.uuid4())
    internal_model_id = model_config["internal_model_id"]
    thinking_enabled = model_config.get("enable_thinking", False)
    
    return {
        "stream": True, "incremental_output": True, "chat_id": chat_id, "chat_mode": "normal",
        "model": internal_model_id, "parent_id": parent_id,
        "messages": [{
            "fid": user_message_fid, "parentId": parent_id, "role": message.role,
            "content": message.content, "user_action": "chat", "files": [], "timestamp": current_timestamp,
            "models": [internal_model_id], "chat_type": "t2t",
            "feature_config": {"thinking_enabled": thinking_enabled, "output_schema": "phase"},
            "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t"
        }],
        "timestamp": current_timestamp
    }

async def _iterate_qwen_events(client: httpx.AsyncClient, qwen_chat_id: str, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={qwen_chat_id}"
    headers = {**get_live_qwen_headers(), "Referer": f"https://chat.qwen.ai/c/{qwen_chat_id}", "x-request-id": str(uuid.uuid4())}
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"): continue
                line_data = line.lstrip("data: ").strip()
                if not line_data or line_data == "[DONE]": continue
                try: yield JSON_DESERIALIZER(line_data)
                except (json.JSONDecodeError, orjson.JSONDecodeError if 'orjson' in globals() else TypeError): print(f"[WARN] No se pudo decodificar la línea del stream: {line_data}")
    except Exception as e:
        print(f"[ERROR] Excepción en la petición de streaming: {e}"); raise e

async def stream_qwen_to_openai_format(client: httpx.AsyncClient, qwen_chat_id: str, state: ConversationState, message: OpenAIMessage, requested_model: str, model_config: Dict[str, Any]) -> AsyncGenerator[str, None]:
    # --- CORRECCIÓN: Pasamos el model_config completo ---
    payload = _build_qwen_completion_payload(qwen_chat_id, message, state.last_parent_id, model_config)
    completion_id, created_ts = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    is_first_chunk = True
    def _format_sse_chunk(data: BaseModel) -> str:
        json_str = JSON_SERIALIZER(data.model_dump(exclude_unset=True), default=str); return f"data: {json_str}\n\n"
    try:
        async for qwen_event in _iterate_qwen_events(client, qwen_chat_id, payload):
            if response_created := qwen_event.get("response.created"):
                if new_parent_id := response_created.get("response_id"):
                    state.last_parent_id = new_parent_id; save_conversation_state(qwen_chat_id, state)
                    print(f"✅ Contexto actualizado para la conversación {qwen_chat_id[:8]} con parent_id: {new_parent_id[:8]}...")
                continue
            try:
                delta = qwen_event.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if model_config["filter_phase"] and delta.get("phase") != "answer":
                    if not (is_first_chunk and content): continue
                if content:
                    delta_payload = {"content": content}
                    if is_first_chunk:
                        delta_payload['role'] = 'assistant'; is_first_chunk = False
                    chunk = OpenAICompletionChunk(id=completion_id, created=created_ts, model=requested_model, choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(**delta_payload))])
                    yield _format_sse_chunk(chunk)
            except (KeyError, IndexError): continue
    except Exception as e:
        error_payload = {"error": {"message": f"Error en el proxy al contactar con el backend: {e}", "type": "proxy_error"}}; yield f"data: {json.dumps(error_payload)}\n\n"
    yield _format_sse_chunk(OpenAICompletionChunk(id=completion_id, created=created_ts, model=requested_model, choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(), finish_reason="stop")]))
    yield "data: [DONE]\n\n"

async def generate_non_streaming_response(client: httpx.AsyncClient, qwen_chat_id: str, state: ConversationState, message: OpenAIMessage, requested_model: str, model_config: Dict[str, Any]) -> OpenAIChatCompletion:
    # --- CORRECCIÓN: Pasamos el model_config completo ---
    payload = _build_qwen_completion_payload(qwen_chat_id, message, state.last_parent_id, model_config)
    completion_id, created_ts = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    full_content = []
    try:
        async for qwen_event in _iterate_qwen_events(client, qwen_chat_id, payload):
            if response_created := qwen_event.get("response.created"):
                if new_parent_id := response_created.get("response_id"):
                    state.last_parent_id = new_parent_id; save_conversation_state(qwen_chat_id, state)
                    print(f"✅ Contexto (no-stream) actualizado para {qwen_chat_id[:8]} con parent_id: {new_parent_id[:8]}...")
                continue
            try:
                delta = qwen_event.get("choices", [{}])[0].get("delta", {})
                is_first_content = not full_content
                if model_config["filter_phase"] and delta.get("phase") != "answer" and not is_first_content: continue
                if content := delta.get("content"): full_content.append(content)
            except (KeyError, IndexError): continue
    except Exception as e: raise HTTPException(status_code=502, detail=f"Error en el proxy al contactar con el backend: {e}")
    final_text = "".join(full_content)
    response = OpenAIChatCompletion(id=completion_id,created=created_ts,model=requested_model, choices=[OpenAIResponseChoice(message=OpenAIMessage(role="assistant", content=final_text), finish_reason="stop")], usage=OpenAIUsage())
    return response

# ==============================================================================
# --- 5. CICLO DE VIDA Y DEPENDENCIAS ---
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    process_cookies_and_extract_token(QWEN_COOKIES_JSON_B64)
    live_qwen_headers = get_live_qwen_headers()
    if not QWEN_AUTH_TOKEN or not QWEN_COOKIE_STRING:
        print("[CRITICAL] El token de autorización o las cookies están vacíos. EL SERVICIO NO FUNCIONARÁ.")
    else:
        print("✅ Cabeceras para el pool manager listas.")
    client = httpx.AsyncClient(timeout=60.0, http2=True)
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE)
    app_state.update({"http_client": client, "chat_id_queue": queue})
    try:
        ids_from_redis = redis_client.lrange(REDIS_POOL_KEY, 0, -1)
        if ids_from_redis:
            print(f"Cargando {len(ids_from_redis)} Chat IDs desde Redis...")
            for chat_id in ids_from_redis:
                if not queue.full(): await queue.put(chat_id)
    except Exception as e: print(f"[WARN] No se pudo cargar pool desde Redis: {e}")
    default_internal_model = next(iter(MODEL_CONFIG.values()))["internal_model_id"]
    pool_task = asyncio.create_task(chat_id_pool_manager(client, queue, default_internal_model, live_qwen_headers))
    print("✅ Gestor del pool de Chat IDs iniciado.")
    yield
    print("Iniciando apagado..."); pool_task.cancel()
    try: await pool_task
    except asyncio.CancelledError: pass
    ids_to_save = []
    while not queue.empty():
        try: ids_to_save.append(queue.get_nowait())
        except asyncio.QueueEmpty: break
    if ids_to_save:
        try:
            pipe = redis_client.pipeline(); pipe.delete(REDIS_POOL_KEY); pipe.rpush(REDIS_POOL_KEY, *ids_to_save); pipe.execute()
            print(f"Guardados {len(ids_to_save)} Chat IDs en Redis.")
        except Exception as e: print(f"[ERROR] No se pudo guardar pool en Redis: {e}")
    await client.aclose(); print("Recursos liberados.")

async def chat_id_pool_manager(client: httpx.AsyncClient, queue: asyncio.Queue[str], internal_model_id: str, headers: Dict[str, str]):
    await asyncio.sleep(5)
    while True:
        try:
            if queue.qsize() < MIN_CHAT_ID_POOL_SIZE:
                num_to_create = MAX_CHAT_ID_POOL_SIZE - queue.qsize()
                if num_to_create > 0:
                    print(f"Pool bajo mínimos ({queue.qsize()}). Creando {num_to_create} nuevos Chat IDs...")
                    tasks = [create_qwen_chat(client, internal_model_id, headers) for _ in range(num_to_create)]
                    results = await asyncio.gather(*tasks)
                    new_ids_count = 0
                    for chat_id in filter(None, results):
                        await queue.put(chat_id); new_ids_count += 1
                    if new_ids_count > 0: print(f"✅ Añadidos {new_ids_count} nuevos IDs al pool. Tamaño actual: {queue.qsize()}")
            await asyncio.sleep(10)
        except asyncio.CancelledError: break
        except Exception as e: print(f"[ERROR CRÍTICO] Gestor del pool: {e}. Reintentando en 30s."); await asyncio.sleep(30)

def get_http_client() -> httpx.AsyncClient: return cast(httpx.AsyncClient, app_state["http_client"])
def get_chat_id_queue() -> asyncio.Queue[str]: return cast(asyncio.Queue[str], app_state["chat_id_queue"])

async def get_chat_id_from_pool(client: httpx.AsyncClient, queue: asyncio.Queue[str], internal_model_id: str) -> str:
    try: return queue.get_nowait()
    except asyncio.QueueEmpty:
        print("[POOL-WARN] Pool vacío. Creando ID sobre la marcha.")
        live_qwen_headers = get_live_qwen_headers()
        chat_id = await create_qwen_chat(client, internal_model_id, live_qwen_headers)
        if chat_id: return chat_id
        raise HTTPException(status_code=503, detail="Pool de Qwen vacío y creación sobre la marcha fallida.")

# ==============================================================================
# --- 6. ENDPOINTS Y LÓGICA DE CONVERSACIÓN ---
# ==============================================================================

app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

def _normalize_and_extract_content(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str): return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "\n".join(text_parts)
    raise ValueError("El formato del contenido del mensaje es inválido o no es de tipo texto.")

async def get_or_create_conversation_id(messages: List[Dict[str, Any]], client: httpx.AsyncClient, queue: asyncio.Queue[str], internal_model_id: str) -> tuple[str, ConversationState]:
    if not messages or "content" not in messages[0]:
        raise ValueError("La lista de mensajes está vacía o el primer mensaje no tiene contenido.")
    first_message_content = _normalize_and_extract_content(messages[0])
    conversation_hash = hashlib.sha256(first_message_content.encode('utf-8')).hexdigest()
    redis_key = f"conv_hash:{conversation_hash}"
    qwen_chat_id = redis_client.get(redis_key)
    if qwen_chat_id:
        print(f"➡️  Conversación existente reconocida (hash: {conversation_hash[:8]}). ID: {qwen_chat_id[:8]}...")
        state = get_conversation_state(qwen_chat_id)
        if not state:
            print(f"[WARN] Estado no encontrado para Conv ID {qwen_chat_id[:8]}. Reinicializando estado.")
            state = ConversationState(last_parent_id=None)
            save_conversation_state(qwen_chat_id, state)
        return qwen_chat_id, state
    else:
        print(f"✨ Nueva conversación detectada (hash: {conversation_hash[:8]}). Obteniendo ID del pool...")
        new_qwen_chat_id = await get_chat_id_from_pool(client, queue, internal_model_id)
        redis_client.set(redis_key, new_qwen_chat_id, ex=86400 * 7)
        new_state = ConversationState(last_parent_id=None)
        save_conversation_state(new_qwen_chat_id, new_state)
        print(f"✅ Conversación asociada: hash {conversation_hash[:8]} -> Qwen ID {new_qwen_chat_id[:8]}")
        return new_qwen_chat_id, new_state

@app.get("/v1/models", summary="Listar Modelos Virtuales")
def list_models():
    return {"object": "list", "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "proxy"} for model_name in MODEL_CONFIG.keys()]}

@app.post("/v1/chat/completions", summary="Generar Completions (Streaming y No-Streaming)")
async def chat_completions_endpoint(request: Request, client: httpx.AsyncClient = Depends(get_http_client), queue: asyncio.Queue[str] = Depends(get_chat_id_queue)):
    try: request_body = await request.json()
    except (json.JSONDecodeError, orjson.JSONDecodeError if 'orjson' in globals() else TypeError):
        raise HTTPException(status_code=400, detail="Cuerpo de la solicitud no es JSON válido.")
    requested_model = request_body.get("model")
    if not requested_model: raise HTTPException(status_code=400, detail="El campo 'model' es requerido.")
    model_config = MODEL_CONFIG.get(requested_model)
    if not model_config: raise HTTPException(status_code=404, detail=f"Modelo '{requested_model}' no encontrado. Modelos disponibles: {list(MODEL_CONFIG.keys())}")
    messages = request_body.get("messages")
    if not messages or not isinstance(messages, list): raise HTTPException(status_code=400, detail="El campo 'messages' debe ser una lista no vacía.")
    is_stream = request_body.get("stream", False)
    try:
        qwen_chat_id, state = await get_or_create_conversation_id(messages=messages, client=client, queue=queue, internal_model_id=model_config["internal_model_id"])
        last_user_message = messages[-1]
        if last_user_message.get("role") != "user": raise ValueError("El último mensaje del historial debe tener el rol 'user'.")
        final_prompt_content = _normalize_and_extract_content(last_user_message)
        final_message_to_send = OpenAIMessage(role="user", content=final_prompt_content)
    except (ValueError, IndexError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Error procesando los mensajes: {e}")
    response_headers = {"X-Conversation-ID": qwen_chat_id}
    if is_stream:
        print(f"⚡️ Petición de Streaming para '{requested_model}' en Conv ID {qwen_chat_id[:8]}...")
        generator = stream_qwen_to_openai_format(client=client, qwen_chat_id=qwen_chat_id, state=state, message=final_message_to_send, requested_model=requested_model, model_config=model_config)
        return StreamingResponse(generator, media_type="text/event-stream", headers=response_headers)
    else:
        print(f"📦 Petición No-Streaming para '{requested_model}' en Conv ID {qwen_chat_id[:8]}...")
        full_response = await generate_non_streaming_response(client=client, qwen_chat_id=qwen_chat_id, state=state, message=final_message_to_send, requested_model=requested_model, model_config=model_config)
        return JSONResponse(content=full_response.model_dump(), headers=response_headers)

@app.get("/", summary="Estado del Servicio")
def read_root(): return {"status": "OK", "message": f"{API_TITLE} está activo."}

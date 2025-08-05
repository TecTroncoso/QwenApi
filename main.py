# main.py
import time
import uuid
import json
import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, AsyncGenerator
import redis
import httpx
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import base64

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
API_TITLE = "Qwen Web API Proxy (Compatible con Copilot)"
API_VERSION = "4.0.0" # [MODIFICADO] Versión con compatibilidad no-streaming
MODEL_QWEN_FINAL, MODEL_QWEN_THINKING = "qwen-final", "qwen-thinking"
QWEN_INTERNAL_MODEL = "qwen3-235b-a22b"
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
        print("✅ Token de autorización procesado.")
    except Exception as e:
        print(f"[ERROR CRÍTICO] Procesando cookies: {e}."); QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING = QWEN_AUTH_TOKEN_FALLBACK or "", ""
process_cookies_and_extract_token(QWEN_COOKIES_JSON_B64)
QWEN_HEADERS = { "Accept": "application/json", "Content-Type": "application/json; charset=UTF-8", "Authorization": QWEN_AUTH_TOKEN, "Cookie": QWEN_COOKIE_STRING, "Origin": "https://chat.qwen.ai", "Referer": "https://chat.qwen.ai/", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36", "source": "web", "x-accel-buffering": "no",}
try:
    redis_client = redis.from_url(UPSTASH_REDIS_URL, decode_responses=True); redis_client.ping(); print("✅ Conexión a Redis (Upstash) exitosa.")
except Exception as e: raise RuntimeError(f"❌ ERROR CRÍTICO AL CONECTAR CON REDIS: {e}") from e
MIN_CHAT_ID_POOL_SIZE, MAX_CHAT_ID_POOL_SIZE = 2, 5; REDIS_POOL_KEY = "qwen_chat_id_pool"; app_state: Dict[str, Any] = {}

# ==============================================================================
# --- 2. MODELOS DE DATOS ---
# ==============================================================================
class OpenAIMessage(BaseModel): role: str; content: str
class OpenAIChatCompletionRequest(BaseModel): model: str; messages: List[OpenAIMessage]; stream: bool = False

# --- Modelos para Streaming ---
class OpenAIChunkDelta(BaseModel): content: str | None = None; role: str | None = None
class OpenAIChunkChoice(BaseModel): index: int = 0; delta: OpenAIChunkDelta; finish_reason: str | None = None
class OpenAICompletionChunk(BaseModel): id: str; object: str = "chat.completion.chunk"; created: int; model: str; choices: List[OpenAIChunkChoice]

# --- [MODIFICADO] Modelos para respuesta No-Streaming ---
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
async def create_qwen_chat(client: httpx.AsyncClient) -> str | None:
    url = f"{QWEN_API_BASE_URL}/chats/new"; payload = {"title": "Proxy Pool Chat", "models": [QWEN_INTERNAL_MODEL], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
    try:
        response = await client.post(url, json=payload, headers=QWEN_HEADERS); response.raise_for_status(); data = response.json()
        return data.get("data", {}).get("id") if data.get("success") else None
    except Exception as e: print(f"[ERROR] Creando chat para el pool: {e}"); return None

# ==============================================================================
# --- 4. LÓGICA DEL CLIENTE QWEN ---
# ==============================================================================
def _build_qwen_completion_payload(chat_id: str, message: OpenAIMessage, parent_id: str | None) -> Dict[str, Any]:
    current_timestamp = int(time.time()); user_message_fid = str(uuid.uuid4())
    return {"stream": True, "incremental_output": True, "chat_id": chat_id, "chat_mode": "normal", "model": QWEN_INTERNAL_MODEL, "parent_id": parent_id, "messages": [{"fid": user_message_fid, "parentId": parent_id, "role": message.role, "content": message.content, "user_action": "chat", "files": [], "timestamp": current_timestamp, "models": [QWEN_INTERNAL_MODEL], "chat_type": "t2t", "feature_config": {"thinking_enabled": True, "output_schema": "phase", "thinking_budget": 81920}, "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t"}], "timestamp": current_timestamp}

def _format_sse_chunk(data: BaseModel) -> str:
    json_str = JSON_SERIALIZER(data.model_dump(exclude_unset=True), default=str); return f"data: {json_str}\n\n"

async def _iterate_qwen_events(
    client: httpx.AsyncClient, qwen_chat_id: str, payload: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:
    """[MODIFICADO] Generador base que procesa el stream de Qwen y produce eventos JSON."""
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={qwen_chat_id}"
    headers = {**QWEN_HEADERS, "Referer": f"https://chat.qwen.ai/c/{qwen_chat_id}", "x-request-id": str(uuid.uuid4())}
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"): continue
                line_data = line.lstrip("data: ").strip()
                if not line_data or line_data == "[DONE]": continue
                try:
                    yield JSON_DESERIALIZER(line_data)
                except (json.JSONDecodeError, orjson.JSONDecodeError if 'orjson' in globals() else TypeError):
                    print(f"[WARN] No se pudo decodificar la línea del stream: {line_data}")
    except Exception as e:
        print(f"[ERROR] Excepción en la petición de streaming: {e}")
        # Lanza una excepción para que el manejador superior la capture
        raise e

async def stream_qwen_to_openai_format(
    client: httpx.AsyncClient, qwen_chat_id: str, state: ConversationState,
    message: OpenAIMessage, requested_model: str
) -> AsyncGenerator[str, None]:
    """Generador para respuestas en formato OpenAI STREAMING."""
    payload = _build_qwen_completion_payload(qwen_chat_id, message, state.last_parent_id)
    completion_id, created_ts = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    is_first_chunk = True

    try:
        async for qwen_event in _iterate_qwen_events(client, qwen_chat_id, payload):
            # 1. Capturar parent_id del primer evento útil
            if response_created := qwen_event.get("response.created"):
                if new_parent_id := response_created.get("response_id"):
                    state.last_parent_id = new_parent_id
                    save_conversation_state(qwen_chat_id, state)
                    print(f"✅ Contexto actualizado para la conversación {qwen_chat_id} con parent_id: {new_parent_id[:8]}...")
                continue # Este evento no contiene contenido, así que pasamos al siguiente

            # 2. Procesar chunks de contenido
            try:
                delta = qwen_event.get("choices", [{}])[0].get("delta", {})
                if requested_model == MODEL_QWEN_FINAL and delta.get("phase") != "answer": continue
                
                content = delta.get("content")
                if content:
                    delta_payload = {"content": content}
                    # [MODIFICADO] Añadir el rol 'assistant' solo en el primer chunk de contenido
                    if is_first_chunk:
                        delta_payload['role'] = 'assistant'
                        is_first_chunk = False
                        
                    chunk = OpenAICompletionChunk(
                        id=completion_id, created=created_ts, model=requested_model,
                        choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(**delta_payload))]
                    )
                    yield _format_sse_chunk(chunk)
            except (KeyError, IndexError):
                continue
    
    except Exception as e:
        # En caso de error en el generador base, envía un chunk de error
        error_payload = {"error": {"message": f"Error en el proxy al contactar con el backend: {e}", "type": "proxy_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n"

    # 3. Enviar chunk final y [DONE]
    yield _format_sse_chunk(OpenAICompletionChunk(id=completion_id, created=created_ts, model=requested_model, choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(), finish_reason="stop")]))
    yield "data: [DONE]\n\n"

async def generate_non_streaming_response(
    client: httpx.AsyncClient, qwen_chat_id: str, state: ConversationState,
    message: OpenAIMessage, requested_model: str
) -> OpenAIChatCompletion:
    """[MODIFICADO] Función para generar una respuesta completa en formato OpenAI NO-STREAMING."""
    payload = _build_qwen_completion_payload(qwen_chat_id, message, state.last_parent_id)
    completion_id, created_ts = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    full_content = []

    try:
        async for qwen_event in _iterate_qwen_events(client, qwen_chat_id, payload):
            if response_created := qwen_event.get("response.created"):
                if new_parent_id := response_created.get("response_id"):
                    state.last_parent_id = new_parent_id
                    save_conversation_state(qwen_chat_id, state)
                    print(f"✅ Contexto (no-stream) actualizado para {qwen_chat_id} con parent_id: {new_parent_id[:8]}...")
                continue
            
            try:
                delta = qwen_event.get("choices", [{}])[0].get("delta", {})
                if requested_model == MODEL_QWEN_FINAL and delta.get("phase") != "answer": continue
                if content := delta.get("content"):
                    full_content.append(content)
            except (KeyError, IndexError):
                continue
    except Exception as e:
        # Si hay un error en el stream, lo lanzamos como una excepción HTTP
        raise HTTPException(status_code=502, detail=f"Error en el proxy al contactar con el backend: {e}")

    final_text = "".join(full_content)
    
    # Construir la respuesta final compatible con OpenAI
    response = OpenAIChatCompletion(
        id=completion_id,
        created=created_ts,
        model=requested_model,
        choices=[
            OpenAIResponseChoice(
                message=OpenAIMessage(role="assistant", content=final_text),
                finish_reason="stop"
            )
        ],
        # Los tokens son ficticios, ya que Qwen no los proporciona
        usage=OpenAIUsage() 
    )
    return response

# ==============================================================================
# --- 5. CICLO DE VIDA Y DEPENDENCIAS ---
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient(timeout=60.0, http2=True); queue = asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE)
    app_state.update({"http_client": client, "chat_id_queue": queue})
    try:
        ids_from_redis = redis_client.lrange(REDIS_POOL_KEY, 0, -1)
        if ids_from_redis:
            print(f"Cargando {len(ids_from_redis)} Chat IDs desde Redis..."); [await queue.put(chat_id) for chat_id in ids_from_redis if not queue.full()]
    except Exception as e: print(f"[WARN] No se pudo cargar pool desde Redis: {e}")
    pool_task = asyncio.create_task(chat_id_pool_manager(client, queue)); print("✅ Aplicación iniciada y lista para recibir peticiones.")
    yield
    print("Iniciando apagado..."); pool_task.cancel()
    try: await pool_task
    except asyncio.CancelledError: pass
    if not queue.empty():
        ids_to_save = list(queue.queue)
        try:
            pipe = redis_client.pipeline(); pipe.delete(REDIS_POOL_KEY)
            if ids_to_save: pipe.rpush(REDIS_POOL_KEY, *ids_to_save)
            pipe.execute(); print(f"Guardados {len(ids_to_save)} Chat IDs en Redis.")
        except Exception as e: print(f"[ERROR] No se pudo guardar pool en Redis: {e}")
    await client.aclose(); print("Recursos liberados.")

async def chat_id_pool_manager(client: httpx.AsyncClient, queue: asyncio.Queue):
    while True:
        try:
            if queue.qsize() < MIN_CHAT_ID_POOL_SIZE:
                num_to_create = MAX_CHAT_ID_POOL_SIZE - queue.qsize()
                if num_to_create > 0:
                    print(f"Pool bajo mínimos ({queue.qsize()}). Creando {num_to_create} nuevos Chat IDs...")
                    tasks = [create_qwen_chat(client) for _ in range(num_to_create)]
                    results = await asyncio.gather(*tasks); [await queue.put(chat_id) for chat_id in filter(None, results)]
            await asyncio.sleep(10)
        except asyncio.CancelledError: break
        except Exception as e: print(f"[ERROR CRÍTICO] Gestor del pool: {e}. Reintentando en 30s."); await asyncio.sleep(30)

def get_http_client() -> httpx.AsyncClient: return app_state["http_client"]
def get_chat_id_queue() -> asyncio.Queue: return app_state["chat_id_queue"]
async def get_chat_id_from_pool(client: httpx.AsyncClient = Depends(get_http_client), queue: asyncio.Queue = Depends(get_chat_id_queue)) -> str:
    try: return queue.get_nowait()
    except asyncio.QueueEmpty:
        print("[POOL-WARN] Pool vacío. Creando ID sobre la marcha."); chat_id = await create_qwen_chat(client)
        if chat_id: return chat_id
        raise HTTPException(status_code=503, detail="Pool de Qwen vacío y creación sobre la marcha fallida.")

# ==============================================================================
# --- 6. ENDPOINTS DE LA API ---
# ==============================================================================
app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
@app.get("/v1/models", summary="Listar Modelos Virtuales")
def list_models(): return {"object": "list", "data": [{"id": MODEL_QWEN_FINAL, "object": "model", "created": int(time.time()), "owned_by": "proxy"}, {"id": MODEL_QWEN_THINKING, "object": "model", "created": int(time.time()), "owned_by": "proxy"}]}

@app.post("/v1/chat/completions", summary="Generar Completions (Streaming y No-Streaming)")
async def chat_completions_endpoint(
    request: OpenAIChatCompletionRequest, 
    client: httpx.AsyncClient = Depends(get_http_client), 
    x_conversation_id: str | None = Header(None, alias="X-Conversation-ID")
):
    """
    [MODIFICADO] Endpoint principal que ahora gestiona ambos modos: streaming y no-streaming.
    """
    if not request.messages: raise HTTPException(status_code=400, detail="El campo 'messages' no puede estar vacío.")
    
    response_headers = {}
    qwen_chat_id = x_conversation_id
    state = None
    
    if qwen_chat_id:
        state = get_conversation_state(qwen_chat_id)
        if not state:
            state = ConversationState(last_parent_id=None)
            save_conversation_state(qwen_chat_id, state)
    else:
        print("➡️  Nueva conversación detectada. Obteniendo Chat ID del pool...")
        qwen_chat_id = await get_chat_id_from_pool(client, app_state["chat_id_queue"])
        state = ConversationState(last_parent_id=None)
        save_conversation_state(qwen_chat_id, state)
        response_headers["X-Conversation-ID"] = qwen_chat_id
        print(f"✨ Conversación iniciada con ID de Qwen (del pool): {qwen_chat_id}")

    last_message = request.messages[-1]

    # --- Lógica de bifurcación: Streaming vs No-Streaming ---
    if request.stream:
        print("⚡️ Petición de Streaming recibida.")
        generator = stream_qwen_to_openai_format(
            client=client, qwen_chat_id=qwen_chat_id, state=state, 
            message=last_message, requested_model=request.model
        )
        return StreamingResponse(generator, media_type="text/event-stream", headers=response_headers)
    else:
        print("📦 Petición No-Streaming recibida.")
        full_response = await generate_non_streaming_response(
            client=client, qwen_chat_id=qwen_chat_id, state=state, 
            message=last_message, requested_model=request.model
        )
        # Usar JSONResponse para asegurar que las cabeceras se envían correctamente
        return JSONResponse(content=full_response.model_dump(), headers=response_headers)

@app.get("/", summary="Estado del Servicio")
def read_root(): return {"status": "OK", "message": f"{API_TITLE} está activo."}

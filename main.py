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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import base64

# --- Intenta usar orjson para un parseo JSON más rápido ---
try:
    import orjson
    def orjson_dumps(v, *, default):
        return orjson.dumps(v, default=default).decode()
    JSON_SERIALIZER = orjson_dumps
    JSON_DESERIALIZER = orjson.loads
    print("Usando 'orjson' para un rendimiento JSON mejorado.")
except ImportError:
    JSON_SERIALIZER = json.dumps
    JSON_DESERIALIZER = json.loads
    print("Usando 'json' estándar. Instala 'orjson' para mejorar el rendimiento.")

# ==============================================================================
# --- 1. CONFIGURACIÓN Y CONSTANTES ---
# ==============================================================================
load_dotenv()

API_TITLE = "Qwen Web API Proxy (Latencia Optimizada)"
API_VERSION = "2.2.1" # Versión corregida
MODEL_QWEN_FINAL = "qwen-final"
MODEL_QWEN_THINKING = "qwen-thinking"
QWEN_INTERNAL_MODEL = "qwen3-235b-a22b"

QWEN_API_BASE_URL = "https://chat.qwen.ai/api/v2"

# --- Secretos cargados desde el entorno de Koyeb ---
QWEN_AUTH_TOKEN_FALLBACK = os.getenv("QWEN_AUTH_TOKEN")
QWEN_COOKIES_JSON_B64 = os.getenv("QWEN_COOKIES_JSON_B64")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")

if not UPSTASH_REDIS_URL:
    raise RuntimeError("La variable de entorno UPSTASH_REDIS_URL debe estar definida.")
    
# --- Variables Globales que serán pobladas por la lógica de inicio ---
QWEN_AUTH_TOKEN = ""
QWEN_COOKIE_STRING = ""

# --- Función Auxiliar INTELIGENTE para Formatear Cookies Y EXTRAER TOKEN ---
def process_cookies_and_extract_token(b64_string: str | None):
    """
    Decodifica el JSON de cookies, lo formatea en un string, y lo más importante,
    busca y extrae el token de autorización JWT de la lista de cookies.
    """
    global QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING

    if not b64_string:
        print("[WARN] La variable de cookies no está definida. Usando Auth Token de respaldo si existe.")
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK if QWEN_AUTH_TOKEN_FALLBACK else ""
        QWEN_COOKIE_STRING = ""
        return

    try:
        cookies_list = JSON_DESERIALIZER(base64.b64decode(b64_string))
        
        # 1. Formatear el string de cookies para el header
        QWEN_COOKIE_STRING = "; ".join([f"{c['name']}={c['value']}" for c in cookies_list])

        # 2. Buscar y extraer el token de autorización
        found_token = ""
        for cookie in cookies_list:
            if cookie.get("name") == "token":
                found_token = cookie.get("value", "")
                break
        
        if found_token:
            QWEN_AUTH_TOKEN = f"Bearer {found_token}"
            print("✅ Token de autorización extraído exitosamente desde las cookies.")
        else:
            print("[WARN] No se encontró un 'token' en las cookies. Usando Auth Token de respaldo.")
            QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK if QWEN_AUTH_TOKEN_FALLBACK else ""

    except Exception as e:
        print(f"[ERROR CRÍTICO] No se pudieron procesar las cookies: {e}. Usando Auth Token de respaldo.")
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK if QWEN_AUTH_TOKEN_FALLBACK else ""
        QWEN_COOKIE_STRING = ""

# --- Ejecutamos nuestra nueva función para poblar las variables globales ---
process_cookies_and_extract_token(QWEN_COOKIES_JSON_B64)

# --- Las Cabeceras ahora usan las variables que acabamos de poblar ---
QWEN_HEADERS = {
    "Accept": "application/json", "Accept-Language": "es-AR,es;q=0.7", "Authorization": QWEN_AUTH_TOKEN,
    "bx-v": "2.5.31", "Content-Type": "application/json; charset=UTF-8", "Cookie": QWEN_COOKIE_STRING,
    "Origin": "https://chat.qwen.ai", "Referer": "https://chat.qwen.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "source": "web", "x-accel-buffering": "no",
}

# --- Configuración del Pool y la Persistencia en Redis ---
MIN_CHAT_ID_POOL_SIZE = 2
MAX_CHAT_ID_POOL_SIZE = 5
REDIS_POOL_KEY = "qwen_chat_id_pool"
try:
    redis_client = redis.from_url(UPSTASH_REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("✅ Conexión a Redis (Upstash) exitosa.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CONECTAR CON REDIS: {e}")
    print("La persistencia del pool de Chat IDs estará desactivada.")
    redis_client = None

# --- Estado Global de la Aplicación ---
app_state: Dict[str, Any] = {}

# ==============================================================================
# --- 2. MODELOS DE DATOS (Pydantic) ---
# ==============================================================================
class OpenAIMessage(BaseModel):
    role: str; content: str
class OpenAIChatCompletionRequest(BaseModel):
    model: str; messages: List[OpenAIMessage]; stream: bool = False
class OpenAIChunkDelta(BaseModel):
    content: str | None = None
class OpenAIChunkChoice(BaseModel):
    index: int = 0; delta: OpenAIChunkDelta; finish_reason: str | None = None
class OpenAICompletionChunk(BaseModel):
    id: str; object: str = "chat.completion.chunk"; created: int; model: str; choices: List[OpenAIChunkChoice]

# ==============================================================================
# --- 3. LÓGICA DEL CLIENTE QWEN ---
# ==============================================================================
async def create_qwen_chat(client: httpx.AsyncClient) -> str | None:
    """Crea una nueva sesión de chat en Qwen y devuelve su ID."""
    url = f"{QWEN_API_BASE_URL}/chats/new"
    payload = {"title": "Proxy Pool Chat", "models": [QWEN_INTERNAL_MODEL], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
    try:
        response = await client.post(url, json=payload, headers=QWEN_HEADERS)
        response.raise_for_status()
        data = response.json()
        if data.get("success") and (chat_id := data.get("data", {}).get("id")):
            return chat_id
        print(f"[WARN] La API de Qwen no devolvió un chat_id válido. Respuesta: {data}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP Error al crear chat para el pool: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"[ERROR] Excepción genérica al crear chat para el pool: {e}")
        return None

def _build_qwen_completion_payload(chat_id: str, message: OpenAIMessage) -> Dict[str, Any]:
    current_timestamp = int(time.time())
    return {
        "stream": True, "incremental_output": True, "chat_id": chat_id, "chat_mode": "normal",
        "model": QWEN_INTERNAL_MODEL, "parent_id": None,
        "messages": [{
            "fid": str(uuid.uuid4()), "parentId": None, "role": message.role, "content": message.content,
            "user_action": "chat", "files": [], "timestamp": current_timestamp, "models": [QWEN_INTERNAL_MODEL],
            "chat_type": "t2t", "feature_config": {"thinking_enabled": True, "output_schema": "phase", "thinking_budget": 81920},
            "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t",
        }],
        "timestamp": current_timestamp,
    }

def _format_sse_chunk(data: BaseModel | dict) -> str:
    """Formatea un objeto Pydantic o un diccionario en un string Server-Sent Event."""
    if isinstance(data, BaseModel):
        # Si es un objeto Pydantic, usamos model_dump
        json_str = JSON_SERIALIZER(data.model_dump(exclude_unset=True), default=str)
    else:
        # Si es un diccionario, lo serializamos directamente
        json_str = JSON_SERIALIZER(data, default=str)
    return f"data: {json_str}\n\n"

async def stream_qwen_to_openai_format(
    client: httpx.AsyncClient, chat_id: str, message: OpenAIMessage, requested_model: str
) -> AsyncGenerator[str, None]:
    """
    Realiza la petición de streaming a Qwen y transforma la respuesta al formato
    de chunks de OpenAI. La primera respuesta es un chunk especial que contiene
    el 'chat_id' para que el cliente pueda mantener el estado de la sesión.
    """
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={chat_id}"
    payload = _build_qwen_completion_payload(chat_id, message)
    
    # Preparamos las cabeceras específicas para esta petición
    request_headers = QWEN_HEADERS.copy()
    request_headers["Referer"] = f"https://chat.qwen.ai/c/{chat_id}"
    request_headers["x-request-id"] = str(uuid.uuid4())

    # Generamos un ID de completion para los chunks y el timestamp de creación
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())

    # --- INICIO DE LA MODIFICACIÓN ---
    # Enviamos un primer chunk "falso" que solo contiene el ID de la conversación.
    # Usamos el campo 'id' del chunk de OpenAI para pasar nuestro 'chat_id'.
    # Este campo normalmente se usa para el completion_id, pero lo reutilizamos
    # aquí para la primera respuesta, ya que el cliente lo puede leer fácilmente.
    initial_chunk_data = {
        "id": chat_id, # <--- ¡AQUÍ ESTÁ LA MAGIA!
        "object": "chat.completion.chunk",
        "created": created_timestamp,
        "model": requested_model,
        "choices": []  # No enviamos contenido, solo metadatos
    }
    yield _format_sse_chunk(initial_chunk_data)
    # --- FIN DE LA MODIFICACIÓN ---

    try:
        async with client.stream("POST", url, json=payload, headers=request_headers, timeout=60.0) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                
                line_data = line.lstrip("data: ")
                if not line_data or line_data.strip() == "[DONE]":
                    continue
                    
                try:
                    qwen_chunk = JSON_DESERIALIZER(line_data)
                    delta = qwen_chunk.get("choices", [{}])[0].get("delta", {})
                    
                    # Filtramos por modo (final vs thinking)
                    if requested_model == MODEL_QWEN_FINAL and delta.get("phase") != "answer":
                        continue
                    
                    if content_chunk := delta.get("content"):
                        # Construimos un chunk estándar en formato OpenAI
                        openai_chunk = OpenAICompletionChunk(
                            id=completion_id,
                            created=created_timestamp,
                            model=requested_model,
                            choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(content=content_chunk))],
                        )
                        yield _format_sse_chunk(openai_chunk)

                except (json.JSONDecodeError, IndexError, KeyError, orjson.JSONDecodeError if 'orjson' in globals() else TypeError):
                    # Ignoramos chunks malformados o que no nos interesan
                    continue

    except Exception as e:
        print(f"[ERROR] Excepción durante el streaming: {e}")
        error_chunk = {"error": {"message": f"Proxy streaming error: {str(e)}", "type": "proxy_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        return

    # Enviamos el chunk final de terminación
    final_chunk = OpenAICompletionChunk(
        id=completion_id,
        created=created_timestamp,
        model=requested_model,
        choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(), finish_reason="stop")],
    )
    yield _format_sse_chunk(final_chunk)
    
    # Enviamos la señal de [DONE] final
    yield "data: [DONE]\n\n"

# ==============================================================================
# --- 4. GESTIÓN DEL CICLO DE VIDA, POOL Y DEPENDENCIAS ---
# ==============================================================================

async def chat_id_pool_manager(client: httpx.AsyncClient, queue: asyncio.Queue):
    """Tarea en segundo plano que mantiene el pool de Chat IDs lleno."""
    while True:
        try:
            if queue.qsize() < MIN_CHAT_ID_POOL_SIZE:
                num_to_create = MAX_CHAT_ID_POOL_SIZE - queue.qsize()
                if num_to_create > 0:
                    print(f"Pool bajo mínimos ({queue.qsize()}). Creando {num_to_create} nuevos Chat IDs...")
                    tasks = [create_qwen_chat(client) for _ in range(num_to_create)]
                    results = await asyncio.gather(*tasks)
                    for chat_id in filter(None, results): await queue.put(chat_id)
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ERROR CRÍTICO] Error en el gestor del pool: {e}. Reintentando en 30s.")
            await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona los recursos de la aplicación, usando Redis para la persistencia."""
    client = httpx.AsyncClient(timeout=60.0, http2=True)
    chat_id_queue = asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE)

    # --- INICIO: Cargar el estado desde Redis ---
    if redis_client:
        try:
            ids_from_redis = redis_client.lrange(REDIS_POOL_KEY, 0, -1)
            if ids_from_redis:
                print(f"Cargando {len(ids_from_redis)} Chat IDs desde el estado persistente de Redis...")
                for chat_id in ids_from_redis:
                    if not chat_id_queue.full():
                        await chat_id_queue.put(chat_id)
        except Exception as e:
            print(f"[WARN] No se pudo cargar el pool desde Redis ({e}). Se iniciará un pool vacío.")
    
    app_state.update({"http_client": client, "chat_id_queue": chat_id_queue})
    pool_task = asyncio.create_task(chat_id_pool_manager(client, chat_id_queue))
    
    print("✅ Aplicación iniciada y lista para recibir peticiones.")
    yield
    
    # --- APAGADO: Guardar el estado en Redis ---
    print("Iniciando apagado... Guardando estado del pool en Redis.")
    pool_task.cancel()
    try:
        await pool_task
    except asyncio.CancelledError:
        pass

    if redis_client and not chat_id_queue.empty():
        ids_to_save = list(chat_id_queue.queue)
        try:
            # Usamos una transacción (pipeline) para una operación atómica
            pipe = redis_client.pipeline()
            pipe.delete(REDIS_POOL_KEY)
            if ids_to_save:
                pipe.rpush(REDIS_POOL_KEY, *ids_to_save)
            pipe.execute()
            print(f"Guardados {len(ids_to_save)} Chat IDs en Redis para el próximo reinicio.")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar el estado del pool en Redis: {e}")

    await client.aclose()
    print("Recursos liberados. Apagado completo.")

def get_http_client() -> httpx.AsyncClient: return app_state["http_client"]
def get_chat_id_queue() -> asyncio.Queue: return app_state["chat_id_queue"]
async def get_or_create_chat_id(
    x_chat_id: str | None = Header(None, alias="X-Chat-ID"),
    client: httpx.AsyncClient = Depends(get_http_client),
    queue: asyncio.Queue = Depends(get_chat_id_queue)
) -> str:
    """
    Lógica inteligente para gestionar la sesión de chat.
    1. Si el cliente envía un 'X-Chat-ID' en la cabecera, lo reutiliza.
    2. Si no, toma uno nuevo del pool para iniciar una nueva conversación.
    """
    if x_chat_id:
        print(f"[SESSION] Reutilizando Chat ID existente: {x_chat_id}")
        return x_chat_id
    
    try:
        chat_id = queue.get_nowait()
        print(f"[SESSION] Usando Chat ID nuevo del pool. IDs restantes: {queue.qsize()}")
        return chat_id
    except asyncio.QueueEmpty:
        print("[SESSION-WARN] El pool de Chat IDs está vacío. Creando uno nuevo sobre la marcha.")
        chat_id = await create_qwen_chat(client)
        if chat_id: return chat_id
        raise HTTPException(status_code=503, detail="No se pudo obtener una sesión de chat de Qwen.")
# ==============================================================================
# --- 5. ENDPOINTS DE LA API (FastAPI) ---
# ==============================================================================
app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

@app.get("/v1/models", summary="Listar Modelos Virtuales")
def list_models():
    model_data = [
        {"id": MODEL_QWEN_FINAL, "object": "model", "created": int(time.time()), "owned_by": "proxy", "description": "Respuesta final."},
        {"id": MODEL_QWEN_THINKING, "object": "model", "created": int(time.time()), "owned_by": "proxy", "description": "Proceso de razonamiento completo."},
    ]
    return {"object": "list", "data": model_data}

@app.post("/v1/chat/completions", summary="Generar Completions de Chat (Latencia Optimizada)")
async def chat_completions_endpoint(
    request: OpenAIChatCompletionRequest,
    chat_id: str = Depends(get_or_create_chat_id),
    client: httpx.AsyncClient = Depends(get_http_client)
):
    if not request.messages:
        raise HTTPException(status_code=400, detail="El campo 'messages' no puede estar vacío.")
    last_message = request.messages[-1]
    generator = stream_qwen_to_openai_format(
        client=client, chat_id=chat_id, message=last_message, requested_model=request.model
    )
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/", summary="Estado del Servicio")
def read_root():
    return {"status": "OK", "message": f"{API_TITLE} está activo."}








import time
import uuid
import json
import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

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
QWEN_AUTH_TOKEN = os.getenv("QWEN_AUTH_TOKEN")
QWEN_COOKIE = os.getenv("QWEN_COOKIE")

if not QWEN_AUTH_TOKEN or not QWEN_COOKIE:
    raise RuntimeError("Las variables de entorno QWEN_AUTH_TOKEN y QWEN_COOKIE deben estar definidas en un archivo .env.")

QWEN_HEADERS = {
    "Accept": "application/json", "Accept-Language": "es-AR,es;q=0.7", "Authorization": QWEN_AUTH_TOKEN,
    "bx-v": "2.5.31", "Content-Type": "application/json; charset=UTF-8", "Cookie": QWEN_COOKIE,
    "Origin": "https://chat.qwen.ai", "Referer": "https://chat.qwen.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "source": "web", "x-accel-buffering": "no",
}

MIN_CHAT_ID_POOL_SIZE = 1
MAX_CHAT_ID_POOL_SIZE = 2

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
    """Crea una nueva sesión de chat en Qwen y devuelve su ID. Usado por el pool."""
    url = f"{QWEN_API_BASE_URL}/chats/new"
    payload = {"title": "Proxy Pool Chat", "models": [QWEN_INTERNAL_MODEL], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        
        # --- CORRECCIÓN FINAL APLICADA AQUÍ ---
        # response.text es una propiedad, no una corrutina. No se usa 'await'.
        response_text = response.text
        data = JSON_DESERIALIZER(response_text)

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

def _format_sse_chunk(data: BaseModel) -> str:
    json_str = JSON_SERIALIZER(data.model_dump(exclude_unset=True), default=str)
    return f"data: {json_str}\n\n"

async def stream_qwen_to_openai_format(
    client: httpx.AsyncClient, chat_id: str, message: OpenAIMessage, requested_model: str
) -> AsyncGenerator[str, None]:
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={chat_id}"
    payload = _build_qwen_completion_payload(chat_id, message)
    request_headers = QWEN_HEADERS.copy()
    request_headers["Referer"] = f"https://chat.qwen.ai/c/{chat_id}"
    request_headers["x-request-id"] = str(uuid.uuid4())
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())
    try:
        async with client.stream("POST", url, json=payload, headers=request_headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"): continue
                line_data = line.lstrip("data: ")
                if not line_data or line_data.strip() == "[DONE]": continue
                try:
                    qwen_chunk = JSON_DESERIALIZER(line_data)
                    delta = qwen_chunk.get("choices", [{}])[0].get("delta", {})
                    if requested_model == MODEL_QWEN_FINAL and delta.get("phase") != "answer": continue
                    if content_chunk := delta.get("content"):
                        openai_chunk = OpenAICompletionChunk(
                            id=completion_id, created=created_timestamp, model=requested_model,
                            choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(content=content_chunk))],
                        )
                        yield _format_sse_chunk(openai_chunk)
                except (json.JSONDecodeError, IndexError, KeyError, orjson.JSONDecodeError if 'orjson' in globals() else TypeError):
                    continue
    except Exception as e:
        print(f"[ERROR] Excepción durante el streaming: {e}")
        error_chunk = {"error": {"message": f"Proxy streaming error: {str(e)}", "type": "proxy_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        return
    final_chunk = OpenAICompletionChunk(
        id=completion_id, created=created_timestamp, model=requested_model,
        choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(), finish_reason="stop")],
    )
    yield _format_sse_chunk(final_chunk)
    yield "data: [DONE]\n\n"

# ==============================================================================
# --- 4. GESTIÓN DEL CICLO DE VIDA, POOL Y DEPENDENCIAS ---
# ==============================================================================
app_state: Dict[str, Any] = {}

async def chat_id_pool_manager(client: httpx.AsyncClient, queue: asyncio.Queue):
    print("Iniciando gestor del pool de Chat IDs...")
    while True:
        try:
            if queue.qsize() < MIN_CHAT_ID_POOL_SIZE:
                num_to_create = MAX_CHAT_ID_POOL_SIZE - queue.qsize()
                print(f"Pool bajo mínimos ({queue.qsize()}). Creando {num_to_create} nuevos Chat IDs...")
                tasks = [create_qwen_chat(client) for _ in range(num_to_create)]
                results = await asyncio.gather(*tasks)
                new_ids = [chat_id for chat_id in results if chat_id]
                for chat_id in new_ids:
                    await queue.put(chat_id)
                if new_ids:
                    print(f"Añadidos {len(new_ids)} Chat IDs al pool. Tamaño actual: {queue.qsize()}")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            print("Gestor del pool de Chat IDs cancelado. Terminando.")
            break
        except Exception as e:
            print(f"[ERROR CRÍTICO] Error en el gestor del pool: {e}. Reintentando en 30s.")
            await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient(timeout=60.0, http2=True, headers=QWEN_HEADERS)
    chat_id_queue = asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE)
    app_state.update({"http_client": client, "chat_id_queue": chat_id_queue})
    print("Cliente HTTPX y cola de Chat IDs inicializados.")
    pool_task = asyncio.create_task(chat_id_pool_manager(client, chat_id_queue))
    app_state["pool_task"] = pool_task
    yield
    print("Iniciando apagado de la aplicación...")
    pool_task.cancel()
    try:
        await pool_task
    except asyncio.CancelledError: pass
    await client.aclose()
    print("Recursos liberados. Apagado completo.")

def get_http_client() -> httpx.AsyncClient: return app_state["http_client"]
def get_chat_id_queue() -> asyncio.Queue: return app_state["chat_id_queue"]

async def get_chat_id_from_pool(
    client: httpx.AsyncClient = Depends(get_http_client),
    queue: asyncio.Queue = Depends(get_chat_id_queue)
) -> str:
    try:
        chat_id = queue.get_nowait()
        print(f"[POOL] Usando Chat ID del pool. IDs restantes: {queue.qsize()}")
        return chat_id
    except asyncio.QueueEmpty:
        print("[POOL-WARN] El pool de Chat IDs está vacío. Creando uno nuevo sobre la marcha.")
        chat_id = await create_qwen_chat(client)
        if chat_id: return chat_id
        raise HTTPException(status_code=503, detail="No se pudo obtener una sesión de chat de Qwen (pool vacío y creación fallida).")

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
    chat_id: str = Depends(get_chat_id_from_pool),
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
import asyncio
import base64
import os
import socket
import time
import uuid
import hashlib  # <--- CAMBIO: Usamos hashlib para un hash estándar y consistente
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import orjson
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------- SERIALIZATION ----------
def JSON_SERIALIZER(v, *, default=str):
    return orjson.dumps(v, default=default).decode()

JSON_DESERIALIZER = orjson.loads

# ---------- GLOBAL CONFIG ----------
load_dotenv()
API_TITLE = "Qwen Web API Proxy (RENDER-FIXED & CONTEXT-FIXED)"
API_VERSION = "14.0.0" # Versión actualizada con la corrección de contexto

MODEL_CONFIG = {
    "qwen-final":     {"internal_model_id": "qwen3-235b-a22b",  "filter_phase": True},
    "qwen-thinking":  {"internal_model_id": "qwen3-235b-a22b",  "filter_phase": False},
    "qwen-coder-plus":{"internal_model_id": "qwen3-coder-plus", "filter_phase": True},
    "qwen-coder-30b": {"internal_model_id": "qwen3-coder-30b-a3b-instruct", "filter_phase": True},
}

QWEN_API_BASE_URL = "https://chat.qwen.ai/api/v2"
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
if not UPSTASH_REDIS_URL:
    raise RuntimeError("UPSTASH_REDIS_URL must be set on Render")

QWEN_AUTH_TOKEN_FALLBACK = os.getenv("QWEN_AUTH_TOKEN", "")
QWEN_COOKIES_JSON_B64      = os.getenv("QWEN_COOKIES_JSON_B64", "")

# ---------- COOKIES / TOKEN ----------
QWEN_AUTH_TOKEN: str   = ""
QWEN_COOKIE_STRING: str = ""

def process_cookies_and_extract_token(b64_string: str | None):
    global QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING
    if not b64_string:
        print("[WARN] Cookies var not found → using fallback token")
        QWEN_AUTH_TOKEN   = QWEN_AUTH_TOKEN_FALLBACK
        QWEN_COOKIE_STRING = ""
        return
    try:
        cookies_list = JSON_DESERIALIZER(base64.b64decode(b64_string))
        QWEN_COOKIE_STRING = "; ".join(f"{c['name']}={c['value']}" for c in cookies_list)
        token_value = next((c.get("value", "") for c in cookies_list if c.get("name") == "token"), "")
        QWEN_AUTH_TOKEN = f"Bearer {token_value}" if token_value else QWEN_AUTH_TOKEN_FALLBACK
        print("✅ Cookies & token processed OK")
    except Exception as e:
        print(f"[ERROR] Cookie parse: {e}")
        QWEN_AUTH_TOKEN   = QWEN_AUTH_TOKEN_FALLBACK
        QWEN_COOKIE_STRING = ""

process_cookies_and_extract_token(QWEN_COOKIES_JSON_B64)

# ---------- REDIS ----------
redis_client = redis.from_url(
    UPSTASH_REDIS_URL,
    decode_responses=True,
    socket_keepalive=True,
    socket_keepalive_options={},
    health_check_interval=30,
)
try:
    redis_client.ping()
    print("✅ Redis connected")
except Exception as e:
    raise RuntimeError(f"❌ Redis connect: {e}") from e

# ---------- MODELS ----------
class OpenAIMessage(BaseModel):
    role: str
    content: str

class ConversationState(BaseModel):
    last_parent_id: Optional[str] = None

# ---------- UTILS ----------
LUA_GET_SET_EX = """
local v = redis.call('GET', KEYS[1])
if not v then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
end
return v
"""

def _build_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": QWEN_AUTH_TOKEN,
        "Cookie": QWEN_COOKIE_STRING,
        "Origin": "https://chat.qwen.ai",
        "Referer": "https://chat.qwen.ai/",
        "User-Agent": "Mozilla/5.0",
        "source": "web",
        "x-accel-buffering": "no",
    }

# ---------- HTTPX ----------
_transport = httpx.AsyncHTTPTransport(
    retries=0,
    socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
)
client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=200),
    timeout=httpx.Timeout(5.0, connect=1.0, read=15.0, write=2.0),
    http2=True,
    transport=_transport,
)

# ---------- POOL ----------
MIN_POOL, MAX_POOL = 2, 5
POOL_KEY = "qwen_chat_id_pool"
lua_script = None

async def create_chat(internal_model: str) -> Optional[str]:
    payload = {
        "title": "ProxyPool",
        "models": [internal_model],
        "chat_mode": "normal",
        "chat_type": "t2t",
        "timestamp": int(time.time() * 1000),
    }
    try:
        r = await client.post(f"{QWEN_API_BASE_URL}/chats/new", json=payload, headers=_build_headers())
        r.raise_for_status()
        return r.json()["data"]["id"]
    except Exception as e:
        print(f"[ERROR] Failed to create chat for pool: {e}")
        return None

async def pool_manager(internal_model: str):
    await asyncio.sleep(2)
    while True:
        try:
            current = await redis_client.llen(POOL_KEY)
            if current < MIN_POOL:
                needed = MAX_POOL - current
                tasks = [create_chat(internal_model) for _ in range(needed)]
                created = [cid for cid in await asyncio.gather(*tasks) if cid]
                if created:
                    await redis_client.rpush(POOL_KEY, *created)
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ERROR] Pool manager failed: {e}")
            await asyncio.sleep(30)

# La caché LRU en memoria no es ideal para este caso de uso,
# ya que el hash de conversación debe persistir entre reinicios del servidor.
# Usaremos Redis directamente, que es más robusto.
# @lru_cache(maxsize=1000)
# def _cached_hash_lookup(prompt_hash: str) -> Optional[str]:
#     import redis
#     r = redis.from_url(UPSTASH_REDIS_URL, decode_responses=True)
#     return r.get(f"conv_hash:{prompt_hash}")

# ---------- STREAMING ----------
BATCH_MS = 0.020
BATCH_TOK = 3
HEARTBEAT_SEC = 15

async def sse_stream(
    chat_id: str,
    state: ConversationState,
    prompt: str,
    model_name: str,
    cfg: dict,
) -> AsyncGenerator[str, None]:
    payload = {
        "stream": True,
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": cfg["internal_model_id"],
        "parent_id": state.last_parent_id,
        "messages": [
            {
                "fid": str(uuid.uuid4()),
                "parentId": state.last_parent_id,
                "role": "user",
                "content": prompt,
                "user_action": "chat",
                "files": [],
                "timestamp": int(time.time()),
                "models": [cfg["internal_model_id"]],
                "chat_type": "t2t",
                "feature_config": {"thinking_enabled": True, "output_schema": "phase", "thinking_budget": 81920},
                "extra": {"meta": {"subChatType": "t2t"}},
                "sub_chat_type": "t2t",
            }
        ],
        "timestamp": int(time.time()),
    }
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={chat_id}"
    headers = {**_build_headers(), "x-request-id": str(uuid.uuid4())}

    comp_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    yield ":\n\n"
    buffer = []
    last_flush = time.time()
    last_heartbeat = time.time()

    try:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    ev = JSON_DESERIALIZER(line)
                except Exception:
                    continue

                if ev.get("response.created"):
                    pid = ev["response.created"].get("response_id")
                    if pid:
                        # Actualizamos el estado con el nuevo parent_id
                        state.last_parent_id = pid
                        # Guardamos el estado actualizado en Redis para la siguiente petición
                        await redis_client.set(f"qwen_conv:{chat_id}", state.model_dump_json(), ex=86400)
                    continue

                delta = ev.get("choices", [{}])[0].get("delta", {})
                if cfg["filter_phase"] and delta.get("phase") != "answer":
                    continue
                if txt := delta.get("content"):
                    buffer.append(txt)

                now = time.time()
                if buffer and (len(buffer) >= BATCH_TOK or now - last_flush >= BATCH_MS):
                    chunk = {"id": comp_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"delta": {"content": "".join(buffer)}, "index": 0}]}
                    yield f"data: {JSON_SERIALIZER(chunk)}\n\n"
                    buffer.clear()
                    last_flush = now

                if now - last_heartbeat >= HEARTBEAT_SEC:
                    yield ":hb\n\n"
                    last_heartbeat = now

    except Exception as e:
        yield f'data: {{"error":"{e}"}}\n\n'

    if buffer:
        chunk = {"id": comp_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"delta": {"content": "".join(buffer)}, "index": 0}]}
        yield f"data: {JSON_SERIALIZER(chunk)}\n\n"

    yield f'data: {JSON_SERIALIZER({"id": comp_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\n'
    yield "data: [DONE]\n\n"

# ---------- FASTAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global lua_script
    lua_script = redis_client.register_script(LUA_GET_SET_EX)
    try:
        await client.head(QWEN_API_BASE_URL + "/health")
    except Exception:
        pass
    mgr = asyncio.create_task(pool_manager(next(iter(MODEL_CONFIG.values()))["internal_model_id"]))
    yield
    mgr.cancel()
    try: await mgr
    except asyncio.CancelledError: pass
    await client.aclose()

app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": k, "object": "model", "created": int(time.time()), "owned_by": "proxy"} for k in MODEL_CONFIG]}

# ==============================================================================
# --- ENDPOINT DE CHAT COMPLETIONS CORREGIDO ---
# ==============================================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")
    if model_name not in MODEL_CONFIG:
        raise HTTPException(status_code=404, detail="Model not found")

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="El campo 'messages' debe ser una lista no vacía.")

    # --- INICIO DE LA LÓGICA DE CONTEXTO CORREGIDA ---

    # 1. Usar el PRIMER mensaje para crear un hash estable que identifique toda la conversación.
    #    Este es el cambio clave para mantener la memoria entre turnos.
    try:
        first_user_message_content = messages[0].get("content", "")
        if not first_user_message_content:
             raise ValueError("El primer mensaje de la conversación no tiene contenido.")
        
        conversation_hash = hashlib.sha256(first_user_message_content.encode('utf-8')).hexdigest()
        redis_conv_hash_key = f"conv_hash:{conversation_hash}"
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Error procesando los mensajes para obtener el contexto: {e}")

    # 2. Buscar si ya existe un `chat_id` para esta conversación.
    chat_id = await redis_client.get(redis_conv_hash_key)
    state: ConversationState

    if not chat_id:
        print(f"✨ Nueva conversación detectada (hash: {conversation_hash[:8]}). Obteniendo ID del pool...")
        # Si no existe, es una conversación nueva. Tomamos un ID del pool.
        if await redis_client.llen(POOL_KEY) == 0:
            raise HTTPException(status_code=429, detail="Pool de chats vacío, por favor reintenta más tarde.")
        
        chat_id = await redis_client.lpop(POOL_KEY)
        if not chat_id:
            raise HTTPException(status_code=429, detail="Pool de chats vacío al intentar obtener un ID, por favor reintenta.")
        
        # Guardamos la asociación: hash de la conversación -> chat_id de Qwen
        await redis_client.set(redis_conv_hash_key, chat_id, ex=86400 * 7) # Persiste por 7 días
        print(f"✅ Conversación asociada: hash {conversation_hash[:8]} -> Qwen ID {chat_id[:8]}")
        # Creamos un estado inicial vacío (sin parent_id)
        state = ConversationState(last_parent_id=None)
    else:
        print(f"➡️ Conversación existente reconocida (hash: {conversation_hash[:8]}). Usando ID: {chat_id[:8]}...")
        # Si existe, cargamos su estado (que contendrá el `last_parent_id` de la última respuesta)
        state_json = await redis_client.get(f"qwen_conv:{chat_id}")
        state = ConversationState.model_validate_json(state_json or "{}")

    # 3. El prompt a enviar es siempre el último mensaje del historial.
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        raise HTTPException(status_code=400, detail="El último mensaje del historial debe tener el rol 'user'")
    
    prompt = last_msg.get("content", "")

    # --- FIN DE LA LÓGICA DE CONTEXTO CORREGIDA ---

    is_stream = body.get("stream", False)
    # Pasamos el `chat_id` y el `state` recuperado/creado a la función de streaming
    if is_stream:
        return StreamingResponse(
            sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]),
            media_type="text/event-stream",
            headers={"X-Conversation-ID": chat_id},
        )
    else:
        full_content = []
        # La función sse_stream se encargará de actualizar el estado en Redis internamente
        async for chunk in sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]):
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = JSON_DESERIALIZER(chunk[6:])
                    if data.get("choices"):
                        full_content.append(data["choices"][0]["delta"].get("content", ""))
                except Exception:
                    pass
        content = "".join(full_content)
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            headers={"X-Conversation-ID": chat_id},
        )

@app.get("/")
def root():
    return {"status": "OK", "message": f"{API_TITLE} is live"}

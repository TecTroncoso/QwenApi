import asyncio
import base64
import os
import time
import uuid
import xxhash
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
import logging

import httpx
import orjson
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Configurar logging para mejor diagnóstico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- SERIALIZATION ----------
def JSON_SERIALIZER(v, *, default=str):
    return orjson.dumps(v, default=default).decode()

JSON_DESERIALIZER = orjson.loads

# ---------- GLOBAL CONFIG ----------
load_dotenv()
API_TITLE = "Qwen Web API Proxy (Hyper-Optimized)"
API_VERSION = "15.2.0" # Versión actualizada con correcciones

MODEL_CONFIG = {
    "qwen3-235b-a22b": {"internal_model_id": "qwen3-235b-a22b", "filter_phase": True},
    "qwen3-235b-a22b-thinking": {"internal_model_id": "qwen3-235b-a22b", "filter_phase": False},
    "qwen3-coder": {"internal_model_id": "qwen3-coder-plus", "filter_phase": True},
    "qwen3-coder-flash": {"internal_model_id": "qwen3-coder-30b-a3b-instruct", "filter_phase": True},
    "qwen3-max": {"internal_model_id": "qwen3-max", "filter_phase": True},
    "qwen3-max-thinking": {"internal_model_id": "qwen3-max", "filter_phase": False},
    "qwen3-next": {"internal_model_id": "qwen-plus-2025-09-11", "filter_phase": True},
    "qwen3-next-thinking": {"internal_model_id": "qwen-plus-2025-09-11", "filter_phase": False},
    "qwen3-max-25-9-2025": {"internal_model_id": "qwen3-max", "filter_phase": True},
    "Qwen3-VL-235B-A22B": {"internal_model_id": "qwen3-vl-plus", "filter_phase": True},
    "Qwen3-VL-235B-A22B-thinking": {"internal_model_id": "qwen3-vl-plus", "filter_phase": False},
    "qwen3-omni-flash": {"internal_model_id": "qwen3-omni-flash", "filter_phase": True},
    "Qwen3-Omni-thinking": {"internal_model_id": "qwen3-omni-flash", "filter_phase": False},
}

QWEN_API_BASE_URL = "https://chat.qwen.ai/api/v2"
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
if not UPSTASH_REDIS_URL:
    raise RuntimeError("UPSTASH_REDIS_URL must be set on Render")

QWEN_AUTH_TOKEN_FALLBACK = os.getenv("QWEN_AUTH_TOKEN", "")
QWEN_COOKIES_JSON_B64 = os.getenv("QWEN_COOKIES_JSON_B64", "")

# ---------- COOKIES / TOKEN ----------
QWEN_AUTH_TOKEN: str = ""
QWEN_COOKIE_STRING: str = ""

def process_cookies_and_extract_token(b64_string: str | None):
    global QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING
    if not b64_string:
        logger.warning("Cookies var not found → using fallback token")
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK
        QWEN_COOKIE_STRING = ""
        return
    try:
        cookies_list = JSON_DESERIALIZER(base64.b64decode(b64_string))
        QWEN_COOKIE_STRING = "; ".join(f"{c['name']}={c['value']}" for c in cookies_list)
        token_value = next((c.get("value", "") for c in cookies_list if c.get("name") == "token"), "")
        QWEN_AUTH_TOKEN = f"Bearer {token_value}" if token_value else QWEN_AUTH_TOKEN_FALLBACK
        logger.info("✅ Cookies & token processed OK")
    except Exception as e:
        logger.error(f"[ERROR] Cookie parse: {e}")
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK
        QWEN_COOKIE_STRING = ""

process_cookies_and_extract_token(QWEN_COOKIES_JSON_B64)

# ---------- REDIS & LUA SCRIPT ----------
LUA_GET_OR_CREATE_CONV = """
-- KEYS[1]: La clave del hash de la conversación (e.g., "conv_hash:...")
-- KEYS[2]: La clave del pool de chat_ids (e.g., "qwen_chat_id_pool")
-- ARGV[1]: El TTL (tiempo de vida) para la nueva asociación de hash en segundos

local chat_id = redis.call('GET', KEYS[1])
if not chat_id then
    chat_id = redis.call('LPOP', KEYS[2])
    if chat_id then
        redis.call('SET', KEYS[1], chat_id, 'EX', ARGV[1])
        return {chat_id, "new"} -- Devuelve el ID y una marca de que es nuevo
    end
end
if chat_id then
    return {chat_id, "existing"} -- Devuelve el ID y una marca de que ya existía
end
return nil -- Devuelve nil si el pool está vacío y no se pudo crear
"""

redis_client = redis.from_url(UPSTASH_REDIS_URL, decode_responses=True)
get_or_create_conv_script = None # Se registrará en el lifespan

# ---------- MODELS & UTILS ----------
class ConversationState(BaseModel):
    last_parent_id: Optional[str] = None

def _build_headers() -> Dict[str, str]:
    # Headers actualizados según el comando curl proporcionado
    return {
        "Accept": "*/*",
        "Accept-Language": "es-AR,es;q=0.5",
        "Connection": "keep-alive",
        "Cookie": QWEN_COOKIE_STRING,
        "DNT": "1",
        "Origin": "https://chat.qwen.ai",
        "Referer": "https://chat.qwen.ai/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Sec-GPC": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "authorization": QWEN_AUTH_TOKEN,
        "content-type": "application/json; charset=UTF-8",
        "source": "web",
        "timezone": "Thu Oct 02 2025 08:47:14 GMT-0300",
        "x-accel-buffering": "no",
    }
    
# ---------- HTTPX ----------
client = httpx.AsyncClient(http2=True, timeout=60.0)

# ---------- POOL MANAGER ----------
MIN_POOL, MAX_POOL = 2, 5
POOL_KEY = "qwen_chat_id_pool"

async def create_chat(internal_model: str) -> Optional[str]:
    payload = {"title": "ProxyPool", "models": [internal_model], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
    try:
        r = await client.post(f"{QWEN_API_BASE_URL}/chats/new", json=payload, headers=_build_headers())
        r.raise_for_status()
        return r.json()["data"]["id"]
    except Exception as e:
        logger.error(f"[ERROR] Failed to create chat for pool: {e}")
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
                if created: await redis_client.rpush(POOL_KEY, *created)
            await asyncio.sleep(10)
        except asyncio.CancelledError: break
        except Exception as e:
            logger.error(f"[ERROR] Pool manager failed: {e}")
            await asyncio.sleep(30)

# ---------- STREAMING ----------
BATCH_MS = 0.015  # 15ms
BATCH_TOK = 2
HEARTBEAT_SEC = 15

async def sse_stream(chat_id: str, state: ConversationState, prompt: str, model_name: str, cfg: dict) -> AsyncGenerator[str, None]:
    # Payload actualizado según el formato del comando curl
    message_id = str(uuid.uuid4())
    payload = {
        "stream": True,
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": cfg["internal_model_id"],
        "parent_id": state.last_parent_id,
        "messages": [{
            "fid": message_id,
            "parentId": state.last_parent_id,
            "childrenIds": [],
            "role": "user",
            "content": prompt,
            "user_action": "chat",
            "files": [],
            "timestamp": int(time.time()),
            "models": [cfg["internal_model_id"]],
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": False,
                "output_schema": "phase"
            },
            "extra": {
                "meta": {
                    "subChatType": "t2t"
                }
            },
            "sub_chat_type": "t2t",
            "parent_id": state.last_parent_id
        }],
        "timestamp": int(time.time()),
    }
    
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={chat_id}"
    headers = {**_build_headers(), "x-request-id": str(uuid.uuid4())}
    comp_id, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    yield ":\n\n"
    buffer, last_flush, last_heartbeat = [], time.time(), time.time()
    has_content = False
    
    try:
        logger.info(f"Sending request to {url} with payload: {payload}")
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            logger.info(f"Response status: {resp.status_code}")
            
            async for line in resp.aiter_lines():
                if not line.startswith("data:"): continue
                line = line[5:].strip()
                if not line or line == "[DONE]": continue
                
                try: 
                    ev = JSON_DESERIALIZER(line)
                    logger.debug(f"Received event: {ev}")
                except Exception as e: 
                    logger.error(f"Error parsing event: {e}")
                    continue
                
                # Manejo de respuestas según el formato real de la API
                if "response" in ev:
                    response = ev["response"]
                    
                    # Actualizar el parent_id si está presente
                    if "response_id" in response:
                        state.last_parent_id = response["response_id"]
                        await redis_client.set(f"qwen_conv:{chat_id}", state.model_dump_json(), ex=86400)
                    
                    # Extraer contenido del texto de la respuesta
                    if "text" in response and response["text"]:
                        content = response["text"]
                        buffer.append(content)
                        has_content = True
                
                # Procesar el buffer si hay contenido
                now = time.time()
                if buffer and (len(buffer) >= BATCH_TOK or now - last_flush >= BATCH_MS):
                    chunk = {
                        "id": comp_id, 
                        "object": "chat.completion.chunk", 
                        "created": created, 
                        "model": model_name, 
                        "choices": [{
                            "delta": {"content": "".join(buffer)}, 
                            "index": 0
                        }]
                    }
                    yield f"data: {JSON_SERIALIZER(chunk)}\n\n"
                    buffer.clear()
                    last_flush = now
                
                # Enviar heartbeat si es necesario
                if now - last_heartbeat >= HEARTBEAT_SEC:
                    yield ":hb\n\n"
                    last_heartbeat = now
    except Exception as e: 
        logger.error(f"Error in stream: {e}")
        yield f'data: {{"error":"{e}"}}\n\n'
    
    # Si no hemos recibido contenido, intentamos proporcionar un mensaje de error más informativo
    if not has_content:
        logger.warning("No content received from stream")
        yield f'data: {{"error":"No content received from the model. This might be due to API changes or authentication issues."}}\n\n'
    
    # Enviar cualquier contenido restante en el buffer
    if buffer:
        chunk = {
            "id": comp_id, 
            "object": "chat.completion.chunk", 
            "created": created, 
            "model": model_name, 
            "choices": [{
                "delta": {"content": "".join(buffer)}, 
                "index": 0
            }]
        }
        yield f"data: {JSON_SERIALIZER(chunk)}\n\n"
    
    # Enviar el chunk final
    yield f'data: {JSON_SERIALIZER({"id": comp_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\n'
    yield "data: [DONE]\n\n"

# ---------- FASTAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global get_or_create_conv_script
    try:
        await redis_client.ping()
        get_or_create_conv_script = redis_client.register_script(LUA_GET_OR_CREATE_CONV)
        logger.info("✅ Redis connected & Lua script registered.")
    except Exception as e: 
        logger.error(f"❌ Redis connect or script registration failed: {e}")
        raise RuntimeError(f"❌ Redis connect or script registration failed: {e}") from e
    
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

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")
    if model_name not in MODEL_CONFIG: 
        logger.error(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail="Model not found")

    messages = body.get("messages")
    if not messages or not isinstance(messages, list): 
        logger.error("Invalid messages format")
        raise HTTPException(status_code=400, detail="Messages required")

    try:
        first_content = messages[0].get("content", "")
        conversation_hash = xxhash.xxh64(first_content.encode('utf-8')).hexdigest()
        redis_conv_hash_key = f"conv_hash:{conversation_hash}"
    except IndexError: 
        logger.error("Messages list is empty")
        raise HTTPException(status_code=400, detail="Messages list is empty")

    lua_result = await get_or_create_conv_script(
        keys=[redis_conv_hash_key, POOL_KEY],
        args=[86400 * 7] # 7-day TTL
    )

    if not lua_result:
        logger.error("Chat pool is empty and a new chat could not be retrieved")
        raise HTTPException(status_code=503, detail="Chat pool is empty and a new chat could not be retrieved. Please try again later.")
    
    chat_id, conv_status = lua_result
    state: ConversationState

    if conv_status == "new":
        logger.info(f"✨ New conversation (hash: {conversation_hash[:8]}) -> assigned ID: {chat_id[:8]}")
        state = ConversationState(last_parent_id=None)
    else: # conv_status == "existing"
        logger.info(f"➡️ Existing conversation (hash: {conversation_hash[:8]}) -> using ID: {chat_id[:8]}")
        state_json = await redis_client.get(f"qwen_conv:{chat_id}")
        state = ConversationState.model_validate_json(state_json or "{}")

    last_msg = messages[-1]
    if last_msg.get("role") != "user": 
        logger.error("Last message is not from user")
        raise HTTPException(status_code=400, detail="Last message must be user")
    
    prompt = last_msg.get("content", "")
    is_stream = body.get("stream", False)
    
    if is_stream:
        return StreamingResponse(
            sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]),
            media_type="text/event-stream",
            headers={"X-Conversation-ID": chat_id, "Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        full_content = []
        async for chunk in sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]):
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = JSON_DESERIALIZER(chunk[6:])
                    if data.get("choices"):
                        full_content.append(data["choices"][0]["delta"].get("content", ""))
                except Exception as e: 
                    logger.error(f"Error parsing chunk: {e}")
                    pass
        content = "".join(full_content)
        
        # Si no hay contenido, devolver un error más descriptivo
        if not content:
            logger.error("No content received in non-streaming mode")
            return JSONResponse(
                {
                    "error": {
                        "message": "No content received from the model. This might be due to API changes or authentication issues.",
                        "type": "api_error",
                        "code": "no_content"
                    }
                },
                status_code=500,
                headers={"X-Conversation-ID": chat_id},
            )
        
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            headers={"X-Conversation-ID": chat_id},
        )

@app.get("/")
def root():
    return {"status": "OK", "message": f"{API_TITLE} is live"}

# Nuevo endpoint para verificar el estado de la API
@app.get("/health")
async def health_check():
    try:
        # Verificar conexión con Redis
        await redis_client.ping()
        
        # Verificar pool de conversaciones
        pool_size = await redis_client.llen(POOL_KEY)
        
        # Verificar token de autenticación
        auth_status = "valid" if QWEN_AUTH_TOKEN else "missing"
        
        return {
            "status": "healthy",
            "redis": "connected",
            "pool_size": pool_size,
            "auth_status": auth_status,
            "models": list(MODEL_CONFIG.keys())
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            {
                "status": "unhealthy",
                "error": str(e)
            },
            status_code=503
        )

# Endpoint para debuggear la configuración actual
@app.get("/debug/config")
def debug_config():
    return {
        "api_base_url": QWEN_API_BASE_URL,
        "auth_token_set": bool(QWEN_AUTH_TOKEN),
        "cookies_set": bool(QWEN_COOKIE_STRING),
        "model_config": MODEL_CONFIG,
        "redis_url_set": bool(UPSTASH_REDIS_URL),
    }

# Endpoint para probar la conexión con la API de Qwen
@app.post("/debug/test-qwen-connection")
async def test_qwen_connection():
    try:
        # Intentar crear un chat de prueba
        test_model = next(iter(MODEL_CONFIG.values()))["internal_model_id"]
        chat_id = await create_chat(test_model)
        
        if chat_id:
            return {
                "status": "success",
                "message": "Successfully connected to Qwen API",
                "test_chat_id": chat_id
            }
        else:
            return {
                "status": "error",
                "message": "Failed to create test chat"
            }
    except Exception as e:
        logger.error(f"Qwen connection test failed: {e}")
        return {
            "status": "error",
            "message": f"Connection test failed: {str(e)}"
        }

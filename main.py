import asyncio
import base64
import os
import socket
import time
import uuid
import xxhash  # <--- MEJORA: xxhash es mucho más rápido que hashlib para hashing no criptográfico
from contextlib import asynccontextmanager
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
API_TITLE = "Qwen Web API Proxy (Hyper-Optimized)"
API_VERSION = "15.0.0" # Versión con optimizaciones de latencia avanzada

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

# ---------- REDIS & LUA SCRIPT ----------
# MEJORA: Este script Lua se ejecuta atómicamente en Redis, reduciendo 3 llamadas de red a 1
# para conversaciones nuevas. Busca un chat_id por hash. Si no lo encuentra, lo toma del pool
# y establece la asociación, todo en una sola operación.
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
    # ... (sin cambios)
    return {
        "Accept": "application/json", "Content-Type": "application/json; charset=UTF-8",
        "Authorization": QWEN_AUTH_TOKEN, "Cookie": QWEN_COOKIE_STRING, "Origin": "https://chat.qwen.ai",
        "Referer": "https://chat.qwen.ai/", "User-Agent": "Mozilla/5.0", "source": "web", "x-accel-buffering": "no",
    }
    
# ---------- HTTPX ----------
client = httpx.AsyncClient(http2=True)

# ---------- POOL MANAGER ----------
MIN_POOL, MAX_POOL = 2, 5
POOL_KEY = "qwen_chat_id_pool"

async def create_chat(internal_model: str) -> Optional[str]:
    # ... (sin cambios)
    payload = {"title": "ProxyPool", "models": [internal_model], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
    try:
        r = await client.post(f"{QWEN_API_BASE_URL}/chats/new", json=payload, headers=_build_headers())
        r.raise_for_status()
        return r.json()["data"]["id"]
    except Exception as e:
        print(f"[ERROR] Failed to create chat for pool: {e}")
        return None

async def pool_manager(internal_model: str):
    # ... (sin cambios)
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
            print(f"[ERROR] Pool manager failed: {e}")
            await asyncio.sleep(30)

# ---------- STREAMING ----------
# MEJORA: Valores ajustados para una latencia percibida aún menor (Time To First Byte).
BATCH_MS = 0.015  # 15ms
BATCH_TOK = 2
HEARTBEAT_SEC = 15

async def sse_stream(chat_id: str, state: ConversationState, prompt: str, model_name: str, cfg: dict) -> AsyncGenerator[str, None]:
    # ... (lógica interna sin cambios significativos, solo la actualización de estado en Redis)
    payload = {
        "stream": True, "incremental_output": True, "chat_id": chat_id, "chat_mode": "normal",
        "model": cfg["internal_model_id"], "parent_id": state.last_parent_id,
        "messages": [{"fid": str(uuid.uuid4()), "parentId": state.last_parent_id, "role": "user", "content": prompt, "user_action": "chat", "files": [], "timestamp": int(time.time()), "models": [cfg["internal_model_id"]], "chat_type": "t2t", "feature_config": {"thinking_enabled": True, "output_schema": "phase", "thinking_budget": 81920}, "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t"}],
        "timestamp": int(time.time()),
    }
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={chat_id}"
    headers = {**_build_headers(), "x-request-id": str(uuid.uuid4())}
    comp_id, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    yield ":\n\n"
    buffer, last_flush, last_heartbeat = [], time.time(), time.time()
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"): continue
                line = line[5:].strip()
                if not line or line == "[DONE]": continue
                try: ev = JSON_DESERIALIZER(line)
                except Exception: continue
                if ev.get("response.created"):
                    if pid := ev["response.created"].get("response_id"):
                        state.last_parent_id = pid
                        await redis_client.set(f"qwen_conv:{chat_id}", state.model_dump_json(), ex=86400)
                    continue
                delta = ev.get("choices", [{}])[0].get("delta", {})
                if cfg["filter_phase"] and delta.get("phase") != "answer": continue
                if txt := delta.get("content"): buffer.append(txt)
                now = time.time()
                if buffer and (len(buffer) >= BATCH_TOK or now - last_flush >= BATCH_MS):
                    chunk = {"id": comp_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"delta": {"content": "".join(buffer)}, "index": 0}]}
                    yield f"data: {JSON_SERIALIZER(chunk)}\n\n"
                    buffer.clear()
                    last_flush = now
                if now - last_heartbeat >= HEARTBEAT_SEC:
                    yield ":hb\n\n"
                    last_heartbeat = now
    except Exception as e: yield f'data: {{"error":"{e}"}}\n\n'
    if buffer:
        chunk = {"id": comp_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"delta": {"content": "".join(buffer)}, "index": 0}]}
        yield f"data: {JSON_SERIALIZER(chunk)}\n\n"
    yield f'data: {JSON_SERIALIZER({"id": comp_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\n'
    yield "data: [DONE]\n\n"

# ---------- FASTAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global get_or_create_conv_script
    try:
        await redis_client.ping()
        get_or_create_conv_script = redis_client.register_script(LUA_GET_OR_CREATE_CONV)
        print("✅ Redis connected & Lua script registered.")
    except Exception as e: raise RuntimeError(f"❌ Redis connect or script registration failed: {e}") from e
    
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
    if model_name not in MODEL_CONFIG: raise HTTPException(status_code=404, detail="Model not found")

    messages = body.get("messages")
    if not messages or not isinstance(messages, list): raise HTTPException(status_code=400, detail="Messages required")

    try:
        first_content = messages[0].get("content", "")
        # MEJORA: xxhash.xxh64 es extremadamente rápido.
        conversation_hash = xxhash.xxh64(first_content.encode('utf-8')).hexdigest()
        redis_conv_hash_key = f"conv_hash:{conversation_hash}"
    except IndexError: raise HTTPException(status_code=400, detail="Messages list is empty")

    # MEJORA: Una sola llamada a Redis para manejar la lógica de la conversación.
    lua_result = await get_or_create_conv_script(
        keys=[redis_conv_hash_key, POOL_KEY],
        args=[86400 * 7] # 7-day TTL
    )

    if not lua_result:
        raise HTTPException(status_code=503, detail="Chat pool is empty and a new chat could not be retrieved. Please try again later.")
    
    chat_id, conv_status = lua_result
    state: ConversationState

    if conv_status == "new":
        print(f"✨ New conversation (hash: {conversation_hash[:8]}) -> assigned ID: {chat_id[:8]}")
        state = ConversationState(last_parent_id=None)
    else: # conv_status == "existing"
        print(f"➡️ Existing conversation (hash: {conversation_hash[:8]}) -> using ID: {chat_id[:8]}")
        state_json = await redis_client.get(f"qwen_conv:{chat_id}")
        state = ConversationState.model_validate_json(state_json or "{}")

    last_msg = messages[-1]
    if last_msg.get("role") != "user": raise HTTPException(status_code=400, detail="Last message must be user")
    
    prompt = last_msg.get("content", "")
    is_stream = body.get("stream", False)
    
    if is_stream:
        return StreamingResponse(
            sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]),
            media_type="text/event-stream",
            headers={"X-Conversation-ID": chat_id},
        )
    else:
        full_content = []
        async for chunk in sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]):
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = JSON_DESERIALIZER(chunk[6:])
                    if data.get("choices"):
                        full_content.append(data["choices"][0]["delta"].get("content", ""))
                except Exception: pass
        content = "".join(full_content)
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

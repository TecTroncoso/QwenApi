# main_memory_fixed.py – 100 % optimizado + memoria persistente + sin coroutines en headers
# ==============================================================================
import asyncio
import base64
import os
import socket
import time
import uuid
import xxhash
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
API_TITLE = "Qwen Web API Proxy (MEMORY-FIXED)"
API_VERSION = "15.0.0"

MODEL_CONFIG = {
    "qwen-final": {"internal_model_id": "qwen3-235b-a22b", "filter_phase": True},
    "qwen-thinking": {"internal_model_id": "qwen3-235b-a22b", "filter_phase": False},
    "qwen-coder-plus": {"internal_model_id": "qwen3-coder-plus", "filter_phase": True},
    "qwen-coder-30b": {"internal_model_id": "qwen3-coder-30b-a3b-instruct", "filter_phase": True},
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
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK
        QWEN_COOKIE_STRING = ""
        return
    try:
        cookies_list = JSON_DESERIALIZER(base64.b64decode(b64_string))
        QWEN_COOKIE_STRING = "; ".join(f"{c['name']}={c['value']}" for c in cookies_list)
        token_value = next((c.get("value", "") for c in cookies_list if c.get("name") == "token"), "")
        QWEN_AUTH_TOKEN = f"Bearer {token_value}" if token_value else QWEN_AUTH_TOKEN_FALLBACK
    except Exception:
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK
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
except Exception as e:
    raise RuntimeError(f"Redis connect error: {e}") from e

# ---------- MODELS ----------
class OpenAIMessage(BaseModel):
    role: str
    content: str

class ConversationState(BaseModel):
    last_parent_id: Optional[str] = None

# ---------- UTILS ----------
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
    except Exception:
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
                    pipe = redis_client.pipeline()
                    pipe.rpush(POOL_KEY, *created)
                    await pipe.execute()
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(30)

@lru_cache(maxsize=1000)
def _cached_hash_lookup(prompt_hash: str) -> Optional[str]:
    r = redis.from_url(UPSTASH_REDIS_URL, decode_responses=True)
    return r.get(f"conv_hash:{prompt_hash}")

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
                        state.last_parent_id = pid
                        await redis_client.set(
                            f"qwen_conv:{chat_id}",
                            state.model_dump_json(),
                            ex=86400
                        )
                    continue

                delta = ev.get("choices", [{}])[0].get("delta", {})
                if cfg["filter_phase"] and delta.get("phase") != "answer":
                    continue
                if txt := delta.get("content"):
                    buffer.append(txt)

                now = time.time()
                if buffer and (len(buffer) >= BATCH_TOK or now - last_flush >= BATCH_MS):
                    chunk = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"delta": {"content": "".join(buffer)}, "index": 0}],
                    }
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
    await client.head(QWEN_API_BASE_URL + "/health")  # TLS pre-warm
    mgr = asyncio.create_task(pool_manager(next(iter(MODEL_CONFIG.values()))["internal_model_id"]))
    yield
    mgr.cancel()
    try:
        await mgr
    except asyncio.CancelledError:
        pass
    await client.aclose()

app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "created": int(time.time()), "owned_by": "proxy"}
            for k in MODEL_CONFIG
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")
    if model_name not in MODEL_CONFIG:
        raise HTTPException(status_code=404, detail="Model not found")
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="Messages required")
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        raise HTTPException(status_code=400, detail="Last message must be user")

    prompt = last_msg.get("content", "")
    loop = asyncio.get_running_loop()
    prompt_hash = await loop.run_in_executor(None, xxhash.xxh64, prompt.encode())
    prompt_hash = prompt_hash.hexdigest()

    chat_id = _cached_hash_lookup(prompt_hash)
    if not chat_id:
        llen = await redis_client.llen(POOL_KEY)
        if llen == 0:
            raise HTTPException(status_code=429, detail="Pool empty, retry later")
        chat_id = await redis_client.lpop(POOL_KEY)
        if not chat_id:
            raise HTTPException(status_code=429, detail="Pool empty, retry later")
        await redis_client.set(f"conv_hash:{prompt_hash}", chat_id, ex=86400 * 7)

    # Recuperar / inicializar estado
    state_json = await redis_client.get(f"qwen_conv:{chat_id}")
    state = ConversationState.model_validate_json(state_json or "{}")

    is_stream = body.get("stream", False)
    if is_stream:
        return StreamingResponse(
            sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]),
            media_type="text/event-stream",
            headers={"X-Conversation-ID": str(chat_id)},
        )
    else:
        full = []
        async for chunk in sse_stream(chat_id, state, prompt, model_name, MODEL_CONFIG[model_name]):
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = JSON_DESERIALIZER(chunk[6:])
                    if data.get("choices"):
                        full.append(data["choices"][0]["delta"].get("content", ""))
                except Exception:
                    pass
        content = "".join(full)
        # Persistir estado ANTES de devolver la respuesta
        await redis_client.set(
            f"qwen_conv:{chat_id}",
            state.model_dump_json(),
            ex=86400
        )
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            headers={"X-Conversation-ID": str(chat_id)},
        )

@app.get("/")
def root():
    return {"status": "OK", "message": f"{API_TITLE} is live"}

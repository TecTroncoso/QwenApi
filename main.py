import json
import time
import uuid
import httpx
import os
import base64
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- Carga Dinámica de Cookies desde Variables de Entorno ---

def format_cookies_from_json_string(json_string: str) -> str:
    """
    Toma un string JSON (que contiene una lista de objetos de cookies de Selenium),
    lo parsea y lo formatea en un único string de header 'Cookie'.
    Ejemplo de formato de salida: 'name1=value1; name2=value2; ...'
    """
    try:
        cookies_list = json.loads(json_string)
        if not isinstance(cookies_list, list):
            raise TypeError("El JSON de cookies no es una lista.")
            
        # Formatea cada cookie como 'name=value' y las une con '; '
        cookie_string = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies_list])
        return cookie_string
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Error al procesar el JSON de las cookies: {e}")
        raise ValueError("El formato del JSON de cookies es inválido.") from e

# 1. Obtener la variable de entorno con las cookies en Base64.
#    En Render, debes crear una variable de entorno llamada QWEN_COOKIES_JSON
#    y pegar el contenido Base64 que tienes.
encoded_cookies = os.getenv("QWEN_COOKIES_JSON")

if not encoded_cookies:
    raise ValueError("La variable de entorno 'QWEN_COOKIES_JSON' no está definida. La API no puede funcionar sin cookies.")

try:
    # 2. Decodificar de Base64 a un string de texto (que es el JSON).
    decoded_json_string = base64.b64decode(encoded_cookies).decode('utf-8')

    # 3. Formatear el JSON en el string de cookie que necesita el header.
    dynamic_cookie_header = format_cookies_from_json_string(decoded_json_string)
    print("✅ Cookies cargadas y formateadas exitosamente desde la variable de entorno.")

except Exception as e:
    print(f"❌ Error crítico al cargar/decodificar las cookies: {e}")
    # Si falla, la aplicación no debe iniciar.
    raise

# 4. Construir los headers usando la cookie dinámica.
QWEN_HEADERS = {
    "Accept": "text/event-stream",
    "Accept-Language": "es-AR,es;q=0.8",
    "Cookie": dynamic_cookie_header,  # <-- ¡AQUÍ USAMOS LA COOKIE DINÁMICA!
    "Origin": "https://chat.qwen.ai",
    "Referer": "https://chat.qwen.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Content-Type": "application/json; charset=UTF-8",
    "X-Accel-Buffering": "no",
}

# --- El resto del código permanece exactamente igual ---

QWEN_BASE_URL = "https://chat.qwen.ai/api/v2"

chat_session: Dict[str, Optional[str]] = {
    "chat_id": None,
    "parent_id": None,
}

app = FastAPI(
    title="Qwen API Proxy",
    description="Un proxy compatible con OpenAI para la API no oficial de Qwen Chat."
)

# --- Mapeo de Modelos y Endpoint /v1/models ---
MODEL_MAPPING = {
    "Qwen3-Max": "qwen3-max",
    "Qwen3-VL-235B-A22B": "qwen3-vl-plus",
    "Qwen3-Coder": "qwen3-coder-plus",
    "Qwen3-VL-32B": "qwen3-vl-32b",
}

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "qwen-proxy"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Endpoint para listar los modelos disponibles, compatible con la API de OpenAI."""
    model_cards = [ModelCard(id=model_id) for model_id in MODEL_MAPPING.keys()]
    return ModelList(data=model_cards)

# --- Modelos Pydantic para Chat ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

# --- Lógica del Proxy ---
async def create_new_chat_session(client: httpx.AsyncClient, model_name: str) -> Optional[str]:
    # ... (el resto de tu código no necesita cambios)
    """Crea una nueva sesión de chat en Qwen para el modelo especificado."""
    payload = {
        "title": "Nuevo Chat",
        "models": [model_name],
        "chat_mode": "normal",
        "chat_type": "t2t",
        "timestamp": int(time.time() * 1000)
    }
    try:
        response = await client.post(f"{QWEN_BASE_URL}/chats/new", json=payload, headers=QWEN_HEADERS)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            chat_id = data.get("data", {}).get("id")
            print(f"Nueva sesión de chat creada con ID: {chat_id} para el modelo {model_name}")
            return chat_id
    except httpx.HTTPStatusError as e:
        print(f"Error al crear la sesión de chat: {e.response.text}")
    return None

async def stream_generator(request: ChatCompletionRequest):
    # ... (el resto de tu código no necesita cambios)
    """Generador que maneja el flujo de chat, con validación estricta de modelos."""
    global chat_session

    requested_model = request.model
    if requested_model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"El modelo '{requested_model}' no es válido. Por favor, elige uno de los siguientes: {list(MODEL_MAPPING.keys())}"
        )
    
    qwen_model_name = MODEL_MAPPING[requested_model]
    print(f"Cliente solicitó '{requested_model}', usando modelo Qwen '{qwen_model_name}'.")

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            if not chat_session.get("chat_id"):
                chat_id = await create_new_chat_session(client, model_name=qwen_model_name)
                if not chat_id:
                    error_chunk = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": "Error: No se pudo crear una nueva sesión de chat en Qwen."}, "finish_reason": "error"}]}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                chat_session["chat_id"] = chat_id
                chat_session["parent_id"] = None

            current_chat_id = chat_session["chat_id"]
            current_parent_id = chat_session["parent_id"]
            last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), "")

            if not last_user_message:
                 raise HTTPException(status_code=400, detail="No se encontró un mensaje de usuario en la solicitud.")

            completion_payload = {
                "stream": True, "incremental_output": True, "chat_id": current_chat_id, "chat_mode": "normal",
                "model": qwen_model_name,
                "parent_id": current_parent_id,
                "messages": [{
                    "fid": str(uuid.uuid4()), "parentId": current_parent_id, "childrenIds": [str(uuid.uuid4())], "role": "user", "content": last_user_message,
                    "user_action": "chat", "files": [], "timestamp": int(time.time()), "models": [qwen_model_name],
                    "chat_type": "t2t", "feature_config": {"thinking_enabled": False, "output_schema": "phase"},
                    "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t", "parent_id": current_parent_id
                }],
                "timestamp": int(time.time())
            }
            url = f"{QWEN_BASE_URL}/chat/completions?chat_id={current_chat_id}"
            
            is_first_chunk = True
            async with client.stream("POST", url, json=completion_payload, headers=QWEN_HEADERS) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=f"Error de la API de Qwen: {error_text.decode()}")

                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        if not data_str: continue
                        try: data = json.loads(data_str)
                        except json.JSONDecodeError: continue
                        if "response.created" in data:
                            new_parent_id = data["response.created"].get("response_id")
                            if new_parent_id:
                                chat_session["parent_id"] = new_parent_id
                                print(f"Parent ID actualizado a: {new_parent_id}")
                            continue
                        if "choices" in data:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            openai_delta = {}
                            finish_reason = None
                            if is_first_chunk and content:
                                openai_delta["role"] = "assistant"
                                is_first_chunk = False
                            if content:
                                openai_delta["content"] = content
                            if delta.get("status") == "finished":
                                finish_reason = "stop"
                            if openai_delta:
                                openai_chunk = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": openai_delta, "finish_reason": None}]}
                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                            if finish_reason:
                                final_chunk = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
                                yield f"data: {json.dumps(final_chunk)}\n\n"
                                break
            
            yield "data: [DONE]\n\n"
    except (httpx.HTTPStatusError, HTTPException) as e:
        error_detail = getattr(e, 'detail', str(e))
        error_message = f"Error en la API: {error_detail}"
        print(error_message)
        error_chunk = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "error"}]}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_message = f"Error interno del proxy: {type(e).__name__} - {str(e)}"
        print(error_message)
        error_chunk = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "error"}]}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.stream:
        raise HTTPException(status_code=400, detail="Esta implementación solo soporta streaming (stream=True).")
    return StreamingResponse(stream_generator(request), media_type="text/event-stream")

@app.get("/reset")
async def reset_session():
    global chat_session
    chat_session["chat_id"] = None
    chat_session["parent_id"] = None
    print("Sesión de chat reiniciada.")
    return {"message": "Sesión de chat reiniciada."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

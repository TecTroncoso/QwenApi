import json
import time
import uuid
import httpx
import os      # <--- Importado
import base64  # <--- Importado
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- FUNCIÓN PARA CARGAR Y FORMATEAR COOKIES DESDE ENV ---
def load_and_format_cookies() -> str:
    """
    Carga el JSON de cookies desde una variable de entorno Base64,
    lo decodifica y lo formatea en una cadena de header 'Cookie'.
    """
    b64_cookies = os.getenv("QWEN_COOKIES_JSON_B64")
    if not b64_cookies:
        raise RuntimeError(
            "La variable de entorno 'QWEN_COOKIES_JSON_B64' no está definida o está vacía. "
            "Asegúrate de configurar esta variable con las cookies codificadas en Base64."
        )

    try:
        # 1. Decodificar de Base64 a bytes, y luego a string UTF-8
        decoded_cookies_json = base64.b64decode(b64_cookies).decode('utf-8')
        
        # 2. Cargar el string JSON en una lista de Python
        cookies_list = json.loads(decoded_cookies_json)
        
        # 3. Formatear la lista en una sola cadena para el header
        # Ejemplo: [{'name': 'token', 'value': 'abc'}, {'name': 'id', 'value': '123'}]
        # se convierte en -> "token=abc; id=123"
        cookie_string = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies_list])
        
        print("✅ Cookies cargadas y formateadas exitosamente desde la variable de entorno.")
        return cookie_string

    except (base64.binascii.Error, json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"❌ Error al decodificar, parsear o formatear las cookies: {e}")
        raise RuntimeError(
            "El valor de 'QWEN_COOKIES_JSON_B64' es inválido. "
            "Debe ser una cadena Base64 que decodifique a un JSON válido con la estructura de cookies de Selenium."
        )

# --- Configuración ---
QWEN_HEADERS = {
    "Accept": "text/event-stream",
    "Accept-Language": "es-AR,es;q=0.8",
    # La cookie hardcodeada se elimina. Se cargará dinámicamente a continuación.
    "Origin": "https://chat.qwen.ai",
    "Referer": "https://chat.qwen.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Content-Type": "application/json; charset=UTF-8",
    "X-Accel-Buffering": "no",
}

# --- Carga dinámica de las cookies al iniciar la aplicación ---
try:
    # Llenamos dinámicamente el header 'Cookie' al arrancar la app
    QWEN_HEADERS["Cookie"] = load_and_format_cookies()
except RuntimeError as e:
    # Si las cookies no se pueden cargar, la aplicación no debe iniciar.
    # Esto te avisará en los logs de Render si algo está mal configurado.
    print(f"FATAL: {e}")
    exit(1)

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
        print(f"Error al crear la sesión de chat: {e.response.status_code} - {e.response.text}")
    return None

async def stream_generator(request: ChatCompletionRequest):
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
                    error_chunk = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": {"content": "Error: No se pudo crear una nueva sesión de chat en Qwen. Revisa las cookies o el estado del servicio."}, "finish_reason": "error"}]}
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
    """Reinicia la sesión de chat actual, forzando la creación de una nueva en la siguiente petición."""
    global chat_session
    chat_session["chat_id"] = None
    chat_session["parent_id"] = None
    print("Sesión de chat reiniciada manualmente.")
    return {"message": "Sesión de chat reiniciada."}

if __name__ == "__main__":
    import uvicorn
    # Si vas a nombrar el archivo 'api.py', el comando para ejecutarlo es:
    # uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

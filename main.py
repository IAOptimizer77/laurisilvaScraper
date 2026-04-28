import os
import time
import uuid
import logging
import json
import warnings
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

# ============================================================
# 🔧 Configuración
# ============================================================

APP_NAME = os.getenv("APP_NAME", "agente-laurisilva")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook/retell_rag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_LAURISILVA    = os.getenv("QDRANT_COLLECTION_LAURISILVA", "laurisilva_productos")
QDRANT_COLLECTION_VENTANA       = os.getenv("QDRANT_COLLECTION_VENTANA", "ventana_natural_productos")

TIENDA_COLLECTION_MAP = {
    "laurisilvabio": QDRANT_COLLECTION_LAURISILVA,
    "LauriSilvaBio": QDRANT_COLLECTION_LAURISILVA,
    "laurisilva":    QDRANT_COLLECTION_LAURISILVA,
    "La Ventana Natural": QDRANT_COLLECTION_VENTANA,
    "ventana natural":    QDRANT_COLLECTION_VENTANA,
    "ventana":            QDRANT_COLLECTION_VENTANA,
}
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "4.0"))
QDRANT_TOP_K = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "10.0"))

# ============================================================
# 🪵 Logger
# ============================================================

class TraceFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "trace_id"): record.trace_id = "SYSTEM"
        return True

logger = logging.getLogger("main")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | [%(trace_id)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addFilter(TraceFilter())
logger.setLevel(logging.INFO)

# ============================================================
# 🧠 Clientes
# ============================================================

app = FastAPI(title=APP_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
_qdrant_client = None

def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
            _qdrant_client.get_collections()
        except Exception:
            _qdrant_client = None  # reset — permite reintentar en el siguiente call
            return None
    return _qdrant_client

# ============================================================
# 🚀 RAG - Motor de Comparativa (Multiproducto)
# ============================================================

async def process_rag(query: str, trace_id: str, tienda: str = "") -> Dict[str, Any]:
    start_time = time.time()
    collection = TIENDA_COLLECTION_MAP.get(tienda, QDRANT_COLLECTION_LAURISILVA)
    logger.info(f"🔎 QUERY: {query} | TIENDA: {tienda} | COLECCIÓN: {collection}", extra={"trace_id": trace_id})

    try:
        # 1. Búsqueda
        emb_res = openai_client.embeddings.create(input=query, model="text-embedding-3-small", timeout=OPENAI_TIMEOUT)
        vector = emb_res.data[0].embedding

        payloads = []
        q_client = get_qdrant()
        if q_client:
            hits = q_client.search(collection_name=collection, query_vector=vector, limit=QDRANT_TOP_K)
            payloads = [hit.payload.get("metadata", hit.payload) for hit in hits]
        
        logger.info(f"📦 PRODUCTOS ENCONTRADOS: {len(payloads)}", extra={"trace_id": trace_id})

        # 2. Formatear Contexto
        context_parts = []
        for m in payloads:
            nombre = m.get("NombreProducto", "N/A")
            precio = m.get("Precio", "N/A")
            marca = m.get("Marca", "N/A")
            info = m.get("InformacionProducto", "")
            context_parts.append(f"PRODUCTO: {nombre}\nMARCA: {marca}\nPRECIO: {precio}\nDESCRIPCION: {info}\n---")

        context_text = "\n".join(context_parts) if context_parts else "SIN STOCK DISPONIBLE."

        # 3. Prompt (Optimizado para alternativas múltiples)
        system_prompt = (
            "Eres el asistente de LaurisilvaBio para llamadas telefónicas.\n\n"
            "PROCESO:\n"
            "1. Buscar productos con Info_Laurisilva usando la consulta del usuario.\n"
            "2. Seleccionar las 2 o 3 mejores alternativas si existen.\n"
            "3. Devolver inmediatamente un JSON.\n\n"
            "FORMATO DE SALIDA (JSON obligatorio):\n"
            "{\n"
            "  \"client_say\": \"Respuesta natural mencionando brevemente las alternativas y sus precios (máx 30 palabras)\",\n"
            "  \"product_description\": \"Lista numerada con la descripción técnica completa de cada producto mencionado\"\n"
            "}\n\n"
            "REGLAS client_say:\n"
            "- Menciona marca y precio de cada opción de forma fluida.\n"
            "- Números en texto: ej. 'veintiséis euros con noventa y tres céntimos'.\n"
            "- Usa 'céntimos', nunca 'centavos'.\n"
            "- Sin símbolos: €, %, mg.\n\n"
            "REGLAS product_description:\n"
            "- Debe listar los mismos productos que en client_say en orden (1, 2, 3).\n"
            "- Incluye el nombre completo, detalles técnicos y precio en formato numérico.\n\n"
            "IMPORTANTE: Devuelve solo el JSON."
        )

        # 4. LLM
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            timeout=OPENAI_TIMEOUT,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"DATOS DE INFO_LAURISILVA:\n{context_text}\n\nCONSULTA: {query}"}
            ]
        )

        res = json.loads(completion.choices[0].message.content)
        
        logger.info(f"💬 RESPUESTA: {res.get('client_say')}", extra={"trace_id": trace_id})
        logger.info(f"⏱️ TIEMPO: {time.time()-start_time:.2f}s", extra={"trace_id": trace_id})
        
        return res

    except Exception as e:
        logger.error(f"Error: {e}", extra={"trace_id": trace_id})
        return {
            "client_say": "Lo siento, tengo problemas para acceder al catálogo. ¿Te puedo ayudar con algo más?",
            "product_description": "Error en el servidor RAG."
        }

# ============================================================
# 📍 Endpoints
# ============================================================

@app.post(WEBHOOK_PATH)
async def webhook_rag(request: Request):
    tid = str(uuid.uuid4())[:8]
    try:
        data = await request.json()
        args = data.get("args", {})
        query = args.get("currentHerbolarioQuery") or data.get("query") or ""
        tienda = args.get("Tienda") or args.get("tienda") or ""
        if not query: return JSONResponse({"error": "No query"}, status_code=400)
        return await process_rag(query, tid, tienda)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/status")
def status():
    return {"status": "ok", "qdrant": get_qdrant() is not None}

@app.post("/admin/scrape")
async def trigger_scrape(request: Request):
    """Endpoint para lanzar el scraper desde n8n u otro sistema externo."""
    import asyncio, subprocess
    data = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    tienda = data.get("tienda", "all")
    allowed = {"all", "ventana_natural", "laurisilva"}
    if tienda not in allowed:
        return JSONResponse({"error": f"tienda debe ser uno de: {allowed}"}, status_code=400)
    subprocess.Popen(["python", "scraper.py", tienda])
    return {"status": "scraper iniciado", "tienda": tienda}
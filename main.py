import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import ejecutar_rag

app = FastAPI(title="Agente Llama3 RAG")

# Modelo para recibir preguntas
class Query(BaseModel):
    pregunta: str

# Endpoint POST con manejo de errores
@app.post("/query")
def api_query(q: Query):
    try:
        respuesta = ejecutar_rag(q.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        # Log del error para diagnóstico
        return {"error": str(e)}

# Endpoint base para comprobar estado
@app.get("/")
def root():
    return {"mensaje": "Agente RAG en línea y funcionando correctamente."}

# Ejecución local con uvicorn (necesario para Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

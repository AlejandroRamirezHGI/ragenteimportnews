import os 
from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import ejecutar_rag

app = FastAPI(title="Agente Llama3 RAG")

class Query(BaseModel):
    pregunta: str

@app.post("/query")
def api_query(q: Query):
    return {"respuesta": ejecutar_rag(q.pregunta)}

@app.get("/")
def root():
    return {"mensaje": "Agente RAG en línea y funcionando correctamente."}

# Este bloque es clave para Render:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))



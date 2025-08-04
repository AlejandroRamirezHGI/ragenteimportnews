from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import ejecutar_rag

app = FastAPI(title="Agente Llama3 RAG")

class Query(BaseModel):
    pregunta: str

@app.post("/query")
def api_query(q: Query):
    return {"respuesta": ejecutar_rag(q.pregunta)}

from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import ejecutar_rag

app = FastAPI()

class Query(BaseModel):
    pregunta: str

@app.post("/query")
def consultar(query: Query):
    return {"respuesta": ejecutar_rag(query.pregunta)}

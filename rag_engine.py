import os
from huggingface_hub import InferenceClient
from retriever import crear_retriever

# Inicializar el retriever
retriever = crear_retriever()

# Inicializar el cliente de Inference API
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.environ["HF_TOKEN"]
)

def construir_prompt(contexto: str, pregunta: str) -> str:
    return f"""
<s>[INST] Eres un asistente útil y preciso. Usa el siguiente contexto para responder la pregunta al final. Si no sabes la respuesta, di que no sabes.

Contexto:
{contexto}

Pregunta:
{pregunta}
[/INST]
""".strip()

def ejecutar_rag(pregunta: str) -> str:
    try:
        # Recuperar documentos relevantes
        docs = retriever.get_relevant_documents(pregunta)
        if not docs:
            return "⚠️ No se encontró información relevante para esta pregunta."

        contexto = "\n".join([doc.page_content for doc in docs])
        prompt = construir_prompt(contexto, pregunta)

        respuesta = client.text_generation(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            stop_sequences=["</s>"]
        )

        return respuesta.strip()

    except Exception as e:
        raise RuntimeError(f"❌ Error interno en RAG: {str(e)}")

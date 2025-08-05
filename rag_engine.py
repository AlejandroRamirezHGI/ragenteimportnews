import os
from huggingface_hub import InferenceClient
from retriever import crear_retriever

# Inicializar retriever
retriever = crear_retriever()

# Inicializar cliente HuggingFace Inference API
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # Cambia según el modelo que uses
    token=os.environ["HF_TOKEN"]
)

def ejecutar_rag(pregunta: str) -> str:
    try:
        # Recuperar documentos relevantes
        docs = retriever.get_relevant_documents(pregunta)
        contexto = "\n".join([doc.page_content for doc in docs])

        # Construir el prompt para el modelo
        prompt = f"""
<s>[INST] Eres un asistente útil y preciso. Usa el siguiente contexto para responder la pregunta.
Contexto:
{contexto}

Pregunta:
{pregunta}
[/INST]
"""

        # Llamada al modelo con text_generation
        respuesta = client.text_generation(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            stop_sequences=["</s>"]
        )

        return respuesta

    except Exception as e:
        raise RuntimeError(f"❌ Error interno en RAG: {str(e)}")


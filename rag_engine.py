import os
from huggingface_hub import InferenceClient
from retriever import crear_retriever

SYSTEM = "Eres un agente RAG. Usa sólo la información del contexto para responder."

# Carga segura del token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("❌ Falta el token de HuggingFace. Define HF_TOKEN como variable de entorno.")

# Inicializa el cliente solo si hay token
client = InferenceClient(
    model="meta-llama/Llama-3-8B-Instruct",
    token=HF_TOKEN,
)

def ejecutar_rag(pregunta: str) -> str:
    try:
        # Recuperación de contexto
        retriever = crear_retriever()
        docs = retriever.get_relevant_documents(pregunta)

        if not docs:
            return "No se encontró contexto relevante para responder a tu pregunta."

        contexto = "\n\n".join([doc.page_content for doc in docs])

        # Construcción del prompt
        prompt = (
            f"{SYSTEM}\n\n"
            f"Contexto:\n{contexto}\n\n"
            f"Pregunta: {pregunta}\n"
            f"Respuesta:"
        )

        # Llamada al modelo
        response = client.text_completion(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.0,
            top_p=0.95
        )

        return response.generated_text.strip()

    except Exception as e:
        return f"❌ Error interno en RAG: {str(e)}"

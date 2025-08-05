import os
from huggingface_hub import InferenceClient
from retriever import crear_retriever

SYSTEM = "Eres un agente RAG. Usa sólo la información del contexto para responder."

# Verificación de token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("❌ Falta el token de HuggingFace. Define HF_TOKEN como variable de entorno.")

client = InferenceClient(
    model="meta-llama/Llama-3-8B-Instruct",  # Puedes cambiar a la versión 70B si lo necesitas
    token=HF_TOKEN,
)

def ejecutar_rag(pregunta: str) -> str:
    retriever = crear_retriever()
    docs = retriever.get_relevant_documents(pregunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    prompt = (
        f"{SYSTEM}\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta: {pregunta}\n"
        f"Respuesta:"
    )

    response = client.text_completion(
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.0,
        top_p=0.95
    )
    
    return response.generated_text.strip()

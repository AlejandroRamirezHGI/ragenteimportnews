from huggingface_hub import InferenceClient
from retriever import crear_retriever

SYSTEM = "Eres un agente RAG. Usa sólo la información del contexto para responder."

client = InferenceClient(
    model="meta-llama/Llama-3-8B-Instruct",  # o version 70B
    token=os.getenv("HF_TOKEN"),
)

def ejecutar_rag(pregunta: str) -> str:
    retriever = crear_retriever()
    docs = retriever.get_relevant_documents(pregunta)
    contexto = "\n\n".join([d.page_content for d in docs])
    prompt = f"{SYSTEM}\n\nContexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"
    response = client.text_completion(prompt=prompt, max_new_tokens=256, temperature=0.0, top_p=0.95)
    return response.generated_text.strip()

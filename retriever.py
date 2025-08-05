import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def crear_retriever(path_docs="documents", persist_dir="chroma_db"):
    try:
        # Verificar existencia de carpeta de persistencia
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            print("🔄 Inicializando base de vectores desde documentos...")

            if not os.path.exists(path_docs) or not os.listdir(path_docs):
                raise FileNotFoundError("❌ No hay archivos .txt en la carpeta 'documents'.")

            # Cargar documentos
            loader = DirectoryLoader(path_docs, glob="*.txt", loader_cls=TextLoader)
            docs = loader.load()

            # Dividir documentos
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            # Crear embeddings
            embeds = HuggingFaceEmbeddings(model_name="intfloat/e5-small")

            # Crear Chroma con persistencia
            Chroma.from_documents(chunks, embedding=embeds, persist_directory=persist_dir)
            print("✅ Base de vectores creada correctamente.")
        else:
            print("✅ Base de vectores cargada desde persistencia.")

        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=HuggingFaceEmbeddings(model_name="intfloat/e5-small")
        )

        return vectordb.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        raise RuntimeError(f"❌ Error al crear el retriever: {str(e)}")

# retriever.py
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def crear_retriever(path_docs="docs"):
    loader = DirectoryLoader(path_docs, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeds = HuggingFaceEmbeddings(model_name="intfloat/e5-small")
    vectordb = Chroma.from_documents(chunks, embeddings=embeds, persist_directory="chroma_db")
    return vectordb.as_retriever(search_kwargs={"k": 3})

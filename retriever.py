import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def crear_retriever(path_docs="documents", persist_dir="chroma_db"):
    if not os.path.exists(persist_dir):
        loader = DirectoryLoader(path_docs, glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeds = HuggingFaceEmbeddings(model_name="intfloat/e5-small")
        Chroma.from_documents(chunks, embedding=embeds, persist_directory=persist_dir)

    vectordb = Chroma(persist_directory=persist_dir, embedding_function=HuggingFaceEmbeddings(model_name="intfloat/e5-small"))
    return vectordb.as_retriever(search_kwargs={"k": 3})

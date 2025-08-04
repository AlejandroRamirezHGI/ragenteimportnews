from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def cargar_retriever():
    loader = TextLoader("documents/noticias.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings)
    return vectordb.as_retriever()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

qdrant_client = QdrantClient(host="qdrant", port=6333)
collection_name = "pdf_chunks"

def embed_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> bool:
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    db = Qdrant.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",
        collection_name=collection_name,
    )

    print(f"Successfully embedded and saved {len(docs)} chunks to Qdrant.")
    return True

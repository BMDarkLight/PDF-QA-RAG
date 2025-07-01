import os
import sys
import backoff
import uuid
from pypdf import PdfReader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

@backoff.on_exception(
    backoff.expo,
    (OpenAI.RateLimitError, OpenAI.APIError, OpenAI.APIConnectionError),
    max_time=60
)
def embed_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append((chunk, response.data[0].embedding))
    
    return embeddings

def save(embedded_chunks: list, collection_name: str = "pdf_chunks"):
    qdrant = QdrantClient(host="localhost", port=6333)

    if collection_name not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(embedded_chunks[0][1]),
                distance=Distance.COSINE
            )
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text}
        )
        for text, embedding in embedded_chunks
    ]

    qdrant.upsert(collection_name=collection_name, points=points)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    chunks = embed_pdf(pdf_path)
    save(chunks, collection_name="pdf_chunks")
    print(f"Successfully embedded and saved {len(chunks)} chunks to Qdrant.")

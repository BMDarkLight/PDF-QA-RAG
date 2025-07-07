# RAG PDF‑QA System

a Retrieval‑Augmented Generation (RAG) web app that lets a user ask questions about any PDF and see both an answer _and_ the supporting passages.


## 1. High‑Level Architecture

```text
User → PDF → Text Chunker → OpenAI Embeddings → Qdrant Vector DB
                                         ↘                ↘
                               LangChain Retriever ← Streamlit UI ← User Question
                                           ↘
                             GPT‑4o‑mini (via LangChain) → Answer + Source Chunks
```

* **Chunking**: `RecursiveCharacterTextSplitter` from LangChain slices the book into ~1 k‑token windows with ~200‑token overlap.  
* **Embedding**: `text‑embedding‑3-small` (OpenAI) converts chunks to 1536‑dim vectors.  
* **Storage**: Vectors & metadata are stored in **Qdrant**.
* **Generation**: `GPT‑4o‑mini` (6‑38 k context) creates concise answers with citations.  
* **UI**: A lightweight **Streamlit** front‑end.

---

## 2. Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Vector DB | **Qdrant 1.9+** | Scalable, hybrid search, easy Docker |  
| NLP / Orchestration | **LangChain 0.2+** | Out‑of‑the‑box chunking, retrievers, agents |  
| Embeddings | **OpenAI `text‑embedding‑3-small`** | Fast & cheap (14k tokens/s) |
| LLM | **GPT‑4o‑mini** | 4‑series reasoning, 128k context |  
| PDF | **PyPDF** | Robust parsing incl. bookmarks/figures |  
| UI | **Streamlit 1.35** | 1‑file prototype -> PaaS ready |  

---

## 3. Repository Layout

```text
PDF-QA-RAG/
├─ docker/
│  ├─ Dockerfile
│  └─ docker-compose.yml
├─ src/
│  ├─ ingest.py        # Phase‑1 pipeline
│  ├─ rag_chain.py     # LangChain wrapper
│  └─ ui_streamlit.py  # Phase‑2 app
├─ .env.example
└─ README.md 
```

---

## 4. How to run

Install and run docker CLI on your operating system and run this command in parent folder of the repository:

```shell
docker build --build-arg OPENAI_API_KEY=$OPENAI_API_KEY -f docker/Dockerfile -t image .
```

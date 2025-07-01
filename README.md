# RAG Book‑QA System – Project Documentation

> **Goal:** Build and ship a Retrieval‑Augmented Generation (RAG) web app that lets a user ask questions about any PDF book and see both an answer _and_ the supporting passages, all before **21 June 2025**.

---

## 1. High‑Level Architecture

```text
PDF → Text Chunker → OpenAI Embeddings → Qdrant Vector DB
                                     ↘                ↘
                           LangChain Retriever ← Streamlit UI ← User Question
                                       ↘
                         GPT‑4o‑mini (via LangChain) → Answer + Source Chunks
```

* **Chunking**: `RecursiveCharacterTextSplitter` from LangChain slices the book into ~1 k‑token windows with ~200‑token overlap.  
* **Embedding**: `text‑embedding‑3-small` (OpenAI) converts chunks to 1536‑dim vectors.  
* **Storage**: Vectors & metadata are stored in **Qdrant** (local `docker‑compose` or Hamravesh managed instance).  
* **Retrieval**: `qdrant.as_retriever()` returns top‑k chunks.  
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
| DevOps | **Docker + hamravesh.com** | Easy container deploy in IR cloud |  

---

## 3. Milestones & Deliverables

| Phase | Scope | Deliverables | Deadline |
|-------|-------|--------------|----------|
| **1. Ingestion** | Parse, chunk, embed **entire book**, push to Qdrant | *`ingest.py`*, Qdrant collection snapshot, **PNG screenshot** of Qdrant UI heat‑map | **16 Jun 2025** |
| **2. RAG Core** | Build Retriever‑Generator pipeline, Streamlit UX | *`app.py`*, github repo link, **≤ 60 s demo.mp4** | **19 Jun 2025** |
| **3. Shipping** | Dockerize, deploy on **Hamravesh**, social sharing | Public URL, LinkedIn post URL, deploy docs | **21 Jun 2025** |

---

## 4. Repository Layout

```text
rag-book-qa/
├─ docker/
│  ├─ Dockerfile
│  └─ docker-compose.yml
├─ notebooks/          # playground & EDA
├─ src/
│  ├─ ingest.py        # Phase‑1 pipeline
│  ├─ rag_chain.py     # LangChain wrapper
│  └─ ui_streamlit.py  # Phase‑2 app
├─ tests/
├─ .env.example
└─ README.md           # quick‑start
```

---

## 5. Streamlit UX

* 💬 **Ask question** → answer appears with **expander** listing top chunks.  
* 🖼️ Phase 1 provides a Qdrant collection **t-SNE plot** PNG on sidebar.

---

## 6. Deployment Notes (Hamravesh)

1. Set environment vars (`OPENAI_API_KEY`, `QDRANT_URL`, etc.).  
2. `docker build -t rag-book-qa:latest . && docker push registry.hamravesh.com/<user>/rag-book-qa`  
3. Configure port `8501`.  



from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient

qdrant = QdrantClient(host="qdrant", port=6333)
collection_name = "pdf_chunks"
embedding = OpenAIEmbeddings()
vectorstore = Qdrant(
    client=qdrant,
    collection_name=collection_name,
    embedding=embedding,
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
prompt = PromptTemplate.from_template(
    "Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.1),
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def ask_question(query: str):
    result = qa_chain(query)
    print("Answer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "No source"), "\n", doc.page_content[:200], "\n")

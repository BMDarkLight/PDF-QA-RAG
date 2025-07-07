import streamlit as st
import tempfile
from ingest import embed_pdf
from rag_chain import ask_question

st.set_page_config(page_title="PDF-QA RAG", layout="wide")
st.title("📄 PDF-QA with LangChain and Qdrant")

st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully.")

    with st.spinner("Processing PDF ..."):
        if embed_pdf(file_path=pdf_path):
            st.success("Ingestion complete!")
        else:
            st.error("Failed to run ingestion.")

    st.subheader("Ask a question about the PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Your question:")
    if user_question:
        with st.spinner("Searching for answer..."):
            response = ask_question(user_question)
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("Bot", response))

    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")
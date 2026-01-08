import os
import shutil
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure data dir exists early (before uploads)
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Enterprise Knowledge Copilot", layout="wide")
st.title("🏢 Enterprise Knowledge Copilot")
st.caption("Upload enterprise documents, build an index, then ask questions with sources.")

# ----------------------------
# Init LLM + Embeddings
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# ----------------------------
# Document Loading
# ----------------------------
def load_documents_from_data_dir() -> List[Document]:
    """Load all supported files from DATA_DIR into LangChain Documents."""
    documents: List[Document] = []

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif filename.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

    return documents

# ----------------------------
# Chunking
# ----------------------------
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ----------------------------
# Vector DB (Build / Load)
# ----------------------------
def build_vectorstore() -> Chroma:
    """(Re)build the Chroma index from docs in DATA_DIR."""
    # Delete old DB if exists
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    docs = load_documents_from_data_dir()
    if not docs:
        raise ValueError("No documents found. Upload PDFs/TXT files first.")

    chunks = split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb

def load_vectorstore() -> Chroma | None:
    """Load existing Chroma index if present."""
    if not os.path.exists(CHROMA_DIR):
        return None
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

# ----------------------------
# Prompt & Chains
# ----------------------------
PROMPT = ChatPromptTemplate.from_template(
    """
You are an enterprise AI assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

Return a clear, professional answer.

Context:
{context}

Question:
{question}
"""
)

def retrieve_docs(query: str) -> List[Document]:
    vectordb = load_vectorstore()
    if vectordb is None:
        return []

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )
    return retriever.invoke(query)

def answer_question(question: str, docs: List[Document]) -> str:
    if not docs:
        return "I don't know. I couldn't find relevant context in the indexed documents."

    context = "\n\n".join(
        [f"Source: {d.metadata.get('source', '')}\n{d.page_content}" for d in docs]
    )

    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# ----------------------------
# Sidebar: Upload + Index
# ----------------------------
st.sidebar.header("📂 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uf in uploaded_files:
        out_path = os.path.join(DATA_DIR, uf.name)
        with open(out_path, "wb") as f:
            f.write(uf.read())
    st.sidebar.success("✅ Files uploaded. Now click 'Build / Rebuild Index'.")

if st.sidebar.button("🔄 Build / Rebuild Index"):
    try:
        with st.spinner("Indexing documents..."):
            build_vectorstore()
        st.sidebar.success("✅ Index built successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to build index: {e}")

# ----------------------------
# Main UI: Ask Questions
# ----------------------------
vectordb = load_vectorstore()
if vectordb is None:
    st.info("Upload documents and build the index to start asking questions.")
    st.stop()

question = st.text_input("Ask your enterprise question:")

if question.strip():
    with st.spinner("Retrieving and generating answer..."):
        docs = retrieve_docs(question)
        answer = answer_question(question, docs)

    st.subheader("✅ Answer")
    st.write(answer)

    st.subheader("📚 Sources")
    if not docs:
        st.info("No sources found for this question.")
    else:
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source", "unknown")
            with st.expander(f"Source {i}: {src}"):
                st.write(doc.page_content)

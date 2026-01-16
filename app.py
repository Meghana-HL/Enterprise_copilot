import os
import time
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma  # no deprecation warning
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import chromadb


# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# API key guard (local .env + Streamlit Cloud)
# ----------------------------
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception:
        pass

if not api_key:
    st.error(
        "OPENAI_API_KEY not found.\n\n"
        "✅ Local: create a .env file in your project root with:\n"
        "OPENAI_API_KEY=sk-...\n\n"
        "✅ Streamlit Cloud: set Secrets with:\n"
        "OPENAI_API_KEY = \"sk-...\""
    )
    st.stop()


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR_BASE = os.path.join(BASE_DIR, "chroma_db")  # base prefix; versioned per rebuild
COLLECTION_NAME = "enterprise_kb"

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Enterprise Knowledge Copilot", layout="wide")
st.title("🏢 Enterprise Knowledge Copilot")
st.caption("Upload enterprise documents, build an index, then chat with your knowledge base (with sources).")

# ----------------------------
# Init LLM + Embeddings
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()


# ----------------------------
# Session helpers (Windows-safe indexing)
# ----------------------------
def get_active_chroma_dir() -> str:
    if "chroma_dir" not in st.session_state:
        st.session_state.chroma_dir = CHROMA_DIR_BASE
    return st.session_state.chroma_dir


def new_chroma_dir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{CHROMA_DIR_BASE}_{ts}"


# ----------------------------
# Document Loading
# ----------------------------
def load_documents_from_data_dir() -> List[Document]:
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
    chunks = splitter.split_documents(documents)

    # Normalize metadata a bit (nice for UI + debugging)
    for d in chunks:
        d.metadata = d.metadata or {}
        d.metadata["source"] = d.metadata.get("source", "unknown")
        if "page" in d.metadata:
            # PyPDFLoader uses 0-index pages typically
            d.metadata["page"] = int(d.metadata["page"])
    return chunks


# ----------------------------
# Vector DB (Build / Load)
# ----------------------------
def build_vectorstore() -> Chroma:
    """
    Build Chroma index into a NEW directory each time (Windows-safe).
    Uses chromadb.PersistentClient so persistence is handled by Chroma itself.
    """
    target_dir = new_chroma_dir()

    docs = load_documents_from_data_dir()
    if not docs:
        raise ValueError("No documents found. Upload PDFs/TXT files first.")

    chunks = split_documents(docs)

    # Persistent client (recommended)
    client = chromadb.PersistentClient(path=target_dir)

    # Create / get collection + wrap with LangChain
    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Add documents (this persists automatically with PersistentClient)
    vectordb.add_documents(chunks)

    # Track active directory in session
    st.session_state.chroma_dir = target_dir
    return vectordb


def load_vectorstore() -> Optional[Chroma]:
    active_dir = get_active_chroma_dir()
    if not os.path.exists(active_dir):
        return None

    client = chromadb.PersistentClient(path=active_dir)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


# ----------------------------
# Prompt & Retrieval / Answer
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
        [
            f"Source: {d.metadata.get('source', '')}"
            + (f" | Page: {d.metadata.get('page')}" if d.metadata.get("page") is not None else "")
            + f"\n{d.page_content}"
            for d in docs
        ]
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
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Failed to build index: {e}")

st.sidebar.divider()

active_dir = get_active_chroma_dir()
st.sidebar.caption("Active index directory:")
st.sidebar.code(active_dir)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# ----------------------------
# Require index before chat
# ----------------------------
vectordb = load_vectorstore()
if vectordb is None:
    st.info("Upload documents and build the index to start chatting.")
    st.stop()


# ----------------------------
# Chat state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the documents you uploaded."}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Source {i}: {src['source']}**")
                    if src.get("page") is not None:
                        st.caption(f"Page: {src['page']}")
                    st.write(src["content"])
                    st.divider()

# Chat input
user_input = st.chat_input("Ask your enterprise question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            docs = retrieve_docs(user_input)
            answer = answer_question(user_input, docs)

        st.write(answer)

        sources_payload = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "content": d.page_content,
            }
            for d in docs
        ]

        if sources_payload:
            with st.expander("📚 Sources"):
                for i, s in enumerate(sources_payload, start=1):
                    st.markdown(f"**Source {i}: {s['source']}**")
                    if s.get("page") is not None:
                        st.caption(f"Page: {s['page']}")
                    st.write(s["content"])
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_payload}
    )

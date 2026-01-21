import os
import time
import json
from typing import List, Optional, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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
st.caption("Upload enterprise documents, build an index, then chat with your knowledge base")

# ----------------------------
# Init LLM + Embeddings
# ----------------------------
# Force JSON output reliably
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)
embeddings = OpenAIEmbeddings()


# ----------------------------
# Retrieval knobs (NO aggressive filtering)
# ----------------------------
TOP_K = 6
FETCH_K = 20


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

    # Normalize metadata for nicer UI
    for d in chunks:
        d.metadata = d.metadata or {}
        d.metadata["source"] = d.metadata.get("source", "unknown")
        if "page" in d.metadata:
            d.metadata["page"] = int(d.metadata["page"])
    return chunks


# ----------------------------
# Vector DB (Build / Load)
# ----------------------------
def build_vectorstore() -> Chroma:
    """
    Build Chroma index into a NEW directory each time (Windows-safe).
    """
    target_dir = new_chroma_dir()

    docs = load_documents_from_data_dir()
    if not docs:
        raise ValueError("No documents found. Upload PDFs/TXT files first.")

    chunks = split_documents(docs)

    client = chromadb.PersistentClient(path=target_dir)

    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    vectordb.add_documents(chunks)

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
If the answer is not in the context, respond with:
- answer: "I don't know."
- used_sources: []

The context blocks are labeled like [S1], [S2], etc.
In used_sources, return ONLY the numbers you actually used to produce the answer.

Return STRICT JSON ONLY with this schema:
{{
  "answer": string,
  "used_sources": number[]
}}

Context:
{context}

Question:
{question}
"""
)


def retrieve_docs(query: str) -> List[Document]:
    """
    Use MMR retrieval (works well and avoids weird score threshold behavior).
    """
    vectordb = load_vectorstore()
    if vectordb is None:
        return []

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K},
    )
    return retriever.invoke(query)


def _build_labeled_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        header = f"[S{i}] Source: {src}" + (f" | Page: {page}" if page is not None else "")
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _parse_json_or_fallback(text: str) -> Dict[str, Any]:
    """
    Model is forced to output JSON, but keep a safe fallback.
    """
    try:
        data = json.loads(text)
        answer = data.get("answer", "")
        used = data.get("used_sources", [])
        if not isinstance(answer, str):
            answer = str(answer)
        if not isinstance(used, list):
            used = []
        used_int = []
        for x in used:
            try:
                used_int.append(int(x))
            except Exception:
                pass
        return {"answer": answer.strip(), "used_sources": used_int}
    except Exception:
        return {"answer": "I don't know.", "used_sources": []}


def answer_question(question: str, docs: List[Document]) -> Dict[str, Any]:
    """
    Returns:
      { "answer": str, "used_sources": [int...] }  # indices are 1-based
    """
    if not docs:
        return {"answer": "I don't know.", "used_sources": []}

    context = _build_labeled_context(docs)

    chain = PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question})
    parsed = _parse_json_or_fallback(raw)

    # Normalize "don't know" behavior: no sources in that case
    if parsed["answer"].strip().lower() in {"i don't know.", "i don't know"}:
        parsed["used_sources"] = []

    # Keep only valid indices
    max_i = len(docs)
    parsed["used_sources"] = [i for i in parsed["used_sources"] if 1 <= i <= max_i]

    return parsed


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
    st.sidebar.success("✅ Files uploaded. Now click 'Build Index'.")

if st.sidebar.button("🔄 Build Index"):
    try:
        with st.spinner("Indexing documents..."):
            build_vectorstore()
        st.sidebar.success("✅ Index built successfully!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Failed to build index: {e}")

st.sidebar.divider()

# active_dir = get_active_chroma_dir()
# st.sidebar.caption("Active index directory:")
# st.sidebar.code(active_dir)

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
            result = answer_question(user_input, docs)

        answer_text = result["answer"]
        used_sources = result["used_sources"]  # 1-based indices into docs

        st.write(answer_text)

        # Build sources payload ONLY from sources the model says it used
        sources_payload = []
        for idx in used_sources:
            d = docs[idx - 1]
            sources_payload.append(
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page"),
                    "content": d.page_content,
                }
            )

        # Show sources only if present
        if sources_payload:
            with st.expander("📚 Sources"):
                for i, s in enumerate(sources_payload, start=1):
                    st.markdown(f"**Source {i}: {s['source']}**")
                    if s.get("page") is not None:
                        st.caption(f"Page: {s['page']}")
                    st.write(s["content"])
                    st.divider()

    msg = {"role": "assistant", "content": answer_text}
    if sources_payload:
        msg["sources"] = sources_payload
    st.session_state.messages.append(msg)

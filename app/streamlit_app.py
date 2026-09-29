# app/streamlit_app.py
"""
Streamlit UI — Enterprise RAG System
======================================
Thesis: Design and Empirical Evaluation of a RAG System
        for Intelligent Document Querying in Digital Enterprises

Architecture: Dynamic upload + cumulative in-memory index
  - Base corpus: 15 pre-loaded enterprise PDFs
  - User uploads: augment the base corpus per session
  - Index resets on browser tab close (in-memory)

Run from project root:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import List

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from src.vectorstore.faiss_store import load_index, build_index
from src.ingestion.pdf_loader import load_pdfs
from src.preprocessing.cleaner import preprocess_documents as clean_documents
from src.preprocessing.chunker import chunk_documents
from src.retrieval.hybrid_retriever import build_hybrid_retriever
from src.retrieval.reranker import Reranker
from src.rag.rag_chain import build_rag_chain

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

    /* ==================================================
       MAIN APPLICATION
       ================================================== */

    .stApp {
        background-color: #0f1117;
        color: #ffffff;
    }

    /* Global text */
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #ffffff;
    }


    /* ==================================================
       SIDEBAR
       ================================================== */

    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
        color: #ffffff;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }


    /* ==================================================
       INPUT FIELD
       ================================================== */

    .stTextInput > div > div > input {
        background-color: #1e2433;
        color: #ffffff;
        border: 1px solid #6b7280;
        border-radius: 8px;
    }

    /* Placeholder text */
    .stTextInput input::placeholder {
        color: #c5cad6;
        opacity: 1;
    }


    /* ==================================================
       SLIDERS / TOGGLE LABELS
       ================================================== */

    [data-testid="stSlider"] label,
    [data-testid="stSlider"] p,
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] p {
        color: #ffffff;
    }


    /* ==================================================
       FILE UPLOADER
       ================================================== */

    [data-testid="stFileUploader"] {
        color: #ffffff;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: #ffffff;
    }


    /* ==================================================
       ANSWER CARD
       ================================================== */

    .answer-card {
        background: linear-gradient(
            135deg,
            #1a2035 0%,
            #1e2840 100%
        );

        border: 1px solid #2e3a5c;
        border-left: 4px solid #4f8ef7;
        border-radius: 10px;

        padding: 1.5rem;
        margin: 1rem 0;

        color: #ffffff;

        font-size: 1rem;
        line-height: 1.7;
    }


    /* ==================================================
       SOURCE CARD
       ================================================== */

    .source-card {
        background-color: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 8px;

        padding: 0.8rem 1rem;
        margin: 0.4rem 0;

        font-size: 0.85rem;
        color: #ffffff;
    }

    .source-card span {
        color: #ffffff;
    }

    .source-card strong {
        color: #7ab3ff;
    }


    /* ==================================================
       SCORE BADGE
       ================================================== */

    .score-badge {
        display: inline-block;

        background-color: #1e3a5f;
        color: #9ac8ff !important;

        border-radius: 4px;
        padding: 1px 7px;

        font-size: 0.78rem;
        margin-left: 6px;
    }


    /* ==================================================
       UPLOAD BADGE
       ================================================== */

    .upload-badge {
        display: inline-block;

        background-color: #1e3d1e;
        color: #8ee68e !important;

        border-radius: 4px;
        padding: 1px 7px;

        font-size: 0.78rem;
        margin-left: 6px;
    }


    /* ==================================================
       SECTION LABEL
       ================================================== */

    .section-label {
        color: #ffffff;

        font-size: 0.75rem;
        font-weight: 600;

        letter-spacing: 0.12em;
        text-transform: uppercase;

        margin-bottom: 0.5rem;
        margin-top: 1.2rem;
    }


    /* ==================================================
       CORPUS ITEMS
       ================================================== */

    .corpus-item {
        background-color: #1a1f2e;

        border-radius: 6px;

        padding: 0.4rem 0.75rem;
        margin: 0.2rem 0;

        font-size: 0.8rem;
        color: #ffffff;

        border: 1px solid #252b3b;
    }

    .corpus-item.uploaded {
        border-left: 3px solid #4caf50;
    }

    .corpus-item.base {
        border-left: 3px solid #4f8ef7;
    }


    /* ==================================================
       EXPANDERS / HISTORY
       ================================================== */

    [data-testid="stExpander"] {
        color: #ffffff;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #ffffff;
    }


    /* ==================================================
       HIDE STREAMLIT DEFAULT ELEMENTS
       ================================================== */

    #MainMenu,
    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ── cached base loaders ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner="Loading base FAISS index …")
def load_base_vectorstore(_embedder):
    return load_index(_embedder)

@st.cache_resource(show_spinner="Loading base corpus chunks …")
def load_base_chunks():
    raw   = load_pdfs()
    clean = clean_documents(raw)
    return chunk_documents(clean)

@st.cache_resource(show_spinner="Loading cross-encoder reranker …")
def load_reranker():
    return Reranker()


# ── session state init ────────────────────────────────────────────────────────

def init_session():
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []       # list of filenames uploaded this session
    if "extra_chunks" not in st.session_state:
        st.session_state.extra_chunks = []        # chunks from uploaded PDFs
    if "history" not in st.session_state:
        st.session_state.history = []
    if "retriever_dirty" not in st.session_state:
        st.session_state.retriever_dirty = True   # rebuild retriever on first query

init_session()


# ── load base resources ───────────────────────────────────────────────────────

embedder       = load_embedder()
base_vectorstore = load_base_vectorstore(embedder)
base_chunks    = load_base_chunks()
reranker       = load_reranker()


# ── dynamic retriever builder ─────────────────────────────────────────────────

def get_retriever(top_k: int):
    """
    Build hybrid retriever over base corpus + any uploaded docs.
    Rebuilds only when new documents have been uploaded.
    """
    all_chunks = base_chunks + st.session_state.extra_chunks

    if st.session_state.extra_chunks:
        # Merge uploaded chunks into a fresh FAISS index
        import faiss
        import numpy as np
        from langchain_community.vectorstores import FAISS

        # Embed the new chunks
        new_texts = [c.page_content for c in st.session_state.extra_chunks]
        new_embeddings = embedder.embed_documents(new_texts)

        # Clone base index and add new vectors
        merged_vs = base_vectorstore
        merged_vs.add_texts(
            new_texts,
            metadatas=[c.metadata for c in st.session_state.extra_chunks],
        )
        vectorstore = merged_vs
    else:
        vectorstore = base_vectorstore

    return build_hybrid_retriever(all_chunks, vectorstore, top_k=top_k)


# ── static retriever wrapper ──────────────────────────────────────────────────

class StaticRetriever(BaseRetriever):
    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Enterprise RAG")
    st.markdown("*MSc Thesis — SRH University*")
    st.divider()

    # ── document upload ────────────────────────────────────────────────────
    st.markdown("**Upload documents**")
    st.markdown(
    "<span style='font-size:0.8rem;color:#4F8EF7'>"
    "Uploaded PDFs are added to the base corpus for this session.</span>",
    unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.uploaded_docs
        ]
        if new_files:
            for uf in new_files:
                with st.spinner(f"Processing {uf.name} …"):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name

                    # Process through pipeline
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(tmp_path)
                    raw_docs = loader.load()

                    # Attach real filename to metadata
                    for doc in raw_docs:
                        doc.metadata["source"] = uf.name
                        doc.metadata["filename"] = uf.name

                    clean  = clean_documents(raw_docs)
                    chunks = chunk_documents(clean)

                    st.session_state.extra_chunks.extend(chunks)
                    st.session_state.uploaded_docs.append(uf.name)
                    os.unlink(tmp_path)

                st.success(f"✅ {uf.name} — {len(chunks)} chunks added")

            st.session_state.retriever_dirty = True

    st.divider()

    # ── retrieval settings ─────────────────────────────────────────────────
    # st.markdown("**Retrieval settings**")
    # top_k        = st.slider("Candidates (hybrid)", 3, 20, 10)
    # top_n        = st.slider("After reranking", 1, 5, 3)
    # use_reranker = st.toggle("Use cross-encoder reranker", value=True)

    # Internal retrieval settings — not shown in Streamlit UI
    top_k = 10
    top_n = 3
    use_reranker = True

    st.divider()

    # ── corpus status ──────────────────────────────────────────────────────
    st.markdown("**Active corpus**")
    total_chunks = len(base_chunks) + len(st.session_state.extra_chunks)
    st.markdown(
        f"<span style='font-size:0.85rem;color:#e8eaf0'>"
        f"{total_chunks:,} chunks indexed</span>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='corpus-item base'>📂 Base: 15 enterprise PDFs</div>",
        unsafe_allow_html=True,
    )
    for name in st.session_state.uploaded_docs:
        st.markdown(
            f"<div class='corpus-item uploaded'>📄 {name}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🗑 Clear uploaded docs", use_container_width=True):
        st.session_state.extra_chunks  = []
        st.session_state.uploaded_docs = []
        st.session_state.history       = []
        st.rerun()


# ── main UI ───────────────────────────────────────────────────────────────────

st.markdown("## Intelligent Document Querying with Retrieval-Augmented Generation")

# Show corpus context
base_count     = 15
upload_count   = len(st.session_state.uploaded_docs)
corpus_desc    = f"15 Enterprise PDFs"
if upload_count:
    corpus_desc += f" + {upload_count} uploaded document{'s' if upload_count > 1 else ''}"
st.markdown(
    f"<span style='color:#ffffff;font-size:0.9rem'>{corpus_desc}</span>",
    unsafe_allow_html=True,
)

st.markdown("")

query = st.text_input(
    label="Your question",
    placeholder="e.g. What was Apple's total revenue in FY2022?",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 6])
with col1:
    run = st.button("Ask →", type="primary", use_container_width=True)


# ── query execution ───────────────────────────────────────────────────────────

if run and query.strip():
    with st.spinner("Retrieving and generating answer …"):

        # Build retriever (merges base + uploaded)
        hybrid = get_retriever(top_k)

        # Hybrid retrieval
        candidates = hybrid.invoke(query)

        # Reranking
        if use_reranker and candidates:
            top_docs = reranker.rerank(query, candidates[:top_k])[:top_n]
        else:
            top_docs = candidates[:top_n]

        # RAG chain
        chain  = build_rag_chain(StaticRetriever(docs=top_docs))
        result = chain.invoke({"query": query})
        answer = result.get("result", "No answer generated.")

    # Save history
    st.session_state.history.insert(0, {
        "q": query, "a": answer, "docs": top_docs
    })

    # Answer
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    # Sources
    st.markdown('<div class="section-label">Sources used</div>', unsafe_allow_html=True)
    seen = set()
    for doc in top_docs:
        fname = doc.metadata.get("filename") or os.path.basename(
            doc.metadata.get("source", "unknown")
        )
        page  = doc.metadata.get("page", "?")
        score = doc.metadata.get("rerank_score", None)
        key   = f"{fname}_p{page}"
        if key in seen:
            continue
        seen.add(key)
        pg         = page + 1 if isinstance(page, int) else page
        score_html = (
            f'<span class="score-badge">score {score}</span>'
            if score is not None else ""
        )
        is_upload  = fname in st.session_state.uploaded_docs
        upload_tag = (
            '<span class="upload-badge">uploaded</span>' if is_upload else ""
        )
        snippet = doc.page_content[:200].replace("\n", " ")
        st.markdown(
            f'<div class="source-card">'
            f'<strong>{fname}</strong> — Page {pg}{score_html}{upload_tag}<br>'
            f'<span style="font-size:0.8rem">{snippet}…</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

elif run and not query.strip():
    st.warning("Please enter a question.")


# ── query history ─────────────────────────────────────────────────────────────

if st.session_state.history:
    st.divider()
    st.markdown('<div class="section-label">Recent queries</div>', unsafe_allow_html=True)
    for item in st.session_state.history[:5]:
        with st.expander(f"Q: {item['q'][:80]}{'…' if len(item['q']) > 80 else ''}"):
            st.markdown(f"**Answer:** {item['a']}")
            for doc in item["docs"]:
                fname = doc.metadata.get("filename") or os.path.basename(
                    doc.metadata.get("source", "unknown")
                )
                page = doc.metadata.get("page", "?")
                pg   = page + 1 if isinstance(page, int) else page
                tag  = " 🟢" if fname in st.session_state.uploaded_docs else ""
                st.markdown(f"- `{fname}` — Page {pg}{tag}")
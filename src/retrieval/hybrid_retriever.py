"""
src/retrieval/hybrid_retriever.py
==================================
Hybrid Retriever — BM25 (sparse) + FAISS (dense) via LangChain EnsembleRetriever.

Architecture (Offline → Online):
  Offline: All chunks are BM25-indexed at load time (in-memory, rebuilt per session).
  Online : EnsembleRetriever fuses BM25 (weight=0.40) + FAISS (weight=0.60)
           using Reciprocal Rank Fusion (RRF), then returns top-K unique docs.

Config values used (from config.py):
  BM25_WEIGHT  = 0.40
  DENSE_WEIGHT = 0.60
  TOP_K        = 5

Usage:
  from src.retrieval.hybrid_retriever import build_hybrid_retriever
  retriever = build_hybrid_retriever(chunks, faiss_vectorstore)
  docs = retriever.invoke("What is the revenue for FY2023?")
"""

import logging
from typing import List

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ── config ────────────────────────────────────────────────────────────────────
BM25_WEIGHT  = 0.40
DENSE_WEIGHT = 0.60
TOP_K        = 5

logger = logging.getLogger(__name__)


# ── builder ───────────────────────────────────────────────────────────────────

def build_hybrid_retriever(
    chunks: List[Document],
    faiss_vectorstore,
    top_k: int = TOP_K,
    candidate_k: int = 10,
    bm25_weight: float = BM25_WEIGHT,
    dense_weight: float = DENSE_WEIGHT,
) -> EnsembleRetriever:
    """
    Build and return a hybrid EnsembleRetriever.

    Parameters
    ----------
    chunks : List[Document]
        The same document chunks that were embedded into FAISS.
        BM25 is built in-memory from these at call time.
    faiss_vectorstore : FAISS
        The loaded FAISS vectorstore (from faiss_store.load_faiss_index()).
    top_k : int
    Number of documents returned after hybrid fusion.

    candidate_k : int
    Number of candidate documents retrieved independently by BM25
    and FAISS before Reciprocal Rank Fusion.
    bm25_weight : float
        RRF weight for sparse BM25 scores (default 0.40).
    dense_weight : float
        RRF weight for dense FAISS scores (default 0.60).

    Returns
    -------
    EnsembleRetriever
        Ready to call with .invoke("your query").
    """

    if not chunks:
        raise ValueError("chunks list is empty — cannot build BM25 index.")

    if round(bm25_weight + dense_weight, 6) != 1.0:
        logger.warning(
            "BM25 weight (%.2f) + dense weight (%.2f) != 1.0. "
            "EnsembleRetriever normalises internally, but check config.",
            bm25_weight, dense_weight,
        )

    # ── 1. Sparse retriever (BM25) ─────────────────────────────────────────
    logger.info("Building BM25 index over %d chunks …", len(chunks))
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = candidate_k
    logger.info("BM25 index ready (candidate_k=%d).", candidate_k)

    # ── 2. Dense retriever (FAISS) ─────────────────────────────────────────
    logger.info(
    "Wrapping FAISS vectorstore as dense retriever (candidate_k=%d) …",
    candidate_k,
    )
    faiss_retriever = faiss_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": candidate_k},
    )
    logger.info(
    "Candidate pool: BM25=%d | FAISS=%d | Final top_k=%d",
    candidate_k,
    candidate_k,
    top_k,
    )
    # ── 3. Ensemble (Reciprocal Rank Fusion) ──────────────────────────────
    logger.info(
        "Assembling EnsembleRetriever — BM25 weight=%.2f | FAISS weight=%.2f",
        bm25_weight, dense_weight,
    )
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[bm25_weight, dense_weight],
    )

    logger.info("HybridRetriever ready.")
    return ensemble


# ── standalone smoke-test ─────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Quick end-to-end test.
    Run from project root:
        python -m src.retrieval.hybrid_retriever
    Requires: FAISS index already built (run faiss_store.py first).
    """
    import sys
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ── path setup ──────────────────────────────────────────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config import config
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.vectorstore.faiss_store import load_index as load_faiss_index

    # ── load embedding model ────────────────────────────────────────────────
    print("\n[1/4] Loading embedding model …")
    embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── load FAISS index ────────────────────────────────────────────────────
    print("[2/4] Loading FAISS index …")
    vectorstore = load_faiss_index(embedder)
    print(f"      FAISS index loaded: {vectorstore.index.ntotal} vectors")

    # ── load chunks for BM25 ───────────────────────────────────────────────
    print("[3/4] Loading chunks for BM25 …")
    from src.ingestion.pdf_loader import load_pdfs
    from src.preprocessing.cleaner import preprocess_documents as clean_documents
    from src.preprocessing.chunker import chunk_documents

    raw_docs   = load_pdfs()
    clean_docs = clean_documents(raw_docs)
    chunks     = chunk_documents(clean_docs)
    print(f"      {len(chunks)} chunks ready for BM25.")

    # ── build hybrid retriever ─────────────────────────────────────────────
    print("[4/4] Building hybrid retriever …")
    retriever = build_hybrid_retriever(chunks, vectorstore)

    # ── test query ─────────────────────────────────────────────────────────
    TEST_QUERY = "What is the total revenue reported in the financial statements?"
    print(f"\nTest query: \"{TEST_QUERY}\"")
    results = retriever.invoke(TEST_QUERY)

    print(f"\n{'='*60}")
    print(f"Retrieved {len(results)} documents (top_k={TOP_K}):\n")
    for i, doc in enumerate(results, 1):
        source  = doc.metadata.get("source", "unknown")
        page    = doc.metadata.get("page", "?")
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  [{i}] source={os.path.basename(source)} | page={page}")
        print(f"       {snippet} …\n")

    print("✅ Hybrid retriever smoke-test PASSED.")


if __name__ == "__main__":
    _smoke_test()
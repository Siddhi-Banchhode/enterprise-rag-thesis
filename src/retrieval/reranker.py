"""
src/retrieval/reranker.py
==========================
Cross-Encoder Reranker — ms-marco-MiniLM-L-6-v2

Takes the fused candidates from HybridRetriever and re-scores every
(query, chunk) pair with a cross-encoder, then returns the top-N chunks
ranked by relevance score.

Architecture position:
  HybridRetriever (10 docs) → Reranker → top 3 docs → Prompt → LLM

Config values used:
  RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
  RERANKER_TOP_N = 3

Usage:
  from src.retrieval.reranker import Reranker
  reranker = Reranker()
  top_docs = reranker.rerank(query, candidate_docs)
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TOP_N = 3


# ── reranker class ────────────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker wrapping sentence-transformers CrossEncoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for the cross-encoder.
    top_n : int
        Number of top documents to return after reranking.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        top_n: int = RERANKER_TOP_N,
    ):
        from sentence_transformers import CrossEncoder

        self.top_n = top_n
        self.model_name = model_name

        logger.info("Loading cross-encoder: %s …", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded (top_n=%d).", top_n)

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Rerank candidate documents against the query.

        Parameters
        ----------
        query : str
            The user's natural language question.
        documents : List[Document]
            Candidate docs from HybridRetriever (typically 10).

        Returns
        -------
        List[Document]
            top_n documents sorted by cross-encoder score (descending).
            Each document gets a 'rerank_score' added to its metadata.
        """
        if not documents:
            logger.warning("Reranker received empty document list.")
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs in one batch
        scores = self._model.predict(pairs)

        # Attach scores to document metadata and sort
        scored = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        top_docs = []
        for score, doc in scored[: self.top_n]:
            doc.metadata["rerank_score"] = round(float(score), 4)
            top_docs.append(doc)

        logger.info(
            "Reranked %d → %d docs | top score=%.4f",
            len(documents),
            len(top_docs),
            top_docs[0].metadata["rerank_score"] if top_docs else 0,
        )
        return top_docs


# ── standalone smoke-test ─────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Run from project root:
        python -m src.retrieval.reranker
    Requires: FAISS index already built.
    """
    import sys
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config import config
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.vectorstore.faiss_store import load_index
    from src.ingestion.pdf_loader import load_pdfs
    from src.preprocessing.cleaner import preprocess_documents as clean_documents
    from src.preprocessing.chunker import chunk_documents
    from src.retrieval.hybrid_retriever import build_hybrid_retriever

    # ── 1. Embedder + FAISS ────────────────────────────────────────────────
    print("\n[1/5] Loading embedding model …")
    embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[2/5] Loading FAISS index …")
    vectorstore = load_index(embedder)
    print(f"      FAISS index: {vectorstore.index.ntotal} vectors")

    # ── 2. Chunks for BM25 ────────────────────────────────────────────────
    print("[3/5] Loading chunks for BM25 …")
    chunks = chunk_documents(clean_documents(load_pdfs()))
    print(f"      {len(chunks)} chunks ready.")

    # ── 3. Hybrid retriever ───────────────────────────────────────────────
    print("[4/5] Building hybrid retriever …")
    hybrid = build_hybrid_retriever(chunks, vectorstore)

    # ── 4. Reranker ───────────────────────────────────────────────────────
    print("[5/5] Loading cross-encoder reranker …")
    reranker = Reranker()

    # ── 5. End-to-end query ───────────────────────────────────────────────
    TEST_QUERY = "What is the total revenue reported in the financial statements?"
    print(f"\nTest query: \"{TEST_QUERY}\"")

    candidates = hybrid.invoke(TEST_QUERY)
    print(f"Hybrid retrieved: {len(candidates)} candidates")

    top_docs = reranker.rerank(TEST_QUERY, candidates)

    print(f"\n{'='*60}")
    print(f"Top {len(top_docs)} after reranking:\n")
    for i, doc in enumerate(top_docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page   = doc.metadata.get("page", "?")
        score  = doc.metadata.get("rerank_score", "?")
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  [{i}] score={score} | {source} | page={page}")
        print(f"       {snippet} …\n")

    print("✅ Reranker smoke-test PASSED.")


if __name__ == "__main__":
    _smoke_test()
    
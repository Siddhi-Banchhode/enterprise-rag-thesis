"""
evaluation/run_legalbench_rag_evaluation.py
=============================================
Evaluates YOUR existing, UNCHANGED retrieval pipeline against the official
LegalBench-RAG CUAD benchmark. Reuses your real classes directly:
  - config.EMBEDDING_MODEL          (same embedding model)
  - HuggingFaceEmbeddings           (same embedder wrapper)
  - chunk_documents()               (same chunker, same size/overlap)
  - build_hybrid_retriever()        (same BM25+FAISS hybrid logic)

What's NEW (not a change to your system, just new inputs):
  - a loader for the CUAD .txt corpus (parallel to your PDF loader)
  - a FRESH, separate FAISS index built over CUAD chunks only
    (this does NOT touch or overwrite your saved enterprise-PDF index)
  - an adapter that locates each retrieved chunk's character position in
    its source file, needed for the benchmark's character-overlap scoring

Two deliberate judgment calls (see conversation for reasoning):
  1. clean_documents() is intentionally SKIPPED here — it could shift
     character positions relative to the benchmark's raw-text gold spans.
  2. Chunk offsets are recovered via string search against the raw corpus
     text, with a whitespace-tolerant fallback, since we don't know
     whether your chunker already tracks start_index.

Run:
    python evaluation/run_legalbench_rag_evaluation.py
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.preprocessing.chunker import chunk_documents  # your real chunker, unchanged
from src.retrieval.hybrid_retriever import build_hybrid_retriever  # unchanged

from legalbench_rag_cuad_evaluator import (
    load_cuad_benchmark,
    referenced_file_paths,
    load_corpus_texts,
    RetrievedChunk,
    official_precision_recall_at_k,
    mean_average_precision,
    map_at_k_sensitivity,
)

# ---------------------------------------------------------------------------
# EDIT THESE THREE THINGS
# ---------------------------------------------------------------------------
CUAD_CORPUS_DIR = "/Users/siddhibanchhode/Documents/enterprise-rag/legalbench_data/corpus"
CUAD_BENCHMARK_JSON = "/Users/siddhibanchhode/Documents/enterprise-rag/legalbench_data/benchmarks/cuad.json"
LIMIT_QUERIES = 50  # start small to catch bugs fast; set to None for the full 4,042-query run

TOP_K = 10  # MAP@10 and the official benchmark both score raw top-10, pre-reranker


def find_offset(full_text: str, chunk_text: str):
    """
    Locate where a chunk sits inside its source file's raw text, so we can
    score character overlap against the gold spans.
    Tries an exact match first; falls back to a whitespace-tolerant anchor
    search in case your chunker altered whitespace/formatting.
    Returns (start, end) or (None, None) if unrecoverable.
    """
    idx = full_text.find(chunk_text)
    if idx != -1:
        return idx, idx + len(chunk_text)

    anchor = re.sub(r"\s+", " ", chunk_text).strip()[:40]
    if anchor:
        idx = full_text.find(anchor)
        if idx != -1:
            return idx, idx + len(chunk_text)

    return None, None


def main():
    print("Loading CUAD benchmark...")
    tests = load_cuad_benchmark(CUAD_BENCHMARK_JSON)
    if LIMIT_QUERIES:
        tests = tests[:LIMIT_QUERIES]
    file_paths = referenced_file_paths(tests)
    print(f"{len(tests)} queries over {len(file_paths)} contract files")

    print("Loading raw CUAD corpus text (kept unmodified, on purpose — see "
          "module docstring re: clean_documents)...")
    corpus_texts = load_corpus_texts(CUAD_CORPUS_DIR, file_paths)

    print("Wrapping as Documents (source = exact file_path from cuad.json, "
          "so it matches the gold snippets exactly)...")
    raw_docs = [
        Document(page_content=text, metadata={"source": fp})
        for fp, text in corpus_texts.items()
    ]

    print("Chunking with YOUR existing chunker (unchanged: same size, same "
          "overlap)...")
    chunks = chunk_documents(raw_docs)
    print(f"{len(chunks)} chunks created")

    print("Loading YOUR embedding model (unchanged)...")
    embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building a FRESH FAISS index over CUAD chunks only (separate "
          "from, and not touching, your saved enterprise-PDF index)...")
    vectorstore = FAISS.from_documents(chunks, embedder)

    print("Building YOUR hybrid retriever (same function, same logic, "
          f"top_k={TOP_K})...")
    hybrid = build_hybrid_retriever(chunks, vectorstore, top_k=TOP_K)

    print(f"\nRunning retrieval over {len(tests)} queries...")
    precisions, recalls = [], []
    qa_results = []
    skipped_chunks = 0

    for i, qa_gt in enumerate(tests, 1):
        candidates = hybrid.invoke(qa_gt.query)[:TOP_K]  # your retriever, unchanged

        retrieved = []
        for doc in candidates:
            file_path = doc.metadata.get("source", "")
            full_text = corpus_texts.get(file_path, "")
            start, end = (
                find_offset(full_text, doc.page_content) if full_text else (None, None)
            )
            if start is None:
                skipped_chunks += 1
                continue
            retrieved.append(
                RetrievedChunk(file_path=file_path, start=start, end=end, text=doc.page_content)
            )

        p, r = official_precision_recall_at_k(qa_gt.snippets, retrieved, k=TOP_K)
        precisions.append(p)
        recalls.append(r)
        qa_results.append((qa_gt, retrieved))

        if i % 50 == 0 or i == len(tests):
            print(f"  ...{i}/{len(tests)} queries done")

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    map10 = mean_average_precision(qa_results, k=TOP_K, min_overlap_ratio=0.0)
    sensitivity = map_at_k_sensitivity(qa_results, k=TOP_K)

    print("\n=== RESULTS (Official LegalBench-RAG CUAD benchmark) ===")
    print(f"Queries evaluated:            {len(tests)}")
    print(f"Chunks skipped (no offset):   {skipped_chunks}  "
          f"(high count here would mean your chunker alters text \u2014 "
          f"share chunker.py if this is large)")
    print(f"Precision@{TOP_K} (official formula): {avg_precision:.3f}")
    print(f"Recall@{TOP_K}    (official formula): {avg_recall:.3f}")
    print(f"MAP@{TOP_K}: {map10:.3f}")
    print(f"MAP@{TOP_K} sensitivity across thresholds: {sensitivity}")


if __name__ == "__main__":
    main()
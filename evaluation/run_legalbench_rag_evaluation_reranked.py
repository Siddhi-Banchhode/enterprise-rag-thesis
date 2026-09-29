"""
evaluation/run_legalbench_rag_evaluation_reranked.py
=======================================================
Extends the raw-retrieval evaluation to ALSO compute reranked metrics,
using YOUR existing Reranker class (unchanged), so you can report both:
  - "no rerank" MAP@10   -> compare against RAGLite's 65.2% baseline
  - "with rerank" MAP@10 -> compare against RAGLite's 71.9% (Cohere) figure

Both numbers come from a SINGLE indexing pass (the expensive ~40-min step),
so you don't need to rebuild the FAISS index twice.

How reranking is applied here, and why:
  - Your production pipeline retrieves top_k*2 candidates then reranks down
    to top_n=3. For this benchmark we need a reranked TOP-10 (not top-3), so
    we retrieve a WIDER candidate pool (RETRIEVE_POOL_SIZE=30) and rerank
    with Reranker(top_n=10) instead of your production top_n=3. This is the
    standard "retrieve broad, rerank narrow" pattern reranking is designed
    for — reranking only 10 candidates down to 10 would barely change
    anything, since nothing new enters the pool.
  - The "no rerank" row for this same script is scored on the first 10 of
    that same 30-candidate pool, so it's directly comparable to your
    earlier standalone raw-retrieval run.

Run:
    python evaluation/run_legalbench_rag_evaluation_reranked.py
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
from src.retrieval.reranker import Reranker  # unchanged

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
# EDIT THESE
# ---------------------------------------------------------------------------
CUAD_CORPUS_DIR = "/Users/siddhibanchhode/Documents/enterprise-rag/legalbench_data/corpus"
CUAD_BENCHMARK_JSON = "/Users/siddhibanchhode/Documents/enterprise-rag/legalbench_data/benchmarks/cuad.json"
LIMIT_QUERIES = None # start small; set to None for the full 4,042-query run

TOP_K = 10                 # final evaluated depth, matching RAGLite's MAP@10
RETRIEVE_POOL_SIZE = 30    # wider candidate pool the reranker chooses from


def find_offset(full_text: str, chunk_text: str):
    """Locate a chunk's position in its source file's raw text. Exact match
    first, whitespace-tolerant anchor fallback second."""
    idx = full_text.find(chunk_text)
    if idx != -1:
        return idx, idx + len(chunk_text)
    anchor = re.sub(r"\s+", " ", chunk_text).strip()[:40]
    if anchor:
        idx = full_text.find(anchor)
        if idx != -1:
            return idx, idx + len(chunk_text)
    return None, None


def docs_to_retrieved_chunks(docs, corpus_texts, skipped_counter):
    """Adapter: LangChain Documents -> RetrievedChunk objects with
    file_path/start/end, used by both the raw and reranked scoring paths."""
    out = []
    for doc in docs:
        file_path = doc.metadata.get("source", "")
        full_text = corpus_texts.get(file_path, "")
        start, end = find_offset(full_text, doc.page_content) if full_text else (None, None)
        if start is None:
            skipped_counter[0] += 1
            continue
        out.append(RetrievedChunk(file_path=file_path, start=start, end=end, text=doc.page_content))
    return out


def summarize(name, qa_results, precisions, recalls, k):
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    map_k = mean_average_precision(qa_results, k=k, min_overlap_ratio=0.0)
    sens = map_at_k_sensitivity(qa_results, k=k)
    print(f"\n--- {name} ---")
    print(f"Precision@{k}: {avg_p:.3f}")
    print(f"Recall@{k}:    {avg_r:.3f}")
    print(f"MAP@{k}:       {map_k:.3f}")
    print(f"MAP@{k} sensitivity: {sens}")
    return {"precision": avg_p, "recall": avg_r, "map": map_k, "sensitivity": sens}


def main():
    print("Loading CUAD benchmark...")
    tests = load_cuad_benchmark(CUAD_BENCHMARK_JSON)
    if LIMIT_QUERIES:
        tests = tests[:LIMIT_QUERIES]
    file_paths = referenced_file_paths(tests)
    print(f"{len(tests)} queries over {len(file_paths)} contract files")

    print("Loading raw CUAD corpus text (unmodified, on purpose)...")
    corpus_texts = load_corpus_texts(CUAD_CORPUS_DIR, file_paths)

    print("Wrapping as Documents (source = exact file_path from cuad.json)...")
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
          "from your enterprise-PDF index)...")
    vectorstore = FAISS.from_documents(chunks, embedder)

    print(f"Building TWO retriever wrappers over the same index: one at "
          f"top_k={TOP_K} (identical methodology to your completed raw run, "
          f"for a clean 'no rerank' baseline) and one at "
          f"top_k={RETRIEVE_POOL_SIZE} (wide pool, feeding the reranker only). "
          f"This avoids conflating pool-width effects with reranking effects.")
    hybrid_baseline = build_hybrid_retriever(chunks, vectorstore, top_k=TOP_K)
    hybrid_wide = build_hybrid_retriever(chunks, vectorstore, top_k=RETRIEVE_POOL_SIZE)

    print(f"Loading YOUR reranker (unchanged class), top_n={TOP_K} "
          f"(note: your production config uses top_n=3 \u2014 here we need "
          f"depth 10 to match MAP@10)...")
    reranker = Reranker(top_n=TOP_K)

    print(f"\nRunning retrieval + reranking over {len(tests)} queries...")
    raw_precisions, raw_recalls, raw_qa_results = [], [], []
    rerank_precisions, rerank_recalls, rerank_qa_results = [], [], []
    skipped = [0]  # shared mutable counter across both paths

    for i, qa_gt in enumerate(tests, 1):
        # --- "no rerank" row: SAME methodology as your completed raw run ---
        raw_top10 = hybrid_baseline.invoke(qa_gt.query)[:TOP_K]  # your retriever, unchanged
        raw_retrieved = docs_to_retrieved_chunks(raw_top10, corpus_texts, skipped)
        p, r = official_precision_recall_at_k(qa_gt.snippets, raw_retrieved, k=TOP_K)
        raw_precisions.append(p)
        raw_recalls.append(r)
        raw_qa_results.append((qa_gt, raw_retrieved))

        # --- "with rerank" row: your Reranker applied to a WIDER, separate pool ---
        wide_pool = hybrid_wide.invoke(qa_gt.query)[:RETRIEVE_POOL_SIZE]
        reranked_top10 = reranker.rerank(qa_gt.query, wide_pool)  # your class, unchanged
        rerank_retrieved = docs_to_retrieved_chunks(reranked_top10, corpus_texts, skipped)
        p2, r2 = official_precision_recall_at_k(qa_gt.snippets, rerank_retrieved, k=TOP_K)
        rerank_precisions.append(p2)
        rerank_recalls.append(r2)
        rerank_qa_results.append((qa_gt, rerank_retrieved))

        if i % 50 == 0 or i == len(tests):
            print(f"  ...{i}/{len(tests)} queries done")

    print("\n=== RESULTS (Official LegalBench-RAG CUAD benchmark) ===")
    print(f"Queries evaluated: {len(tests)}")
    print(f"Chunks skipped (no offset, across both passes): {skipped[0]}")

    raw_summary = summarize("NO RERANK (compare to RAGLite 65.2% / 65.9% w/ query adapter)",
                             raw_qa_results, raw_precisions, raw_recalls, TOP_K)
    rerank_summary = summarize("WITH RERANK (compare to RAGLite 71.9% w/ Cohere)",
                                rerank_qa_results, rerank_precisions, rerank_recalls, TOP_K)

    return {"no_rerank": raw_summary, "with_rerank": rerank_summary}


if __name__ == "__main__":
    main()
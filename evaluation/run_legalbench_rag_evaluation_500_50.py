"""
evaluation/run_legalbench_rag_evaluation_500_50.py
=============================================
PRODUCTION-CONFIG RERUN of the BM25/FAISS weight ablation.

This is identical to run_legalbench_rag_evaluation.py EXCEPT:
  - CHUNK_SIZE / CHUNK_OVERLAP are set to your actual production values
    (500 / 50) instead of the earlier chunk-size-experiment values
    (2000 / 200).
  - Results are additionally written to a CSV so they can be merged with
    your other evaluation tables (RAGAS, pdfQA-Benchmark, SQuAD2) without
    retyping numbers from the console.

Why this run exists:
  The 2000/200 run gave MAP@10=0.210 (best) but with an oddly low P@10
  (0.007-0.016) relative to Recall@10 (0.241-0.449). The likely cause is
  chunk size: LegalBench-RAG gold spans are short, so a 2000-char chunk
  can contain the gold span (inflating recall) while being mostly
  irrelevant text (deflating character-level precision). Rerunning at
  500/50 isolates whether that gap closes, and makes MAP@10 here directly
  comparable to your earlier baseline (0.102 no-rerank / 0.086 reranked),
  which also used 500/50.

Reuses your real classes directly, same as the original script:
  - config.EMBEDDING_MODEL          (same embedding model)
  - HuggingFaceEmbeddings           (same embedder wrapper)
  - chunk_documents()               (same chunker, now with prod size/overlap)
  - build_hybrid_retriever()        (same BM25+FAISS hybrid logic)

Same two judgment calls as before (see original script's docstring):
  1. clean_documents() is intentionally SKIPPED — avoids shifting character
     positions relative to the benchmark's raw-text gold spans.
  2. Chunk offsets are recovered via string search against the raw corpus
     text, with a whitespace-tolerant fallback.

Run:
    python evaluation/run_legalbench_rag_evaluation_500_50.py
"""
import csv
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
LIMIT_QUERIES = None  # start small to catch bugs fast; set to None for the full 4,042-query run
TOP_K = 10  # MAP@10 and the official benchmark both score raw top-10, pre-reranker

# Production chunk config (matches your saved enterprise-PDF index settings)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Candidate pool
CANDIDATE_K = 10

# BM25 / FAISS weight ablation
WEIGHT_CONFIGS = [
    (0.0, 1.0),
    (0.2, 0.8),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.8, 0.2),
    (1.0, 0.0),
]

# Where to save the ablation table for later reconciliation with your other
# evaluation results (RAGAS, pdfQA-Benchmark, SQuAD2)
RESULTS_CSV = str(Path(__file__).parent / "legalbench_rag_500_50_results.csv")


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

    print(f"Chunking with YOUR existing chunker at PRODUCTION config "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(
        raw_docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
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

    # ---------------------------------------------------------------------------
    # BM25 / FAISS WEIGHT ABLATION
    # ---------------------------------------------------------------------------
    ablation_results = []

    for bm25_weight, dense_weight in WEIGHT_CONFIGS:
        print("\n" + "=" * 70)
        print(
            f"WEIGHT EXPERIMENT: BM25={bm25_weight:.1f} | "
            f"FAISS={dense_weight:.1f}"
        )
        print("=" * 70)

        print(
            f"Building hybrid retriever "
            f"(candidate_k={CANDIDATE_K}, top_k={TOP_K})..."
        )
        hybrid = build_hybrid_retriever(
            chunks,
            vectorstore,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

        print(f"Running retrieval over {len(tests)} queries...")
        precisions = []
        recalls = []
        qa_results = []
        skipped_chunks = 0

        for i, qa_gt in enumerate(tests, 1):
            candidates = hybrid.invoke(qa_gt.query)[:TOP_K]

            retrieved = []
            for doc in candidates:
                file_path = doc.metadata.get("source", "")
                full_text = corpus_texts.get(file_path, "")
                start, end = (
                    find_offset(full_text, doc.page_content)
                    if full_text
                    else (None, None)
                )
                if start is None:
                    skipped_chunks += 1
                    continue
                retrieved.append(
                    RetrievedChunk(
                        file_path=file_path,
                        start=start,
                        end=end,
                        text=doc.page_content,
                    )
                )

            p, r = official_precision_recall_at_k(
                qa_gt.snippets,
                retrieved,
                k=TOP_K,
            )
            precisions.append(p)
            recalls.append(r)
            qa_results.append((qa_gt, retrieved))

            if i % 50 == 0 or i == len(tests):
                print(f"  ...{i}/{len(tests)} queries done")

        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        map10 = mean_average_precision(
            qa_results,
            k=TOP_K,
            min_overlap_ratio=0.0,
        )
        sensitivity = map_at_k_sensitivity(
            qa_results,
            k=TOP_K,
        )

        ablation_results.append({
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "bm25": bm25_weight,
            "faiss": dense_weight,
            "precision": avg_precision,
            "recall": avg_recall,
            "map": map10,
            "sensitivity": sensitivity,
            "skipped": skipped_chunks,
        })

        print("\nRESULT")
        print(f"BM25 weight:       {bm25_weight:.1f}")
        print(f"FAISS weight:      {dense_weight:.1f}")
        print(f"Precision@{TOP_K}: {avg_precision:.3f}")
        print(f"Recall@{TOP_K}:    {avg_recall:.3f}")
        print(f"MAP@{TOP_K}:       {map10:.3f}")
        print(f"Skipped chunks:    {skipped_chunks}")

    # ---------------------------------------------------------------------------
    # FINAL ABLATION SUMMARY
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print(f"BM25 / FAISS WEIGHT ABLATION SUMMARY (chunk_size={CHUNK_SIZE}, "
          f"overlap={CHUNK_OVERLAP})")
    print("=" * 80)
    print(
        f"{'BM25':>8} "
        f"{'FAISS':>8} "
        f"{'P@10':>10} "
        f"{'Recall@10':>12} "
        f"{'MAP@10':>10}"
    )
    print("-" * 80)
    for result in ablation_results:
        print(
            f"{result['bm25']:>8.1f} "
            f"{result['faiss']:>8.1f} "
            f"{result['precision']:>10.3f} "
            f"{result['recall']:>12.3f} "
            f"{result['map']:>10.3f}"
        )

    # Find best configuration by MAP@10
    best = max(ablation_results, key=lambda x: x["map"])
    print("\nBest configuration by MAP@10:")
    print(
        f"BM25={best['bm25']:.1f} | "
        f"FAISS={best['faiss']:.1f} | "
        f"MAP@10={best['map']:.3f}"
    )

    # ---------------------------------------------------------------------------
    # SAVE TO CSV — for merging into your Chapter 5 results table alongside
    # RAGAS, pdfQA-Benchmark, and SQuAD2 numbers
    # ---------------------------------------------------------------------------
    fieldnames = [
        "chunk_size", "chunk_overlap", "bm25", "faiss",
        "precision", "recall", "map", "sensitivity", "skipped",
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_results)
    print(f"\nResults saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
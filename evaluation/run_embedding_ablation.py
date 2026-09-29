"""
evaluation/run_embedding_ablation.py
====================================

Controlled embedding-model ablation for the LegalBench-RAG CUAD benchmark.

Purpose
-------
Tests whether changing the embedding model improves retrieval performance.

Everything is kept fixed except the embedding model:

    Dataset              : LegalBench-RAG CUAD
    Queries              : first 50
    Contract files       : determined by benchmark queries
    Corpus               : raw CUAD corpus
    Chunk size           : 2000 characters
    Chunk overlap        : 200 characters
    Retrieval            : FAISS only
    BM25 weight          : 0.0
    FAISS weight         : 1.0
    Candidate K          : 10
    Top K                : 10
    Embedding normalization: True
    Device               : CPU

Embedding models tested
-----------------------
    1. all-mpnet-base-v2
    2. bge-base-en-v1.5
    3. bge-large-en-v1.5
    4. e5-base-v2

IMPORTANT
---------
This is a controlled model-replacement experiment.

BGE/E5-specific query/document prompting is intentionally NOT added here,
because doing so would introduce model-specific preprocessing as another
experimental variable.

A separate optimized BGE/E5 experiment can be performed later if required.

The evaluation uses the same LegalBench-RAG relevance definition already
implemented in:

    evaluation/legalbench_rag_cuad_evaluator.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

import config

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.retrieval.hybrid_retriever import build_hybrid_retriever

from evaluation.legalbench_rag_cuad_evaluator import (
    load_cuad_benchmark,
    referenced_file_paths,
    load_corpus_texts,
    official_precision_recall_at_k,
    mean_average_precision,
    map_at_k_sensitivity,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CUAD_CORPUS_DIR = (
    PROJECT_ROOT
    / "legalbench_data"
    / "corpus"
)

CUAD_BENCHMARK_JSON = (
    PROJECT_ROOT
    / "legalbench_data"
    / "benchmarks"
    / "cuad.json"
)

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

LIMIT_QUERIES = 50

TOP_K = 10
CANDIDATE_K = 10

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# FAISS-only because previous ablation showed that BM25 did not improve MAP@10.
BM25_WEIGHT = 0.0
DENSE_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = [
    "all-mpnet-base-v2",
    "bge-base-en-v1.5",
    "bge-large-en-v1.5",
    "e5-base-v2",
]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "embedding_ablation_results.csv"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def separator(char: str = "=", width: int = 78) -> None:
    print(char * width)


def find_offset(
    chunk_text: str,
    source_text: str,
) -> Tuple[int, int]:
    """
    Find the character offsets of a retrieved chunk inside the original
    source document.

    First attempts exact matching.

    Then attempts the same whitespace-normalized fallback used by the
    LegalBench evaluation workflow.

    Returns:
        (start, end)

    If no match is found:
        (-1, -1)
    """

    if not chunk_text or not source_text:
        return -1, -1

    # ---------------------------------------------------------------
    # Exact match
    # ---------------------------------------------------------------

    start = source_text.find(chunk_text)

    if start != -1:
        return start, start + len(chunk_text)

    # ---------------------------------------------------------------
    # Whitespace-normalized fallback
    # ---------------------------------------------------------------

    import re

    normalized_chunk = re.sub(r"\s+", " ", chunk_text).strip()
    normalized_source = re.sub(r"\s+", " ", source_text)

    if not normalized_chunk:
        return -1, -1

    normalized_start = normalized_source.find(
        normalized_chunk[:40]
    )

    if normalized_start == -1:
        return -1, -1

    # ---------------------------------------------------------------
    # Map normalized position approximately back to original text.
    #
    # We deliberately avoid simply doing:
    #
    #     start = normalized_start
    #
    # because whitespace normalization changes character positions.
    # ---------------------------------------------------------------

    # Build a normalized-to-original character mapping.
    mapping = []

    i = 0
    previous_was_space = False

    while i < len(source_text):
        char = source_text[i]

        if char.isspace():
            if not previous_was_space:
                mapping.append(i)
                previous_was_space = True
        else:
            mapping.append(i)
            previous_was_space = False

        i += 1

    if normalized_start >= len(mapping):
        return -1, -1

    original_start = mapping[normalized_start]

    # Use the original chunk length as an approximation for the end.
    original_end = min(
        len(source_text),
        original_start + len(chunk_text),
    )

    return original_start, original_end


def document_to_offsets(
    document: Document,
    corpus_texts: Dict[str, str],
):
    """
    Convert a retrieved LangChain Document into the offset representation
    expected by the LegalBench evaluator.
    """

    source = document.metadata.get("source")

    if not source:
        return None

    full_text = corpus_texts.get(source)

    if full_text is None:
        return None

    chunk_text = document.page_content

    start, end = find_offset(
        chunk_text,
        full_text,
    )

    if start < 0 or end <= start:
        return None

    return {
        "source": source,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------------
# Load benchmark
# ---------------------------------------------------------------------------

def prepare_benchmark():
    print()
    separator()

    print("Loading LegalBench-RAG CUAD benchmark...")

    benchmark = load_cuad_benchmark(
        str(CUAD_BENCHMARK_JSON)
    )

    print(
        f"Total benchmark queries available: {len(benchmark)}"
    )

    queries = benchmark[:LIMIT_QUERIES]

    print(
        f"Queries selected for this experiment: {len(queries)}"
    )

    # ---------------------------------------------------------------
    # Determine required contract files
    # ---------------------------------------------------------------

    file_paths = referenced_file_paths(queries)

    print(
        f"Contract files referenced by selected queries: "
        f"{len(file_paths)}"
    )

    # ---------------------------------------------------------------
    # Load raw corpus
    # ---------------------------------------------------------------

    print("Loading raw CUAD corpus text...")

    corpus_texts = load_corpus_texts(
        str(CUAD_CORPUS_DIR),
        file_paths,
    )

    print(
        f"Loaded raw text for {len(corpus_texts)} contract files."
    )

    # ---------------------------------------------------------------
    # Wrap corpus as Documents
    #
    # IMPORTANT:
    # source metadata must remain the exact benchmark file path.
    # ---------------------------------------------------------------

    raw_documents: List[Document] = []

    for file_path in file_paths:

        text = corpus_texts.get(file_path)

        if text is None:
            print(
                f"WARNING: corpus text missing for {file_path}"
            )
            continue

        raw_documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path
                },
            )
        )

    print(
        f"Created {len(raw_documents)} raw Documents."
    )

    # ---------------------------------------------------------------
    # Chunk using the same chunking configuration for every model
    # ---------------------------------------------------------------

    print()
    print(
        f"Chunking documents: "
        f"size={CHUNK_SIZE}, "
        f"overlap={CHUNK_OVERLAP}"
    )

    # Import YOUR existing production chunker.
    from src.preprocessing.chunker import chunk_documents

    chunks = chunk_documents(
        raw_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    return queries, corpus_texts, chunks


# ---------------------------------------------------------------------------
# Build embeddings
# ---------------------------------------------------------------------------

def build_embeddings(model_name: str):
    """
    Build the same LangChain HuggingFaceEmbeddings wrapper used by the
    retrieval pipeline.
    """

    print()
    print(f"Loading embedding model: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


# ---------------------------------------------------------------------------
# Build FAISS
# ---------------------------------------------------------------------------

def build_faiss(
    chunks: List[Document],
    embeddings,
):
    """
    Build a fresh FAISS index for the current embedding model.

    A new index is required for every embedding model because the vector
    dimensionality and vector representation can differ.
    """

    print(
        f"Building FAISS index for {len(chunks)} chunks..."
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    print(
        f"FAISS index ready: {vectorstore.index.ntotal} vectors"
    )

    return vectorstore


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve_all_queries(
    queries,
    chunks,
    corpus_texts,
    vectorstore,
):
    """
    Retrieve top-K chunks for every query using the existing retriever.

    BM25 = 0
    FAISS = 1

    Therefore this is effectively a dense-only retrieval experiment while
    still using the project's existing retriever construction.
    """

    retriever = build_hybrid_retriever(
        chunks=chunks,
        faiss_vectorstore=vectorstore,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        bm25_weight=BM25_WEIGHT,
        dense_weight=DENSE_WEIGHT,
    )

    print(
        "Retriever ready "
        f"(BM25={BM25_WEIGHT:.1f}, "
        f"FAISS={DENSE_WEIGHT:.1f}, "
        f"candidate_k={CANDIDATE_K}, "
        f"top_k={TOP_K})"
    )

    all_results = []

    skipped_chunks = 0

    for index, item in enumerate(queries, start=1):

        # -----------------------------------------------------------
        # Handle benchmark item formats.
        #
        # The official CUAD benchmark loader normally provides
        # dictionaries containing the question.
        # -----------------------------------------------------------

        if isinstance(item, dict):

            question = (
                item.get("question")
                or item.get("query")
                or item.get("text")
            )

        else:
            question = str(item)

        if not question:
            print(
                f"WARNING: query {index} has no question text."
            )
            continue

        # -----------------------------------------------------------
        # Retrieval
        # -----------------------------------------------------------

        docs = retriever.invoke(question)

        retrieved_offsets = []

        for doc in docs[:TOP_K]:

            offset = document_to_offsets(
                doc,
                corpus_texts,
            )

            if offset is None:
                skipped_chunks += 1
                continue

            retrieved_offsets.append(offset)

        all_results.append(
            {
                "question": question,
                "retrieved": retrieved_offsets,
                "benchmark_item": item,
            }
        )

        if index % 10 == 0 or index == len(queries):

            print(
                f"  Processed {index}/{len(queries)} queries"
            )

    return all_results, skipped_chunks


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_results(
    retrieval_results,
    benchmark_items,
):
    """
    Calculate the LegalBench-RAG precision/recall and MAP metrics.

    The evaluator's relevance definition is based on character-level
    overlap with the official gold answer spans.
    """

    precisions = []
    recalls = []
    average_precisions = []

    # ---------------------------------------------------------------
    # Evaluate every query
    # ---------------------------------------------------------------

    for result, benchmark_item in zip(
        retrieval_results,
        benchmark_items,
    ):

        retrieved = result["retrieved"]

        # -----------------------------------------------------------
        # Official precision / recall
        # -----------------------------------------------------------

        precision, recall = official_precision_recall_at_k(
            benchmark_item,
            retrieved,
            k=TOP_K,
        )

        precisions.append(precision)
        recalls.append(recall)

        # -----------------------------------------------------------
        # MAP
        # -----------------------------------------------------------

        ap = mean_average_precision(
            benchmark_item,
            retrieved,
            k=TOP_K,
        )

        average_precisions.append(ap)

    # ---------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------

    precision_at_10 = (
        sum(precisions) / len(precisions)
        if precisions
        else 0.0
    )

    recall_at_10 = (
        sum(recalls) / len(recalls)
        if recalls
        else 0.0
    )

    map_at_10 = (
        sum(average_precisions) / len(average_precisions)
        if average_precisions
        else 0.0
    )

    return (
        precision_at_10,
        recall_at_10,
        map_at_10,
    )


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_result(row: Dict):
    """
    Append one embedding-model result to CSV.
    """

    file_exists = RESULTS_CSV.exists()

    fieldnames = [
        "model",
        "precision_at_10",
        "recall_at_10",
        "map_at_10",
        "skipped_chunks",
        "load_seconds",
        "faiss_build_seconds",
        "retrieval_seconds",
        "total_seconds",
    ]

    with open(
        RESULTS_CSV,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():

    separator()
    print("EMBEDDING MODEL ABLATION")
    separator()

    print()
    print("Experimental configuration:")
    print(f"  Queries       : {LIMIT_QUERIES}")
    print(f"  Top K         : {TOP_K}")
    print(f"  Candidate K   : {CANDIDATE_K}")
    print(f"  Chunk size    : {CHUNK_SIZE}")
    print(f"  Chunk overlap : {CHUNK_OVERLAP}")
    print(f"  BM25 weight   : {BM25_WEIGHT}")
    print(f"  FAISS weight  : {DENSE_WEIGHT}")
    print()

    print("Models:")
    for model in EMBEDDING_MODELS:
        print(f"  - {model}")

    # ---------------------------------------------------------------
    # Prepare benchmark once.
    #
    # The same chunks and queries are reused for every model.
    # ---------------------------------------------------------------

    queries, corpus_texts, chunks = prepare_benchmark()

    all_summary_results = []

    # ---------------------------------------------------------------
    # Run each embedding model
    # ---------------------------------------------------------------

    for model_number, model_name in enumerate(
        EMBEDDING_MODELS,
        start=1,
    ):

        separator("-")

        print(
            f"MODEL {model_number}/{len(EMBEDDING_MODELS)}"
        )

        print(f"Embedding model: {model_name}")

        model_start = time.perf_counter()

        try:

            # -------------------------------------------------------
            # Load model
            # -------------------------------------------------------

            load_start = time.perf_counter()

            embeddings = build_embeddings(
                model_name
            )

            load_seconds = (
                time.perf_counter() - load_start
            )

            print(
                f"Embedding model loaded in "
                f"{load_seconds:.2f}s"
            )

            # -------------------------------------------------------
            # Build FAISS
            # -------------------------------------------------------

            faiss_start = time.perf_counter()

            vectorstore = build_faiss(
                chunks,
                embeddings,
            )

            faiss_build_seconds = (
                time.perf_counter() - faiss_start
            )

            print(
                f"FAISS built in "
                f"{faiss_build_seconds:.2f}s"
            )

            # -------------------------------------------------------
            # Retrieval
            # -------------------------------------------------------

            retrieval_start = time.perf_counter()

            retrieval_results, skipped_chunks = (
                retrieve_all_queries(
                    queries,
                    chunks,
                    corpus_texts,
                    vectorstore,
                )
            )

            retrieval_seconds = (
                time.perf_counter() - retrieval_start
            )

            print(
                f"Retrieval completed in "
                f"{retrieval_seconds:.2f}s"
            )

            # -------------------------------------------------------
            # Evaluation
            # -------------------------------------------------------

            print("Calculating evaluation metrics...")

            precision_at_10, recall_at_10, map_at_10 = (
                evaluate_results(
                    retrieval_results,
                    queries,
                )
            )

            total_seconds = (
                time.perf_counter() - model_start
            )

            # -------------------------------------------------------
            # Result
            # -------------------------------------------------------

            result = {
                "model": model_name,
                "precision_at_10": round(
                    precision_at_10,
                    6,
                ),
                "recall_at_10": round(
                    recall_at_10,
                    6,
                ),
                "map_at_10": round(
                    map_at_10,
                    6,
                ),
                "skipped_chunks": skipped_chunks,
                "load_seconds": round(
                    load_seconds,
                    2,
                ),
                "faiss_build_seconds": round(
                    faiss_build_seconds,
                    2,
                ),
                "retrieval_seconds": round(
                    retrieval_seconds,
                    2,
                ),
                "total_seconds": round(
                    total_seconds,
                    2,
                ),
            }

            all_summary_results.append(result)

            # -------------------------------------------------------
            # Save immediately
            # -------------------------------------------------------

            save_result(result)

            print()
            print("RESULT")
            print(f"  P@10       : {precision_at_10:.3f}")
            print(f"  Recall@10  : {recall_at_10:.3f}")
            print(f"  MAP@10     : {map_at_10:.3f}")
            print(
                f"  Skipped    : {skipped_chunks}"
            )
            print(
                f"  Total time : {total_seconds:.2f}s"
            )

        except Exception as exc:

            print()
            print(
                f"ERROR while evaluating {model_name}:"
            )
            print(str(exc))

            traceback.print_exc()

            print()
            print(
                "Continuing with the next embedding model..."
            )

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------

    separator()

    print("FINAL EMBEDDING ABLATION RESULTS")
    separator()

    print()
    print(
        f"{'Model':32}"
        f"{'P@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MAP@10':>10}"
        f"{'Skipped':>10}"
    )

    print("-" * 78)

    for result in all_summary_results:

        print(
            f"{result['model']:32}"
            f"{result['precision_at_10']:10.3f}"
            f"{result['recall_at_10']:12.3f}"
            f"{result['map_at_10']:10.3f}"
            f"{result['skipped_chunks']:10}"
        )

    # ----------------------------------------------------------------
    # Identify best model
    # ----------------------------------------------------------------

    if all_summary_results:

        best = max(
            all_summary_results,
            key=lambda x: x["map_at_10"],
        )

        print()
        print(
            f"Best model by MAP@10: {best['model']}"
        )

        print(
            f"Best MAP@10: {best['map_at_10']:.3f}"
        )

        print(
            f"Best Recall@10: "
            f"{best['recall_at_10']:.3f}"
        )

        print()
        print(
            "Results saved to:"
        )
        print(
            f"  {RESULTS_CSV}"
        )

    else:

        print()
        print("No embedding model completed successfully.")

    separator()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
"""
evaluation/run_reranker_evaluation.py
======================================

Evaluates the effect of the Cross-Encoder Reranker on the
LegalBench-RAG CUAD benchmark.

Experimental pipeline
---------------------

    CUAD query
        |
        v
    BM25 + FAISS Hybrid Retriever
        |
        | Top 10 candidates
        v
    BAAI/bge-reranker-base
        |
        | Top 3
        v
    LegalBench-RAG evaluation

The following components are reused from the existing system:

  - config.EMBEDDING_MODEL
  - HuggingFaceEmbeddings
  - chunk_documents()
  - build_hybrid_retriever()
  - Reranker

The reranker itself is imported from:

    src.retrieval.reranker

Evaluation metrics:

  - Precision@3
  - Recall@3
  - MAP@3

A baseline Hybrid Top-3 result is also calculated so that the
effect of reranking can be measured directly.

Baseline:
    Hybrid Retriever -> Top 3

Reranked:
    Hybrid Retriever -> Top 10 -> BGE Reranker -> Top 3


Run from project root:

    python3 evaluation/run_reranker_evaluation.py

For a full LegalBench-RAG CUAD evaluation:

    Set LIMIT_QUERIES = None

"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from config import config

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.preprocessing.chunker import chunk_documents
from src.retrieval.hybrid_retriever import build_hybrid_retriever
from src.retrieval.reranker import Reranker

from legalbench_rag_cuad_evaluator import (
    load_cuad_benchmark,
    referenced_file_paths,
    load_corpus_texts,
    RetrievedChunk,
    official_precision_recall_at_k,
    mean_average_precision,
    map_at_k_sensitivity,
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# CUAD raw corpus directory
CUAD_CORPUS_DIR = (
    "/Users/siddhibanchhode/Documents/enterprise-rag/"
    "legalbench_data/corpus"
)

# LegalBench-RAG CUAD benchmark
CUAD_BENCHMARK_JSON = (
    "/Users/siddhibanchhode/Documents/enterprise-rag/"
    "legalbench_data/benchmarks/cuad.json"
)

# ---------------------------------------------------------------------------
# Query limit
# ---------------------------------------------------------------------------
# Use 50 while debugging/testing.
#
# For the final thesis experiment:
#
#     LIMIT_QUERIES = None
#
# which evaluates all 4,042 queries.

LIMIT_QUERIES = 50


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

# Number of documents retrieved by HybridRetriever before reranking.
CANDIDATE_K = 10

# Number of documents retained after reranking.
RERANKER_TOP_N = 3

# Number of documents used for final evaluation.
TOP_K = 3


# ---------------------------------------------------------------------------
# Hybrid retrieval weights
# ---------------------------------------------------------------------------
#
# IMPORTANT:
#
# This should normally be the BEST hybrid configuration identified from
# your BM25 / FAISS weight ablation.
#
# If your previous experiment showed BM25=0.5 / FAISS=0.5 as the best,
# keep these values.
#
# Change these two values if your weight ablation produced a different
# best configuration.
#
BM25_WEIGHT = 0.0
DENSE_WEIGHT = 1.0

CANDIDATE_K = 10
RERANKER_TOP_N = 3
TOP_K = 3

RERANKER_MODEL = "BAAI/bge-reranker-base"


# =============================================================================
# OFFSET RECOVERY
# =============================================================================

def find_offset(full_text: str, chunk_text: str):
    """
    Locate a retrieved chunk inside the original CUAD raw text.

    The LegalBench-RAG CUAD evaluation uses character-level overlap
    between retrieved text spans and gold answer spans.

    Therefore, we need the character start/end position of every
    retrieved chunk.

    Strategy:
        1. Try exact string matching.
        2. If that fails, use a whitespace-normalized anchor.

    Returns
    -------
    tuple
        (start, end)

        or

        (None, None) if the chunk cannot be located.
    """

    # -----------------------------------------------------------------------
    # Exact match
    # -----------------------------------------------------------------------

    idx = full_text.find(chunk_text)

    if idx != -1:
        return idx, idx + len(chunk_text)

    # -----------------------------------------------------------------------
    # Whitespace-tolerant fallback
    # -----------------------------------------------------------------------

    normalized_chunk = re.sub(r"\s+", " ", chunk_text).strip()

    anchor = normalized_chunk[:40]

    if anchor:

        idx = full_text.find(anchor)

        if idx != -1:
            return idx, idx + len(chunk_text)

    return None, None


# =============================================================================
# CONVERT LANGCHAIN DOCUMENTS TO LEGALBENCH RETRIEVED CHUNKS
# =============================================================================

def convert_documents_to_retrieved_chunks(
    documents,
    corpus_texts,
):
    """
    Convert LangChain Documents into RetrievedChunk objects required
    by the LegalBench-RAG evaluator.

    Parameters
    ----------
    documents:
        Retrieved LangChain Documents.

    corpus_texts:
        Dictionary mapping source file paths to raw CUAD text.

    Returns
    -------
    retrieved:
        List[RetrievedChunk]

    skipped:
        Number of documents for which an offset could not be recovered.
    """

    retrieved = []

    skipped = 0

    for doc in documents:

        file_path = doc.metadata.get("source", "")

        full_text = corpus_texts.get(file_path, "")

        if not full_text:
            skipped += 1
            continue

        start, end = find_offset(
            full_text,
            doc.page_content,
        )

        if start is None:
            skipped += 1
            continue

        retrieved.append(
            RetrievedChunk(
                file_path=file_path,
                start=start,
                end=end,
                text=doc.page_content,
            )
        )

    return retrieved, skipped


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")
    print("=" * 80)
    print("LEGALBENCH-RAG CUAD — RERANKER EVALUATION")
    print("=" * 80)

    print(
        """
Experimental pipeline:

    CUAD Query
        |
        v
    BM25 + FAISS Hybrid
        |
        | Top 10
        v
    BAAI/bge-reranker-base
        |
        | Top 3
        v
    LegalBench-RAG Evaluation
"""
    )

    print("=" * 80)


    # =========================================================================
    # 1. LOAD BENCHMARK
    # =========================================================================

    print("\n[1/7] Loading CUAD benchmark...")

    tests = load_cuad_benchmark(
        CUAD_BENCHMARK_JSON
    )

    total_queries = len(tests)

    if LIMIT_QUERIES is not None:

        tests = tests[:LIMIT_QUERIES]

    print(
        f"Full benchmark contains {total_queries} queries."
    )

    print(
        f"Using {len(tests)} queries for this experiment."
    )


    # =========================================================================
    # 2. DETERMINE REQUIRED CORPUS FILES
    # =========================================================================

    file_paths = referenced_file_paths(tests)

    print(
        f"{len(tests)} queries over "
        f"{len(file_paths)} contract files"
    )


    # =========================================================================
    # 3. LOAD RAW CUAD CORPUS
    # =========================================================================

    print(
        "\n[2/7] Loading raw CUAD corpus text..."
    )

    print(
        "IMPORTANT: raw text is kept unmodified so that character "
        "offsets remain compatible with LegalBench-RAG gold spans."
    )

    corpus_texts = load_corpus_texts(
        CUAD_CORPUS_DIR,
        file_paths,
    )


    # =========================================================================
    # 4. WRAP CORPUS AS LANGCHAIN DOCUMENTS
    # =========================================================================

    print(
        "\n[3/7] Creating LangChain Documents..."
    )

    raw_docs = [
        Document(
            page_content=text,
            metadata={
                "source": fp
            },
        )
        for fp, text in corpus_texts.items()
    ]

    print(
        f"Loaded {len(raw_docs)} contract documents."
    )


    # =========================================================================
    # 5. CHUNK DOCUMENTS
    # =========================================================================

    print(
        "\n[4/7] Chunking CUAD documents..."
    )

    print(
        "Using the configured chunk size and overlap."
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # These values should match the chunk configuration you used in your
    # current experiment.
    #
    # Based on your current evaluation setup:
    #
    #     chunk_size = 2000
    #     chunk_overlap = 200
    # -------------------------------------------------------------------------

    chunks = chunk_documents(
        raw_docs,
        chunk_size=2000,
        chunk_overlap=200,
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    # =========================================================================
    # 6. LOAD EMBEDDING MODEL
    # =========================================================================

    print(
        "\n[5/7] Loading embedding model..."
    )

    print(
        f"Embedding model: {config.EMBEDDING_MODEL}"
    )

    embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


    # =========================================================================
    # 7. BUILD FRESH FAISS INDEX
    # =========================================================================

    print(
        "\n[6/7] Building fresh FAISS index..."
    )

    print(
        "This index contains only the CUAD chunks and does not modify "
        "your saved enterprise-PDF FAISS index."
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embedder,
    )

    print(
        f"FAISS index contains "
        f"{vectorstore.index.ntotal} vectors."
    )


    # =========================================================================
    # 8. BUILD HYBRID RETRIEVER
    # =========================================================================

    print(
        "\n[7/7] Building Hybrid Retriever..."
    )

    print(
        f"Candidate K: {CANDIDATE_K}"
    )

    print(
        f"BM25 weight: {BM25_WEIGHT:.2f}"
    )

    print(
        f"FAISS weight: {DENSE_WEIGHT:.2f}"
    )

    hybrid = build_hybrid_retriever(
        chunks,
        vectorstore,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K,
        bm25_weight=BM25_WEIGHT,
        dense_weight=DENSE_WEIGHT,
    )


    # =========================================================================
    # LOAD RERANKER
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Loading Cross-Encoder Reranker..."
    )

    print(
        f"Reranker model: {RERANKER_MODEL}"
    )

    print(
        f"Reranker input: Top {CANDIDATE_K} hybrid candidates"
    )

    print(
        f"Reranker output: Top {RERANKER_TOP_N}"
    )

    reranker = Reranker(
        model_name=RERANKER_MODEL,
        top_n=RERANKER_TOP_N,
    )


    # =========================================================================
    # RESULT STORAGE
    # =========================================================================

    baseline_precisions = []
    baseline_recalls = []
    baseline_qa_results = []

    reranked_precisions = []
    reranked_recalls = []
    reranked_qa_results = []

    baseline_skipped = 0
    reranked_skipped = 0


    # =========================================================================
    # RUN EVALUATION
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Running evaluation over {len(tests)} queries..."
    )

    print(
        "=" * 80
    )


    for i, qa_gt in enumerate(tests, 1):

        # =====================================================================
        # HYBRID RETRIEVAL
        # =====================================================================

        candidates = hybrid.invoke(
            qa_gt.query
        )

        # Make absolutely sure we only pass the requested candidate pool
        # to the reranker.
        candidates = candidates[:CANDIDATE_K]


        # =====================================================================
        # BASELINE: HYBRID TOP-3
        # =====================================================================

        baseline_docs = candidates[:TOP_K]

        baseline_retrieved, skipped = (
            convert_documents_to_retrieved_chunks(
                baseline_docs,
                corpus_texts,
            )
        )

        baseline_skipped += skipped


        baseline_p, baseline_r = (
            official_precision_recall_at_k(
                qa_gt.snippets,
                baseline_retrieved,
                k=TOP_K,
            )
        )

        baseline_precisions.append(
            baseline_p
        )

        baseline_recalls.append(
            baseline_r
        )

        baseline_qa_results.append(
            (
                qa_gt,
                baseline_retrieved,
            )
        )


        # =====================================================================
        # RERANKING
        # =====================================================================

        reranked_docs = reranker.rerank(
            qa_gt.query,
            candidates,
        )

        # The Reranker itself applies top_n=3.
        reranked_docs = reranked_docs[:RERANKER_TOP_N]


        # =====================================================================
        # CONVERT RERANKED DOCUMENTS
        # =====================================================================

        reranked_retrieved, skipped = (
            convert_documents_to_retrieved_chunks(
                reranked_docs,
                corpus_texts,
            )
        )

        reranked_skipped += skipped


        # =====================================================================
        # RERANKED METRICS
        # =====================================================================

        reranked_p, reranked_r = (
            official_precision_recall_at_k(
                qa_gt.snippets,
                reranked_retrieved,
                k=RERANKER_TOP_N,
            )
        )

        reranked_precisions.append(
            reranked_p
        )

        reranked_recalls.append(
            reranked_r
        )

        reranked_qa_results.append(
            (
                qa_gt,
                reranked_retrieved,
            )
        )


        # =====================================================================
        # PROGRESS
        # =====================================================================

        if (
            i % 10 == 0
            or i == len(tests)
        ):

            print(
                f"  ... {i}/{len(tests)} queries completed"
            )


    # =========================================================================
    # CALCULATE BASELINE METRICS
    # =========================================================================

    baseline_precision = (
        sum(baseline_precisions)
        / len(baseline_precisions)
    )

    baseline_recall = (
        sum(baseline_recalls)
        / len(baseline_recalls)
    )

    baseline_map = mean_average_precision(
        baseline_qa_results,
        k=TOP_K,
        min_overlap_ratio=0.0,
    )

    baseline_sensitivity = map_at_k_sensitivity(
        baseline_qa_results,
        k=TOP_K,
    )


    # =========================================================================
    # CALCULATE RERANKED METRICS
    # =========================================================================

    reranked_precision = (
        sum(reranked_precisions)
        / len(reranked_precisions)
    )

    reranked_recall = (
        sum(reranked_recalls)
        / len(reranked_recalls)
    )

    reranked_map = mean_average_precision(
        reranked_qa_results,
        k=RERANKER_TOP_N,
        min_overlap_ratio=0.0,
    )

    reranked_sensitivity = map_at_k_sensitivity(
        reranked_qa_results,
        k=RERANKER_TOP_N,
    )


    # =========================================================================
    # CALCULATE IMPROVEMENT
    # =========================================================================

    precision_delta = (
        reranked_precision
        - baseline_precision
    )

    recall_delta = (
        reranked_recall
        - baseline_recall
    )

    map_delta = (
        reranked_map
        - baseline_map
    )


    # =========================================================================
    # PRINT FINAL RESULTS
    # =========================================================================

    print(
        "\n\n"
        + "=" * 90
    )

    print(
        "RERANKER EVALUATION RESULTS"
    )

    print(
        "=" * 90
    )


    print(
        "\nEXPERIMENT CONFIGURATION"
    )

    print(
        "-" * 90
    )

    print(
        f"Queries evaluated:       {len(tests)}"
    )

    print(
        f"Chunk size:              2000"
    )

    print(
        f"Chunk overlap:           200"
    )

    print(
        f"BM25 weight:             {BM25_WEIGHT:.2f}"
    )

    print(
        f"FAISS weight:            {DENSE_WEIGHT:.2f}"
    )

    print(
        f"Hybrid candidate pool:   {CANDIDATE_K}"
    )

    print(
        f"Reranker:                {RERANKER_MODEL}"
    )

    print(
        f"Reranker output:         {RERANKER_TOP_N}"
    )


    # =========================================================================
    # BASELINE
    # =========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "BASELINE: HYBRID RETRIEVAL → TOP 3"
    )

    print(
        "=" * 90
    )

    print(
        f"Precision@3:       {baseline_precision:.4f}"
    )

    print(
        f"Recall@3:          {baseline_recall:.4f}"
    )

    print(
        f"MAP@3:             {baseline_map:.4f}"
    )

    print(
    f"MAP sensitivity:   {baseline_sensitivity}"
    )

    print(
        f"Skipped chunks:    {baseline_skipped}"
    )


    # =========================================================================
    # RERANKED
    # =========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RERANKED: HYBRID TOP 10 → BGE RERANKER → TOP 3"
    )

    print(
        "=" * 90
    )

    print(
        f"Precision@3:       {reranked_precision:.4f}"
    )

    print(
        f"Recall@3:          {reranked_recall:.4f}"
    )

    print(
        f"MAP@3:             {reranked_map:.4f}"
    )

    print(
    f"MAP sensitivity:   {reranked_sensitivity}"
    )

    print(
        f"Skipped chunks:    {reranked_skipped}"
    )


    # =========================================================================
    # IMPROVEMENT
    # =========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RERANKER IMPACT"
    )

    print(
        "=" * 90
    )

    print(
        f"Precision@3 change: {precision_delta:+.4f}"
    )

    print(
        f"Recall@3 change:    {recall_delta:+.4f}"
    )

    print(
        f"MAP@3 change:       {map_delta:+.4f}"
    )


    # =========================================================================
    # PERCENTAGE IMPROVEMENT
    # =========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RELATIVE IMPROVEMENT"
    )

    print(
        "=" * 90
    )

    if baseline_precision != 0:

        precision_pct = (
            precision_delta
            / baseline_precision
            * 100
        )

        print(
            f"Precision@3: {precision_pct:+.2f}%"
        )

    else:

        print(
            "Precision@3: N/A"
        )


    if baseline_recall != 0:

        recall_pct = (
            recall_delta
            / baseline_recall
            * 100
        )

        print(
            f"Recall@3:    {recall_pct:+.2f}%"
        )

    else:

        print(
            "Recall@3:    N/A"
        )


    if baseline_map != 0:

        map_pct = (
            map_delta
            / baseline_map
            * 100
        )

        print(
            f"MAP@3:       {map_pct:+.2f}%"
        )

    else:

        print(
            "MAP@3:       N/A"
        )


    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Pipeline':<48}"
        f"{'P@3':>10}"
        f"{'Recall@3':>12}"
        f"{'MAP@3':>10}"
    )

    print(
        "-" * 90
    )

    print(
        f"{'Hybrid Top-3':<48}"
        f"{baseline_precision:>10.4f}"
        f"{baseline_recall:>12.4f}"
        f"{baseline_map:>10.4f}"
    )

    print(
        f"{'Hybrid Top-10 + BGE Reranker Top-3':<48}"
        f"{reranked_precision:>10.4f}"
        f"{reranked_recall:>12.4f}"
        f"{reranked_map:>10.4f}"
    )

    print(
        "-" * 90
    )

    print(
        f"{'Change':<48}"
        f"{precision_delta:>+10.4f}"
        f"{recall_delta:>+12.4f}"
        f"{map_delta:>+10.4f}"
    )


    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 90
    )

    if map_delta > 0:

        print(
            "✓ The reranker improved MAP@3 compared with the Hybrid Top-3 "
            "baseline."
        )

    elif map_delta < 0:

        print(
            "✗ The reranker decreased MAP@3 compared with the Hybrid Top-3 "
            "baseline."
        )

    else:

        print(
            "→ The reranker produced no change in MAP@3."
        )


    if recall_delta > 0:

        print(
            "✓ Recall@3 improved after reranking."
        )

    elif recall_delta < 0:

        print(
            "✗ Recall@3 decreased after reranking."
        )

    else:

        print(
            "→ Recall@3 remained unchanged."
        )


    print(
        "\nReranker experiment completed."
    )

    print(
        "=" * 90
        + "\n"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
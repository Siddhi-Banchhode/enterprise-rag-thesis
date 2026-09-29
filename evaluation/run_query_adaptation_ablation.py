"""
evaluation/run_query_adaptation_ablation.py
============================================

Query Adaptation / Expansion Ablation
--------------------------------------

Evaluates whether query adaptation helps the existing retrieval pipeline
handle terminology mismatch in LegalBench-RAG CUAD.

EXPERIMENTAL PRINCIPLE
----------------------

Only the QUERY changes.

The underlying retrieval pipeline remains fixed:

    - CUAD benchmark
    - First 50 queries
    - Referenced CUAD contract files only
    - Raw corpus text (no cleaning)
    - YOUR existing chunk_documents()
    - Chunk size = 2000
    - Chunk overlap = 200
    - all-mpnet-base-v2
    - Fresh FAISS index
    - FAISS-only retrieval
    - candidate_k = 10
    - top_k = 10
    - YOUR existing build_hybrid_retriever()
    - YOUR existing CUAD evaluator

Three configurations are evaluated:

    1. ORIGINAL QUERY

       Original query
           -> MPNet
           -> FAISS
           -> Top 10

    2. SINGLE QUERY ADAPTATION

       Original query
           -> Gemini terminology-aware reformulation
           -> MPNet
           -> FAISS
           -> Top 10

    3. MULTI-QUERY EXPANSION

       Original query
           +
       3 Gemini-generated retrieval queries
           -> MPNet
           -> FAISS for each query
           -> Reciprocal Rank Fusion
           -> Top 10

IMPORTANT EVALUATION RULE
-------------------------

The LLM receives ONLY the original CUAD question.

Gold snippets / answers are NEVER supplied to the LLM.

This prevents benchmark leakage.

GEMINI CONFIGURATION
--------------------

This script uses Google's Gemini API through its OpenAI-compatible
endpoint.

Set:

    export GEMINI_API_KEY="your-google-ai-studio-key"

Optional:

    export QUERY_ADAPTER_MODEL="gemini-2.5-flash"

The script:

    - spaces Gemini requests approximately 14 seconds apart
    - retries temporary 429 rate-limit responses
    - immediately stops on daily quota exhaustion
    - saves the query cache after every successful query
    - resumes from the existing cache on subsequent runs

IMPORTANT:
----------

Do NOT delete:

    evaluation/query_adaptation_cache.json

Keep:

    FORCE_REGENERATE_QUERIES = False

This allows the experiment to resume without regenerating queries that
were already successfully generated.

Run:

    python3 evaluation/run_query_adaptation_ablation.py
"""

# ===========================================================================
# IMPORTS
# ===========================================================================

import json
import os
import re
import sys
import time

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# ===========================================================================
# MAKE PROJECT ROOT IMPORTABLE
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


# ===========================================================================
# YOUR EXISTING PROJECT COMPONENTS
# ===========================================================================

from config import config

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.preprocessing.chunker import chunk_documents
from src.retrieval.hybrid_retriever import build_hybrid_retriever

from legalbench_rag_cuad_evaluator import (
    load_cuad_benchmark,
    referenced_file_paths,
    load_corpus_texts,
    RetrievedChunk,
    official_precision_recall_at_k,
    mean_average_precision,
    map_at_k_sensitivity,
)


# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================

# ---------------------------------------------------------------------------
# CUAD paths
# ---------------------------------------------------------------------------

CUAD_CORPUS_DIR = (
    "/Users/siddhibanchhode/Documents/enterprise-rag/"
    "legalbench_data/corpus"
)

CUAD_BENCHMARK_JSON = (
    "/Users/siddhibanchhode/Documents/enterprise-rag/"
    "legalbench_data/benchmarks/cuad.json"
)


# ---------------------------------------------------------------------------
# Benchmark size
# ---------------------------------------------------------------------------

# Screening experiment only.
#
# Full benchmark:
#     4042 queries
#
# Screening:
#     first 50 queries

LIMIT_QUERIES = 50


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

TOP_K = 10
CANDIDATE_K = 10


# IMPORTANT:
#
# This experiment deliberately uses FAISS-only retrieval.
#
# BM25 was tested separately in the retrieval ablation and did not improve
# MAP@10 on the evaluated CUAD screening subset.
#
# Therefore:
#
#     BM25 = 0
#     FAISS = 1
#
# ONLY QUERY FORMULATION changes in this experiment.
#
BM25_WEIGHT = 0.0
FAISS_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = config.EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Gemini query adaptation
# ---------------------------------------------------------------------------

# Gemini model used ONLY for query reformulation.
#
# Can be overridden from terminal:
#
#     export QUERY_ADAPTER_MODEL="gemini-2.5-flash"
#
QUERY_ADAPTER_MODEL = os.getenv(
    "QUERY_ADAPTER_MODEL",
    "gemini-2.5-flash",
)


# Number of alternative queries generated for each question.
NUM_EXPANDED_QUERIES = 3


# ---------------------------------------------------------------------------
# Gemini rate-limit protection
# ---------------------------------------------------------------------------

# Minimum interval between successful/request attempts.
#
# This reduces the likelihood of short-term rate limiting.
#
# IMPORTANT:
# This does NOT solve a daily quota limit.
# Daily quota exhaustion is handled separately below.
#
GEMINI_REQUEST_INTERVAL_SECONDS = 14


# Maximum retries for TEMPORARY rate-limit responses.
#
# Daily quota errors do NOT use these retries.
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

# Standard Reciprocal Rank Fusion constant.
#
# Used only for multi-query expansion.
RRF_K = 60


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------

QUERY_CACHE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "query_adaptation_cache.json"
)


# ---------------------------------------------------------------------------
# Whether to regenerate cached queries
# ---------------------------------------------------------------------------

# IMPORTANT:
#
# Keep this FALSE.
#
# This allows the experiment to resume after interruption or quota
# exhaustion.
#
FORCE_REGENERATE_QUERIES = False


# ===========================================================================
# GEMINI REQUEST TIMING
# ===========================================================================

_last_gemini_request_time = None


def wait_before_gemini_request():
    """
    Enforce a minimum delay between Gemini API requests.
    """

    global _last_gemini_request_time

    if _last_gemini_request_time is None:
        return

    elapsed = (
        time.perf_counter()
        - _last_gemini_request_time
    )

    remaining = (
        GEMINI_REQUEST_INTERVAL_SECONDS
        - elapsed
    )

    if remaining > 0:

        print(
            f"    Waiting {remaining:.1f}s "
            f"before next Gemini request..."
        )

        time.sleep(remaining)


# ===========================================================================
# OFFSET RECOVERY
# ===========================================================================

def find_offset(
    full_text: str,
    chunk_text: str,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Locate a retrieved chunk in its original raw CUAD document.

    First attempts exact matching.

    Then uses whitespace-normalized 40-character anchor matching.

    Returns:

        (start, end)

    or:

        (None, None)
    """

    # -----------------------------------------------------------------------
    # Exact match
    # -----------------------------------------------------------------------

    idx = full_text.find(
        chunk_text
    )

    if idx != -1:

        return (
            idx,
            idx + len(chunk_text),
        )


    # -----------------------------------------------------------------------
    # Whitespace-normalized fallback
    # -----------------------------------------------------------------------

    anchor = re.sub(
        r"\s+",
        " ",
        chunk_text,
    ).strip()[:40]

    if anchor:

        idx = full_text.find(
            anchor
        )

        if idx != -1:

            return (
                idx,
                idx + len(chunk_text),
            )


    return None, None


# ===========================================================================
# GEMINI CLIENT
# ===========================================================================

def get_gemini_client():
    """
    Create the Gemini client using Google's OpenAI-compatible API.

    Required environment variable:

        GEMINI_API_KEY

    Endpoint:

        https://generativelanguage.googleapis.com/v1beta/openai/
    """

    try:

        from openai import OpenAI

    except ImportError:

        raise RuntimeError(
            "\nThe 'openai' package is not installed.\n\n"
            "Install it with:\n"
            "    pip install openai\n"
        )


    api_key = os.getenv(
        "GEMINI_API_KEY"
    )


    if not api_key:

        raise RuntimeError(
            "\nGEMINI_API_KEY is not set.\n\n"
            "Set your Google AI Studio API key before running:\n\n"
            '    export GEMINI_API_KEY="your-google-ai-studio-key"\n'
        )


    return OpenAI(
        api_key=api_key,
        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        ),
    )


# ===========================================================================
# RATE LIMIT CLASSIFICATION
# ===========================================================================

def is_daily_quota_error(
    exc: Exception,
) -> bool:
    """
    Determine whether a Gemini 429 error indicates exhaustion of a
    daily/project/model quota rather than a temporary rate limit.

    The Gemini error observed in this experiment contains:

        GenerateRequestsPerDayPerProjectPerModel-FreeTier

    and:

        generate_content_free_tier_requests

    Daily quota errors should NEVER be retried immediately.
    """

    error_text = str(exc).lower()


    daily_quota_indicators = (
        "perday",
        "per_day",
        "generate_content_free_tier_requests",
        "generaterequestsperdayperprojectpermodel",
        "generate_requests_per_day",
        "daily quota",
        "quota exceeded",
    )


    return any(
        indicator in error_text
        for indicator in daily_quota_indicators
    )


# ===========================================================================
# LLM QUERY ADAPTER
# ===========================================================================

def generate_query_adaptations(
    query: str,
    num_queries: int = 3,
) -> List[str]:
    """
    Generate terminology-aware legal retrieval queries.

    IMPORTANT:
    The model receives ONLY the original query.

    It does NOT receive:

        - gold snippets
        - gold answers
        - retrieved documents
        - benchmark labels

    Therefore this does not intentionally leak ground truth.

    Rate-limit handling:

        Temporary 429
            -> retry with backoff

        Daily quota exhausted
            -> stop immediately

    Returns:

        List[str]
    """

    client = get_gemini_client()


    # -----------------------------------------------------------------------
    # Prompt
    # -----------------------------------------------------------------------

    prompt = f"""
You are a legal information retrieval query reformulation system.

Your task is to reformulate a user's question so that it can retrieve
relevant passages from legal contracts.

The original question may use terminology that differs from the wording
used inside the contract.

Generate exactly {num_queries} alternative retrieval queries.

Rules:

1. Preserve the original question's intent exactly.

2. Do NOT answer the question.

3. Do NOT introduce facts that are not implied by the question.

4. Use legally appropriate terminology where relevant.

5. Add valid contractual synonyms or closely related terminology.

6. Prefer terminology likely to occur in contracts.

7. Do not invent names, dates, companies, jurisdictions, amounts,
   or other contract-specific facts.

8. Each query should remain focused on the original information need.

9. Do not make the queries unnecessarily long.

10. The original question is the only information available to you.

11. Do not assume access to the contract or its answer.

Original question:

{query}

Return exactly {num_queries} alternative queries.

Return one query per line and nothing else.
""".strip()


    # -----------------------------------------------------------------------
    # Gemini API request
    # -----------------------------------------------------------------------

    global _last_gemini_request_time


    for attempt in range(
        MAX_RETRIES
    ):

        try:

            # ---------------------------------------------------------------
            # Proactive rate-limit protection
            # ---------------------------------------------------------------

            wait_before_gemini_request()


            print(
                f"    Sending Gemini request "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )


            _last_gemini_request_time = (
                time.perf_counter()
            )


            response = client.chat.completions.create(
                model=QUERY_ADAPTER_MODEL,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise legal information "
                            "retrieval query reformulation system."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )


            # ---------------------------------------------------------------
            # Successful API request
            # ---------------------------------------------------------------

            content = (
                response.choices[0].message.content
                or ""
            )


            lines = []


            for line in content.splitlines():

                line = line.strip()


                if not line:

                    continue


                # Remove common numbering:
                #
                # 1. ...
                # 2) ...
                # - ...
                # * ...

                line = re.sub(
                    r"^\s*(?:\d+[\.\)]|\-|\*)\s*",
                    "",
                    line,
                )


                if line:

                    lines.append(
                        line
                    )


            # ---------------------------------------------------------------
            # Remove duplicates while preserving order
            # ---------------------------------------------------------------

            unique = []

            seen = set()


            for line in lines:

                normalized = (
                    line.lower().strip()
                )


                if normalized not in seen:

                    seen.add(
                        normalized
                    )

                    unique.append(
                        line
                    )


            # ---------------------------------------------------------------
            # Validate generated query count
            # ---------------------------------------------------------------

            if len(unique) < num_queries:

                raise RuntimeError(
                    f"Query adapter returned only "
                    f"{len(unique)} unique queries "
                    f"instead of {num_queries}.\n"
                    f"Original query: {query}\n"
                    f"Raw response:\n{content}"
                )


            return unique[
                :num_queries
            ]


        except Exception as exc:

            # ---------------------------------------------------------------
            # Import RateLimitError
            # ---------------------------------------------------------------

            try:

                from openai import RateLimitError

            except ImportError:

                raise


            # ---------------------------------------------------------------
            # Non-429 error
            # ---------------------------------------------------------------

            if not isinstance(
                exc,
                RateLimitError,
            ):

                raise


            # ===============================================================
            # DAILY QUOTA EXHAUSTION
            # ===============================================================

            if is_daily_quota_error(
                exc
            ):

                print(
                    "\n"
                    + "=" * 80
                )

                print(
                    "GEMINI DAILY QUOTA EXHAUSTED"
                )

                print(
                    "=" * 80
                )

                print(
                    "The Gemini free-tier daily request quota "
                    "for the selected model/project has been exhausted."
                )

                print(
                    f"Model: {QUERY_ADAPTER_MODEL}"
                )

                print(
                    "\nNo further Gemini requests will be attempted."
                )

                print(
                    "\nAlready generated query adaptations remain "
                    "saved in:"
                )

                print(
                    f"  {QUERY_CACHE_PATH}"
                )

                print(
                    "\nThe experiment can be resumed later."
                )

                print(
                    "Keep:"
                )

                print(
                    "  FORCE_REGENERATE_QUERIES = False"
                )

                print(
                    "\nRun the same command again when quota is available:"
                )

                print(
                    "  python3 evaluation/"
                    "run_query_adaptation_ablation.py"
                )

                print(
                    "=" * 80
                )


                # -----------------------------------------------------------
                # IMPORTANT:
                #
                # Exit cleanly.
                #
                # Do NOT retry a daily quota error.
                # Do NOT delete the cache.
                # -----------------------------------------------------------

                raise SystemExit(0)


            # ===============================================================
            # TEMPORARY RATE LIMIT
            # ===============================================================

            if attempt == MAX_RETRIES - 1:

                print(
                    "\nGemini temporary rate limit persisted "
                    f"after {MAX_RETRIES} attempts."
                )

                raise


            # ---------------------------------------------------------------
            # Progressive backoff
            # ---------------------------------------------------------------

            wait_time = min(
                60,
                35 * (attempt + 1),
            )


            print(
                "\n    Gemini temporary rate limit reached."
            )


            print(
                f"    Waiting {wait_time}s before retry "
                f"({attempt + 1}/{MAX_RETRIES})..."
            )


            time.sleep(
                wait_time
            )


    raise RuntimeError(
        "Gemini query adaptation generation failed."
    )


# ===========================================================================
# QUERY CACHE
# ===========================================================================

def load_query_cache() -> Dict[str, Any]:
    """
    Load previously generated query adaptations.

    This prevents unnecessary Gemini API calls and allows the experiment
    to resume after interruption.
    """

    if not QUERY_CACHE_PATH.exists():

        return {}


    try:

        with open(
            QUERY_CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            cache = json.load(
                f
            )


        if not isinstance(
            cache,
            dict,
        ):

            print(
                "WARNING: Query cache is not a dictionary."
            )

            return {}


        return cache


    except Exception as exc:

        print(
            f"WARNING: Could not read query cache: {exc}"
        )

        return {}


# ---------------------------------------------------------------------------
# Save cache
# ---------------------------------------------------------------------------

def save_query_cache(
    cache: Dict[str, Any],
):
    """
    Save generated query adaptations.

    The cache is written after every successful query.

    Uses an atomic temporary-file replacement so that an interruption
    during writing does not normally corrupt the existing cache.
    """

    QUERY_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temp_path = QUERY_CACHE_PATH.with_suffix(
        ".tmp"
    )


    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False,
        )


    temp_path.replace(
        QUERY_CACHE_PATH
    )


# ===========================================================================
# PREPARE QUERY ADAPTATIONS
# ===========================================================================

def prepare_query_adaptations(
    tests,
) -> Dict[str, List[str]]:
    """
    Generate or load 3 alternative queries for every benchmark question.

    Cache key is the exact original query.

    Returned structure:

        {
            original_query: [
                adapted_query_1,
                adapted_query_2,
                adapted_query_3
            ]
        }
    """

    cache = load_query_cache()

    adaptations = {}

    missing = []


    # -----------------------------------------------------------------------
    # Determine which queries need generation
    # -----------------------------------------------------------------------

    for qa_gt in tests:

        query = qa_gt.query


        if (
            not FORCE_REGENERATE_QUERIES
            and query in cache
            and "expanded_queries" in cache[query]
            and isinstance(
                cache[query]["expanded_queries"],
                list,
            )
            and len(
                cache[query]["expanded_queries"]
            ) >= NUM_EXPANDED_QUERIES
        ):

            adaptations[query] = (
                cache[query]["expanded_queries"]
            )

        else:

            missing.append(
                query
            )


    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "QUERY ADAPTATION PREPARATION"
    )

    print(
        "=" * 80
    )


    print(
        f"Queries requiring LLM generation: "
        f"{len(missing)}/{len(tests)}"
    )


    print(
        f"Queries already available in cache: "
        f"{len(tests) - len(missing)}/{len(tests)}"
    )


    # -----------------------------------------------------------------------
    # No missing queries
    # -----------------------------------------------------------------------

    if not missing:

        print(
            "\nAll query adaptations loaded from cache."
        )

        print(
            f"Cache: {QUERY_CACHE_PATH}"
        )

        return adaptations


    # -----------------------------------------------------------------------
    # Missing queries exist
    # -----------------------------------------------------------------------

    print(
        f"\nGenerating {NUM_EXPANDED_QUERIES} adapted queries "
        f"for each missing query..."
    )


    print(
        f"Gemini model: {QUERY_ADAPTER_MODEL}"
    )


    print(
        f"Minimum request interval: "
        f"{GEMINI_REQUEST_INTERVAL_SECONDS}s"
    )


    print(
        f"Maximum temporary retries: "
        f"{MAX_RETRIES}"
    )


    print(
        f"Cache path: "
        f"{QUERY_CACHE_PATH}"
    )


    start_time = time.perf_counter()


    # -----------------------------------------------------------------------
    # Generate missing queries
    # -----------------------------------------------------------------------

    for i, query in enumerate(
        missing,
        1,
    ):

        print(
            "\n" + "-" * 70
        )


        print(
            f"Generating query adaptations "
            f"{i}/{len(missing)}"
        )


        print(
            f"Original query: {query}"
        )


        # ---------------------------------------------------------------
        # Gemini generation
        #
        # If the daily quota is exhausted, generate_query_adaptations()
        # exits cleanly.
        # ---------------------------------------------------------------

        expanded = generate_query_adaptations(
            query,
            num_queries=NUM_EXPANDED_QUERIES,
        )


        # ---------------------------------------------------------------
        # Display generated queries
        # ---------------------------------------------------------------

        print(
            "Generated adaptations:"
        )


        for j, adapted in enumerate(
            expanded,
            1,
        ):

            print(
                f"  {j}. {adapted}"
            )


        # ---------------------------------------------------------------
        # Store in memory
        # ---------------------------------------------------------------

        adaptations[query] = expanded


        # ---------------------------------------------------------------
        # Store in cache
        # ---------------------------------------------------------------

        cache[query] = {
            "original_query": query,
            "expanded_queries": expanded,
            "model": QUERY_ADAPTER_MODEL,
        }


        # ---------------------------------------------------------------
        # CRITICAL:
        #
        # Save after EVERY successful query.
        #
        # If the experiment stops after query 17, the next run will
        # start from query 18 instead of regenerating 1-17.
        # ---------------------------------------------------------------

        save_query_cache(
            cache
        )


        print(
            "    ✓ Saved to query cache."
        )


    elapsed = (
        time.perf_counter()
        - start_time
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "Query adaptation generation completed."
    )

    print(
        f"Generation time: {elapsed:.2f}s"
    )

    print(
        f"Cache: {QUERY_CACHE_PATH}"
    )


    return adaptations


# ===========================================================================
# DOCUMENT / OFFSET CONVERSION
# ===========================================================================

def convert_documents_to_retrieved_chunks(
    documents: List[Document],
    corpus_texts: Dict[str, str],
) -> Tuple[List[RetrievedChunk], int]:
    """
    Convert LangChain Documents into RetrievedChunk objects required by
    the official CUAD evaluator.

    Returns:

        retrieved_chunks
        skipped_count
    """

    retrieved = []

    skipped = 0


    for doc in documents:

        file_path = doc.metadata.get(
            "source",
            "",
        )


        full_text = corpus_texts.get(
            file_path,
            "",
        )


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


# ===========================================================================
# EVALUATION
# ===========================================================================

def evaluate_retrieval_results(
    tests,
    retrieved_by_query: Dict[str, List[RetrievedChunk]],
    label: str,
) -> Dict[str, Any]:
    """
    Run the exact metrics from legalbench_rag_cuad_evaluator.py.
    """

    print(
        "\n" + "-" * 70
    )

    print(
        f"RESULT: {label}"
    )

    print(
        "-" * 70
    )


    precisions = []

    recalls = []

    qa_results = []

    skipped = 0


    for qa_gt in tests:

        retrieved = retrieved_by_query.get(
            qa_gt.query,
            [],
        )


        # Keep exactly Top-K for official scoring.
        retrieved = retrieved[:TOP_K]


        p, r = official_precision_recall_at_k(
            qa_gt.snippets,
            retrieved,
            k=TOP_K,
        )


        precisions.append(
            p
        )

        recalls.append(
            r
        )


        qa_results.append(
            (
                qa_gt,
                retrieved,
            )
        )


    if not precisions:

        raise RuntimeError(
            f"No results available for evaluation: {label}"
        )


    avg_precision = (
        sum(precisions)
        / len(precisions)
    )


    avg_recall = (
        sum(recalls)
        / len(recalls)
    )


    # MAP@10 using the existing official evaluation helper.
    map10 = mean_average_precision(
        qa_results,
        k=TOP_K,
        min_overlap_ratio=0.0,
    )


    # Sensitivity analysis.
    sensitivity = map_at_k_sensitivity(
        qa_results,
        k=TOP_K,
    )


    print(
        f"Precision@{TOP_K}: {avg_precision:.3f}"
    )


    print(
        f"Recall@{TOP_K}:    {avg_recall:.3f}"
    )


    print(
        f"MAP@{TOP_K}:       {map10:.3f}"
    )


    print(
        f"MAP sensitivity:   {sensitivity}"
    )


    return {
        "label": label,
        "precision": avg_precision,
        "recall": avg_recall,
        "map": map10,
        "sensitivity": sensitivity,
        "skipped": skipped,
        "qa_results": qa_results,
    }


# ===========================================================================
# RRF
# ===========================================================================

def document_identity(
    doc: Document,
) -> Tuple[str, str]:
    """
    Identity used for multi-query result fusion.

    We use:

        source + exact page content

    The same contract chunk retrieved by different query formulations
    therefore receives a combined RRF score.
    """

    source = doc.metadata.get(
        "source",
        "",
    )


    return (
        source,
        doc.page_content,
    )


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    rrf_k: int = 60,
) -> List[Document]:
    """
    Fuse multiple ranked retrieval lists using Reciprocal Rank Fusion.

    score(d) = sum 1 / (rrf_k + rank)

    Rank is 1-based.
    """

    scores = {}

    documents = {}


    for ranked_docs in ranked_lists:

        for rank, doc in enumerate(
            ranked_docs,
            start=1,
        ):

            identity = document_identity(
                doc
            )


            scores[identity] = (
                scores.get(
                    identity,
                    0.0,
                )
                + 1.0 / (
                    rrf_k + rank
                )
            )


            documents[identity] = doc


    ranked = sorted(
        documents.items(),
        key=lambda item: scores[item[0]],
        reverse=True,
    )


    return [
        documents[identity]
        for identity, _score in ranked
    ]


# ===========================================================================
# RETRIEVER
# ===========================================================================

def build_faiss_retriever(
    chunks,
):
    """
    Build the same FAISS-only retriever used in the embedding ablation.

    The existing build_hybrid_retriever() is still used, but with:

        BM25 = 0
        FAISS = 1
    """

    # -----------------------------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------------------------

    print(
        "\nLoading embedding model:"
    )

    print(
        f"  {EMBEDDING_MODEL}"
    )


    start = time.perf_counter()


    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


    load_time = (
        time.perf_counter()
        - start
    )


    print(
        f"Embedding model loaded in "
        f"{load_time:.2f} seconds."
    )


    # -----------------------------------------------------------------------
    # Build FAISS index
    # -----------------------------------------------------------------------

    print(
        f"\nBuilding fresh FAISS index over "
        f"{len(chunks)} chunks..."
    )


    start = time.perf_counter()


    vectorstore = FAISS.from_documents(
        chunks,
        embedder,
    )


    index_time = (
        time.perf_counter()
        - start
    )


    print(
        f"FAISS index built in "
        f"{index_time:.2f} seconds."
    )


    print(
        f"FAISS vectors: "
        f"{vectorstore.index.ntotal}"
    )


    # -----------------------------------------------------------------------
    # Existing retriever
    # -----------------------------------------------------------------------

    print(
        "\nBuilding YOUR existing retriever..."
    )


    hybrid = build_hybrid_retriever(
        chunks,
        vectorstore,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        bm25_weight=BM25_WEIGHT,
        dense_weight=FAISS_WEIGHT,
    )


    print(
        "Retriever ready."
    )


    return (
        hybrid,
        load_time,
        index_time,
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():

    overall_start = time.perf_counter()


    # =======================================================================
    # HEADER
    # =======================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "QUERY ADAPTATION / EXPANSION ABLATION"
    )

    print(
        "LegalBench-RAG CUAD"
    )

    print(
        "=" * 80
    )


    print(
        """
Experimental principle:
Only query formulation changes.

Fixed:
  - CUAD benchmark
  - 50 queries
  - 5 referenced contract files
  - raw corpus
  - existing chunk_documents()
  - chunk size = 2000
  - chunk overlap = 200
  - all-mpnet-base-v2
  - FAISS-only
  - candidate_k = 10
  - top_k = 10
  - existing build_hybrid_retriever()
  - official CUAD evaluator
"""
    )


    # =======================================================================
    # CONFIGURATION DISPLAY
    # =======================================================================

    print(
        "=" * 80
    )

    print(
        "QUERY ADAPTER CONFIGURATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Query adapter model:       {QUERY_ADAPTER_MODEL}"
    )

    print(
        "API provider:              Google Gemini"
    )

    print(
        "API mode:                  OpenAI-compatible"
    )

    print(
        f"Expanded queries/question: {NUM_EXPANDED_QUERIES}"
    )

    print(
        f"Minimum request interval:  "
        f"{GEMINI_REQUEST_INTERVAL_SECONDS}s"
    )

    print(
        f"Maximum temporary retries: "
        f"{MAX_RETRIES}"
    )

    print(
        f"Cache path:                {QUERY_CACHE_PATH}"
    )

    print(
        f"Force regeneration:        {FORCE_REGENERATE_QUERIES}"
    )


    # =======================================================================
    # LOAD BENCHMARK
    # =======================================================================

    print(
        "\nLoading CUAD benchmark..."
    )


    tests = load_cuad_benchmark(
        CUAD_BENCHMARK_JSON
    )


    print(
        f"Full benchmark contains "
        f"{len(tests)} queries."
    )


    if LIMIT_QUERIES:

        tests = tests[
            :LIMIT_QUERIES
        ]


    file_paths = referenced_file_paths(
        tests
    )


    print(
        f"Using first {len(tests)} queries "
        f"for this screening experiment."
    )


    print(
        f"{len(tests)} queries over "
        f"{len(file_paths)} contract files."
    )


    # =======================================================================
    # LOAD RAW CORPUS
    # =======================================================================

    print(
        "\nLoading raw CUAD corpus..."
    )


    corpus_texts = load_corpus_texts(
        CUAD_CORPUS_DIR,
        file_paths,
    )


    print(
        f"Loaded {len(corpus_texts)} contract files."
    )


    # =======================================================================
    # WRAP AS DOCUMENTS
    # =======================================================================

    print(
        "\nWrapping raw corpus text as Documents "
        "(source = exact benchmark file_path)..."
    )


    raw_docs = [
        Document(
            page_content=text,
            metadata={
                "source": fp,
            },
        )
        for fp, text in corpus_texts.items()
    ]


    # =======================================================================
    # CHUNK
    # =======================================================================

    print(
        "\nChunking with YOUR existing chunker..."
    )


    chunks = chunk_documents(
        raw_docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


    print(
        f"Created {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )


    # =======================================================================
    # BUILD RETRIEVER
    # =======================================================================

    (
        hybrid,
        load_time,
        index_time,
    ) = build_faiss_retriever(
        chunks
    )


    # =======================================================================
    # PREPARE QUERY ADAPTATIONS
    # =======================================================================

    adaptations = prepare_query_adaptations(
        tests
    )


    # =======================================================================
    # EXPERIMENT 1:
    # ORIGINAL QUERY
    # =======================================================================

    print(
        "\n\n" + "=" * 80
    )

    print(
        "EXPERIMENT 1: ORIGINAL QUERY BASELINE"
    )

    print(
        "=" * 80
    )


    print(
        f"FAISS weight:    {FAISS_WEIGHT}"
    )

    print(
        f"Candidate K:     {CANDIDATE_K}"
    )

    print(
        f"Top K:           {TOP_K}"
    )

    print(
        f"Queries:         {len(tests)}"
    )


    baseline_retrieved = {}

    baseline_skipped = 0


    start = time.perf_counter()


    for i, qa_gt in enumerate(
        tests,
        start=1,
    ):

        docs = hybrid.invoke(
            qa_gt.query
        )[:TOP_K]


        retrieved, skipped = (
            convert_documents_to_retrieved_chunks(
                docs,
                corpus_texts,
            )
        )


        baseline_retrieved[
            qa_gt.query
        ] = retrieved


        baseline_skipped += skipped


        if (
            i % 10 == 0
            or i == len(tests)
        ):

            print(
                f"  ...{i}/{len(tests)} queries done"
            )


    baseline_retrieval_time = (
        time.perf_counter()
        - start
    )


    baseline_result = (
        evaluate_retrieval_results(
            tests,
            baseline_retrieved,
            "Original Query",
        )
    )


    baseline_result["skipped"] = (
        baseline_skipped
    )


    baseline_result["retrieval_time"] = (
        baseline_retrieval_time
    )


    # =======================================================================
    # EXPERIMENT 2:
    # SINGLE QUERY ADAPTATION
    # =======================================================================

    print(
        "\n\n" + "=" * 80
    )

    print(
        "EXPERIMENT 2: SINGLE QUERY ADAPTATION"
    )

    print(
        "=" * 80
    )


    print(
        """
Original query
      ↓
Gemini terminology-aware reformulation
      ↓
MPNet
      ↓
FAISS
      ↓
Top 10
"""
    )


    single_retrieved = {}

    single_skipped = 0


    start = time.perf_counter()


    for i, qa_gt in enumerate(
        tests,
        start=1,
    ):

        expanded_queries = adaptations[
            qa_gt.query
        ]


        # First generated adaptation.
        adapted_query = (
            expanded_queries[0]
        )


        docs = hybrid.invoke(
            adapted_query
        )[:TOP_K]


        retrieved, skipped = (
            convert_documents_to_retrieved_chunks(
                docs,
                corpus_texts,
            )
        )


        single_retrieved[
            qa_gt.query
        ] = retrieved


        single_skipped += skipped


        if (
            i % 10 == 0
            or i == len(tests)
        ):

            print(
                f"  ...{i}/{len(tests)} queries done"
            )


    single_retrieval_time = (
        time.perf_counter()
        - start
    )


    single_result = (
        evaluate_retrieval_results(
            tests,
            single_retrieved,
            "Single Query Adaptation",
        )
    )


    single_result["skipped"] = (
        single_skipped
    )


    single_result["retrieval_time"] = (
        single_retrieval_time
    )


    # =======================================================================
    # EXPERIMENT 3:
    # MULTI-QUERY EXPANSION + RRF
    # =======================================================================

    print(
        "\n\n" + "=" * 80
    )

    print(
        "EXPERIMENT 3: MULTI-QUERY EXPANSION + RRF"
    )

    print(
        "=" * 80
    )


    print(
        """
Original query
      +
3 adapted queries
      ↓
4 independent FAISS retrievals
      ↓
Reciprocal Rank Fusion
      ↓
Top 10
"""
    )


    multi_retrieved = {}

    multi_skipped = 0


    start = time.perf_counter()


    for i, qa_gt in enumerate(
        tests,
        start=1,
    ):

        expanded_queries = adaptations[
            qa_gt.query
        ]


        # ---------------------------------------------------------------
        # Original + 3 adapted queries
        # ---------------------------------------------------------------

        all_queries = (
            [qa_gt.query]
            + expanded_queries[
                :NUM_EXPANDED_QUERIES
            ]
        )


        ranked_lists = []


        for retrieval_query in all_queries:

            docs = hybrid.invoke(
                retrieval_query
            )[:CANDIDATE_K]


            ranked_lists.append(
                docs
            )


        # ---------------------------------------------------------------
        # RRF
        # ---------------------------------------------------------------

        fused_docs = reciprocal_rank_fusion(
            ranked_lists,
            rrf_k=RRF_K,
        )


        fused_docs = fused_docs[
            :TOP_K
        ]


        # ---------------------------------------------------------------
        # Convert to official evaluator format
        # ---------------------------------------------------------------

        retrieved, skipped = (
            convert_documents_to_retrieved_chunks(
                fused_docs,
                corpus_texts,
            )
        )


        multi_retrieved[
            qa_gt.query
        ] = retrieved


        multi_skipped += skipped


        if (
            i % 10 == 0
            or i == len(tests)
        ):

            print(
                f"  ...{i}/{len(tests)} queries done"
            )


    multi_retrieval_time = (
        time.perf_counter()
        - start
    )


    multi_result = (
        evaluate_retrieval_results(
            tests,
            multi_retrieved,
            "Multi-Query Expansion + RRF",
        )
    )


    multi_result["skipped"] = (
        multi_skipped
    )


    multi_result["retrieval_time"] = (
        multi_retrieval_time
    )


    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================

    print(
        "\n\n" + "=" * 90
    )

    print(
        "QUERY ADAPTATION ABLATION SUMMARY"
    )

    print(
        "=" * 90
    )


    print(
        f"{'Configuration':<32}"
        f"{'P@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MAP@10':>10}"
        f"{'Skipped':>10}"
        f"{'Retrieval':>12}"
    )


    print(
        "-" * 90
    )


    results = [
        baseline_result,
        single_result,
        multi_result,
    ]


    for result in results:

        print(
            f"{result['label']:<32}"
            f"{result['precision']:>10.3f}"
            f"{result['recall']:>12.3f}"
            f"{result['map']:>10.3f}"
            f"{result['skipped']:>10}"
            f"{result['retrieval_time']:>11.2f}s"
        )


    # =======================================================================
    # IMPROVEMENT ANALYSIS
    # =======================================================================

    baseline_map = (
        baseline_result["map"]
    )


    baseline_recall = (
        baseline_result["recall"]
    )


    baseline_precision = (
        baseline_result["precision"]
    )


    print(
        "\n" + "=" * 90
    )

    print(
        "IMPROVEMENT OVER ORIGINAL QUERY BASELINE"
    )

    print(
        "=" * 90
    )


    for result in [
        single_result,
        multi_result,
    ]:

        map_abs = (
            result["map"]
            - baseline_map
        )


        recall_abs = (
            result["recall"]
            - baseline_recall
        )


        precision_abs = (
            result["precision"]
            - baseline_precision
        )


        if baseline_map != 0:

            map_relative = (
                map_abs
                / baseline_map
            ) * 100

        else:

            map_relative = 0.0


        print(
            f"\n{result['label']}"
        )


        print(
            f"  MAP@10 absolute:     "
            f"{map_abs:+.3f}"
        )


        print(
            f"  MAP@10 relative:     "
            f"{map_relative:+.2f}%"
        )


        print(
            f"  Recall@10 change:    "
            f"{recall_abs:+.3f}"
        )


        print(
            f"  Precision@10 change: "
            f"{precision_abs:+.3f}"
        )


    # =======================================================================
    # BEST CONFIGURATION
    # =======================================================================

    best = max(
        results,
        key=lambda x: x["map"],
    )


    print(
        "\n" + "=" * 90
    )

    print(
        "BEST CONFIGURATION BY MAP@10"
    )

    print(
        "=" * 90
    )


    print(
        f"Configuration: {best['label']}"
    )


    print(
        f"MAP@10:       {best['map']:.3f}"
    )


    print(
        f"Recall@10:    {best['recall']:.3f}"
    )


    print(
        f"Precision@10: {best['precision']:.3f}"
    )


    # =======================================================================
    # RUNTIME
    # =======================================================================

    overall_time = (
        time.perf_counter()
        - overall_start
    )


    print(
        "\n" + "=" * 90
    )

    print(
        "RUNTIME SUMMARY"
    )

    print(
        "=" * 90
    )


    print(
        f"Embedding model load:       "
        f"{load_time:.2f}s"
    )


    print(
        f"FAISS index build:           "
        f"{index_time:.2f}s"
    )


    print(
        f"Baseline retrieval:          "
        f"{baseline_retrieval_time:.2f}s"
    )


    print(
        f"Single adaptation retrieval: "
        f"{single_retrieval_time:.2f}s"
    )


    print(
        f"Multi-query retrieval:       "
        f"{multi_retrieval_time:.2f}s"
    )


    print(
        f"Total experiment runtime:    "
        f"{overall_time:.2f}s"
    )


    # =======================================================================
    # CONFIGURATION
    # =======================================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "EXPERIMENT CONFIGURATION"
    )

    print(
        "=" * 90
    )


    print(
        f"Embedding model:             "
        f"{EMBEDDING_MODEL}"
    )


    print(
        f"Chunk size:                  "
        f"{CHUNK_SIZE}"
    )


    print(
        f"Chunk overlap:               "
        f"{CHUNK_OVERLAP}"
    )


    print(
        f"Candidate K:                 "
        f"{CANDIDATE_K}"
    )


    print(
        f"Top K:                       "
        f"{TOP_K}"
    )


    print(
        f"BM25 weight:                 "
        f"{BM25_WEIGHT}"
    )


    print(
        f"FAISS weight:                "
        f"{FAISS_WEIGHT}"
    )


    print(
        f"Query adapter provider:      "
        f"Google Gemini"
    )


    print(
        f"Query adapter model:         "
        f"{QUERY_ADAPTER_MODEL}"
    )


    print(
        f"Expanded queries:            "
        f"{NUM_EXPANDED_QUERIES}"
    )


    print(
        f"Gemini request interval:     "
        f"{GEMINI_REQUEST_INTERVAL_SECONDS}s"
    )


    print(
        f"Maximum temporary retries:   "
        f"{MAX_RETRIES}"
    )


    print(
        f"RRF K:                       "
        f"{RRF_K}"
    )


    print(
        f"Query cache:                 "
        f"{QUERY_CACHE_PATH}"
    )


    print(
        f"Force regenerate queries:    "
        f"{FORCE_REGENERATE_QUERIES}"
    )


    # =======================================================================
    # COMPLETION
    # =======================================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 90
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    main()
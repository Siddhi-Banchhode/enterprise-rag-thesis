"""
Evaluate a single BM25/FAISS weight config (default: BM25=0.0, FAISS=1.0)
on the FULL official LegalBench-RAG CUAD benchmark (462 contracts /
4,042 queries).

WHY THIS SCRIPT EXISTS
-----------------------
Your dev-split weight sweep found MAP@10 highest at BM25=0.0/FAISS=1.0
(0.337), well above your current hybrid default of 0.40/0.60 (0.262 on
that same dev split). But those numbers came from a smaller dev subset,
so they aren't directly comparable to your reported full-scale baseline
(MAP@10 = 10.2% no-rerank / 8.6% reranked, at BM25=0.40/FAISS=0.60, on
the full 462-contract / 4,042-query benchmark).

This script re-runs the winning config on that SAME full benchmark, so
you get one number that's actually comparable to your reported 0.102.

CRITICAL FOR VALID COMPARISON
-------------------------------
This script does NOT reimplement your P@10/Recall@10/MAP@10 metric
logic. It imports it from wherever you computed the 0.102/0.086
numbers. If you compute metrics differently here, the two numbers
are not comparable and the whole point of this run is lost. Fix the
three TODOs below to point at your actual code, then run it.

USAGE
-----
    python evaluate_faiss_only_full_benchmark.py
    python evaluate_faiss_only_full_benchmark.py --rerank
    python evaluate_faiss_only_full_benchmark.py --weights 0.0 1.0 --top-k 10
"""

import argparse
import json
import time
from pathlib import Path

# =============================================================================
# TODO #1 — point this at whatever builds your hybrid retriever
#           (the same function your weight-sweep script called with
#           different (bm25_weight, faiss_weight) pairs)
# -----------------------------------------------------------------------------
# from src.retrieval.hybrid import build_hybrid_retriever
#
# Expected signature, adjust to match yours:
#     build_hybrid_retriever(bm25_weight: float, faiss_weight: float, top_k: int) -> Retriever
# =============================================================================

# =============================================================================
# TODO #2 — point this at your reranker, only needed if you pass --rerank
# -----------------------------------------------------------------------------
# from src.retrieval.reranker import rerank_chunks
#
# Expected signature, adjust to match yours:
#     rerank_chunks(query: str, candidates: list, top_n: int = 10) -> list
# =============================================================================

# =============================================================================
# TODO #3 — MOST IMPORTANT: point this at the exact metric function that
#           produced your reported 10.2% / 8.6% MAP@10 numbers. Do not
#           let this fall back to the default implementation below if
#           you have your own — the whole point is identical methodology.
# -----------------------------------------------------------------------------
# from src.eval.legalbench_rag import compute_precision_recall_map
#
# Expected signature, adjust to match yours:
#     compute_precision_recall_map(retrieved_chunks: list, ground_truth_spans: list, k: int) -> dict
#     -> returns {"precision_at_k": float, "recall_at_k": float, "average_precision": float}
# =============================================================================


# ---------------------------------------------------------------------------
# CONFIG — adjust paths to match your repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path.home() / "Documents" / "enterprise-rag"

# The full official benchmark file you already used to get the 0.102 baseline
# — NOT the dev split you used for the weight sweep.
BENCHMARK_QUERIES_PATH = REPO_ROOT / "data" / "legalbench_rag" / "queries_full.json"

# Reuse the FAISS index you already built for the 462-contract corpus if you
# have one saved — rebuilding/re-embedding 462 contracts costs real time for
# no reason if the index already exists on disk.
FAISS_INDEX_PATH = REPO_ROOT / "data" / "legalbench_rag" / "faiss_index"

RESULTS_OUT_PATH = REPO_ROOT / "results" / "faiss_only_full_benchmark.json"
CHECKPOINT_PATH = REPO_ROOT / "results" / "faiss_only_full_benchmark.checkpoint.json"
CHECKPOINT_EVERY = 200  # queries; this is retrieval-only so it should be fast, but 4,042 queries is still worth checkpointing in case it's interrupted


def load_benchmark_queries(path: Path) -> list:
    """Load the full 4,042-query benchmark with ground-truth spans."""
    with open(path, "r") as f:
        data = json.load(f)
    # Adjust this if your queries file has a different structure
    return data["queries"] if isinstance(data, dict) and "queries" in data else data


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"completed_query_ids": [], "per_query_results": []}


def save_checkpoint(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)


def run_config(bm25_weight: float, faiss_weight: float, top_k: int, use_rerank: bool):
    print(f"\n{'='*60}")
    print(f"Config: BM25={bm25_weight} | FAISS={faiss_weight} | rerank={use_rerank}")
    print(f"{'='*60}")

    queries = load_benchmark_queries(BENCHMARK_QUERIES_PATH)
    print(f"Loaded {len(queries)} queries from {BENCHMARK_QUERIES_PATH}")

    # --- TODO #1 wiring ---
    # retriever = build_hybrid_retriever(bm25_weight, faiss_weight, top_k)
    raise NotImplementedError(
        "Wire up TODO #1 (retriever), TODO #2 (reranker, if --rerank), "
        "and TODO #3 (metric function) above before running this script."
    )

    state = load_checkpoint(CHECKPOINT_PATH)
    completed_ids = set(state["completed_query_ids"])
    per_query_results = state["per_query_results"]

    start = time.time()
    for i, q in enumerate(queries):
        qid = q.get("id", i)
        if qid in completed_ids:
            continue

        # --- retrieval ---
        # candidates = retriever.retrieve(q["question"])
        candidates = None  # placeholder

        # --- optional reranking ---
        if use_rerank:
            # candidates = rerank_chunks(q["question"], candidates, top_n=top_k)
            pass

        # --- TODO #3 wiring: use YOUR metric function, not a new one ---
        # metrics = compute_precision_recall_map(candidates, q["ground_truth_spans"], k=top_k)
        metrics = None  # placeholder

        per_query_results.append({"query_id": qid, **(metrics or {})})
        completed_ids.add(qid)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            state["completed_query_ids"] = list(completed_ids)
            state["per_query_results"] = per_query_results
            save_checkpoint(CHECKPOINT_PATH, state)
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(queries)}] checkpointed — {elapsed:.1f}s elapsed")

    # final save
    state["completed_query_ids"] = list(completed_ids)
    state["per_query_results"] = per_query_results
    save_checkpoint(CHECKPOINT_PATH, state)

    # --- aggregate ---
    n = len(per_query_results)
    avg_precision = sum(r.get("precision_at_k", 0) for r in per_query_results) / n
    avg_recall = sum(r.get("recall_at_k", 0) for r in per_query_results) / n
    map_at_k = sum(r.get("average_precision", 0) for r in per_query_results) / n

    summary = {
        "bm25_weight": bm25_weight,
        "faiss_weight": faiss_weight,
        "rerank": use_rerank,
        "top_k": top_k,
        "n_queries": n,
        "precision_at_10": round(avg_precision, 4),
        "recall_at_10": round(avg_recall, 4),
        "map_at_10": round(map_at_k, 4),
    }

    RESULTS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def print_comparison_table(new_result: dict):
    """Print in the same format as your existing thesis table for easy copy-in."""
    print("\n" + "=" * 80)
    print("COMPARISON — full 462-contract / 4,042-query benchmark")
    print("=" * 80)
    print(f"{'Config':<28} {'P@10':>8} {'Recall@10':>12} {'MAP@10':>10}")
    print("-" * 80)
    print(f"{'Hybrid 0.40/0.60 (baseline)':<28} {'—':>8} {'—':>12} {'10.2%':>10}")
    print(f"{'Hybrid 0.40/0.60 + rerank':<28} {'—':>8} {'—':>12} {'8.6%':>10}")
    label = f"FAISS-only{' + rerank' if new_result['rerank'] else ''}"
    print(
        f"{label:<28} "
        f"{new_result['precision_at_10']*100:>7.1f}% "
        f"{new_result['recall_at_10']*100:>11.1f}% "
        f"{new_result['map_at_10']*100:>9.1f}%"
    )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs=2, type=float, default=[0.0, 1.0],
                         metavar=("BM25", "FAISS"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true",
                         help="Also apply your reranker on top of the FAISS-only results")
    args = parser.parse_args()

    result = run_config(args.weights[0], args.weights[1], args.top_k, args.rerank)
    print_comparison_table(result)
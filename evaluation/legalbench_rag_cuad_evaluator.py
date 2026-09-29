"""
legalbench_rag_cuad_evaluator.py

Evaluates YOUR retrieval pipeline against the official LegalBench-RAG CUAD
benchmark (the same ground truth RAGLite's published numbers come from),
using:

  1. OFFICIAL Precision@10 / Recall@10 — reproduces the exact character-level
     overlap formula used in zeroentropy-ai/legalbenchrag's own
     run_benchmark.py. This is verifiably identical to the benchmark's own
     metric, so it's the most defensible number for a like-for-like claim.

  2. MAP@10 — a standard rank-based IR metric computed on the SAME ground
     truth, using an explicitly documented relevance rule (a chunk counts as
     "relevant" if it overlaps a gold span by at least `min_overlap_ratio`).
     NOTE: the official LegalBench-RAG repo does not itself compute MAP@10 —
     RAGLite computed that independently on top of this ground truth, and
     hasn't published their exact relevance-binarization rule. So treat your
     MAP@10 as "MAP@10 computed on the same official benchmark data," not as
     a guaranteed exact reproduction of RAGLite's internal method — and say
     so explicitly in your thesis (template at the bottom of this file).

DATA YOU NEED
-------------
Download from the link in the LegalBench-RAG README:
  https://github.com/zeroentropy-ai/legalbenchrag
  (Dropbox link is under "Download" in that README)

After unzipping, you should have:
    data/
      corpus/         <- raw .txt files (the actual contracts), some nested
                          in subfolders
      benchmarks/
        cuad.json      <- THIS is the one you want (NOT privacy_qa.json,
                           maud.json, or contractnli.json)

Do not run their benchmark.py — it randomly subsamples every benchmark
down to 194 queries. Load cuad.json directly with this script instead, to
get the full ~462-contract / ~4,042-query scale that matches RAGLite's
reported numbers.

SCHEMA (confirmed from legalbenchrag/benchmark_types.py)
----------------------------------------------------------
cuad.json:
{
  "tests": [
    {
      "query": "...",
      "snippets": [
        {"file_path": "path/within/corpus/some_contract.txt", "span": [1234, 1400]},
        ...
      ],
      "tags": [...]   # optional, ignore
    },
    ...
  ]
}

`span` is a [start, end) character index range into the raw text file at
`data/corpus/{file_path}`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# Data classes matching the official schema
# ---------------------------------------------------------------------------

@dataclass
class Snippet:
    file_path: str
    start: int
    end: int


@dataclass
class QAGroundTruth:
    query: str
    snippets: List[Snippet]


@dataclass
class RetrievedChunk:
    file_path: str
    start: int
    end: int
    text: str = ""


# ---------------------------------------------------------------------------
# Loading the official benchmark
# ---------------------------------------------------------------------------

def load_cuad_benchmark(benchmark_json_path: str) -> List[QAGroundTruth]:
    """Loads data/benchmarks/cuad.json in full (no subsampling)."""
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tests = []
    for test in raw["tests"]:
        snippets = [
            Snippet(file_path=s["file_path"], start=s["span"][0], end=s["span"][1])
            for s in test["snippets"]
        ]
        tests.append(QAGroundTruth(query=test["query"], snippets=snippets))
    return tests


def referenced_file_paths(tests: List[QAGroundTruth]) -> set:
    """The set of corpus file paths actually used by CUAD's tests, so you
    only need to load/index ~462 contracts, not the full multi-dataset corpus."""
    paths = set()
    for t in tests:
        for s in t.snippets:
            paths.add(s.file_path)
    return paths


def load_corpus_texts(corpus_dir: str, file_paths: set) -> Dict[str, str]:
    """Loads only the referenced raw text files into memory: {file_path: content}."""
    corpus_root = Path(corpus_dir)
    texts = {}
    missing = []
    for fp in file_paths:
        full_path = corpus_root / fp
        if not full_path.exists():
            missing.append(fp)
            continue
        texts[fp] = full_path.read_text(encoding="utf-8", errors="ignore")
    if missing:
        print(f"WARNING: {len(missing)} referenced corpus files not found, "
              f"e.g. {missing[:3]}. Check your corpus_dir path.")
    return texts


# ---------------------------------------------------------------------------
# Chunking (adapt to match YOUR existing 500/50 chunker — this is a plain
# fixed-size fallback if you don't want to reuse LangChain's splitter here)
# ---------------------------------------------------------------------------

def chunk_text_with_offsets(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Tuple[str, int, int]]:
    """
    Returns list of (chunk_text, start_offset, end_offset) tuples.
    Prefer reusing your existing LangChain RecursiveCharacterTextSplitter
    with add_start_index=True instead of this — it respects sentence/
    paragraph boundaries, which this simple version doesn't. This is only
    here as a drop-in fallback.
    """
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((text[start:end], start, end))
        if end == n:
            break
        start = end - chunk_overlap
    return chunks


# ---------------------------------------------------------------------------
# OFFICIAL metric: exact character-overlap Precision@k / Recall@k
# (mirrors legalbenchrag/run_benchmark.py's QAResult.precision / .recall)
# ---------------------------------------------------------------------------

def official_precision_recall_at_k(
    gt_snippets: List[Snippet],
    retrieved: List[RetrievedChunk],
    k: int = 10,
) -> Tuple[float, float]:
    retrieved = retrieved[:k]

    total_retrieved_len = sum(r.end - r.start for r in retrieved)
    total_relevant_len = sum(g.end - g.start for g in gt_snippets)
    relevant_retrieved_len = 0

    for r in retrieved:
        for g in gt_snippets:
            if r.file_path == g.file_path:
                common_min = max(r.start, g.start)
                common_max = min(r.end, g.end)
                if common_max > common_min:
                    relevant_retrieved_len += common_max - common_min

    precision = relevant_retrieved_len / total_retrieved_len if total_retrieved_len else 0.0
    recall = relevant_retrieved_len / total_relevant_len if total_relevant_len else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# MAP@10 (binary relevance, explicitly documented threshold)
# ---------------------------------------------------------------------------

def _char_overlap(a: Snippet, b: RetrievedChunk) -> int:
    if a.file_path != b.file_path:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def is_relevant(gt_snippet: Snippet, chunk: RetrievedChunk, min_overlap_ratio: float) -> bool:
    """A chunk is 'relevant' if it covers at least `min_overlap_ratio` of the
    gold span's characters. min_overlap_ratio=0.0 means 'any overlap at all'."""
    overlap = _char_overlap(gt_snippet, chunk)
    if min_overlap_ratio <= 0.0:
        return overlap > 0
    gold_len = max(1, gt_snippet.end - gt_snippet.start)
    return (overlap / gold_len) >= min_overlap_ratio


def average_precision_at_k(
    gt_snippets: List[Snippet],
    retrieved: List[RetrievedChunk],
    k: int = 10,
    min_overlap_ratio: float = 0.0,
) -> float:
    retrieved = retrieved[:k]
    hits = [any(is_relevant(g, r, min_overlap_ratio) for g in gt_snippets) for r in retrieved]

    total_relevant = max(1, len(gt_snippets))
    num_hit = 0
    precision_sum = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            num_hit += 1
            precision_sum += num_hit / rank

    if num_hit == 0:
        return 0.0
    return precision_sum / min(total_relevant, k)


def mean_average_precision(
    qa_results: List[Tuple[QAGroundTruth, List[RetrievedChunk]]],
    k: int = 10,
    min_overlap_ratio: float = 0.0,
) -> float:
    if not qa_results:
        return 0.0
    aps = [
        average_precision_at_k(qa_gt.snippets, retrieved, k, min_overlap_ratio)
        for qa_gt, retrieved in qa_results
    ]
    return sum(aps) / len(aps)


def map_at_k_sensitivity(
    qa_results: List[Tuple[QAGroundTruth, List[RetrievedChunk]]],
    k: int = 10,
    thresholds: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.7),
) -> Dict[float, float]:
    return {t: mean_average_precision(qa_results, k, t) for t in thresholds}


# ---------------------------------------------------------------------------
# Putting it together — adapt this to your actual pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    benchmark_json_path: str,
    corpus_dir: str,
    retriever_query_fn,
    k: int = 10,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    limit_queries: Optional[int] = None,
):
    """
    retriever_query_fn: a function you provide with signature
        retriever_query_fn(query: str, k: int) -> List[RetrievedChunk]
    It must return RetrievedChunk objects carrying file_path + start + end,
    which means YOUR retriever needs to track which corpus file_path and
    character offset each chunk came from. See integration notes below.
    """
    print("Loading benchmark...")
    tests = load_cuad_benchmark(benchmark_json_path)
    if limit_queries:
        tests = tests[:limit_queries]
    print(f"Loaded {len(tests)} CUAD queries.")

    file_paths = referenced_file_paths(tests)
    print(f"Referenced {len(file_paths)} contract files.")

    print("Loading corpus text (this is where YOU plug in your indexing "
          "step — chunk + embed + build FAISS/BM25 over these texts)...")
    _corpus_texts = load_corpus_texts(corpus_dir, file_paths)
    # >>> At this point, index _corpus_texts with your existing pipeline,
    # >>> using chunk_text_with_offsets() or your LangChain splitter with
    # >>> add_start_index=True, so retriever_query_fn can return chunks
    # >>> with accurate file_path/start/end.

    qa_results: List[Tuple[QAGroundTruth, List[RetrievedChunk]]] = []
    for i, qa_gt in enumerate(tests):
        retrieved = retriever_query_fn(qa_gt.query, k)
        qa_results.append((qa_gt, retrieved))
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(tests)} queries retrieved")

    # Official metric
    precisions, recalls = [], []
    for qa_gt, retrieved in qa_results:
        p, r = official_precision_recall_at_k(qa_gt.snippets, retrieved, k)
        precisions.append(p)
        recalls.append(r)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    # MAP@10 with sensitivity sweep
    map10 = mean_average_precision(qa_results, k=k, min_overlap_ratio=0.0)
    sensitivity = map_at_k_sensitivity(qa_results, k=k)

    print("\n=== RESULTS (Official LegalBench-RAG CUAD benchmark) ===")
    print(f"Num queries evaluated: {len(qa_results)}")
    print(f"Precision@{k} (official character-overlap formula): {avg_precision:.3f}")
    print(f"Recall@{k}    (official character-overlap formula): {avg_recall:.3f}")
    print(f"MAP@{k} (any-overlap relevance): {map10:.3f}")
    print(f"MAP@{k} sensitivity across thresholds: {sensitivity}")

    return {
        "num_queries": len(qa_results),
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "map_at_k": map10,
        "map_at_k_sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------------
# INTEGRATION EXAMPLE — adapt the imports/variable names to your repo
# ---------------------------------------------------------------------------

def example_main():
    """
    Sketch only — wire up the commented-out parts to your actual
    ~/Documents/enterprise-rag retriever before running.
    """
    BENCHMARK_JSON = "./data/benchmarks/cuad.json"
    CORPUS_DIR = "./data/corpus"

    # --- Step A: build a fresh index over the CUAD corpus with your pipeline ---
    # from your_pipeline.chunking import chunk_text_with_offsets  # or use the
    #                                                              # one above
    # from your_pipeline.embeddings import embed_texts             # SBERT all-mpnet-base-v2
    # from your_pipeline.indexing import build_faiss_index, build_bm25_index
    #
    # tests = load_cuad_benchmark(BENCHMARK_JSON)
    # file_paths = referenced_file_paths(tests)
    # corpus_texts = load_corpus_texts(CORPUS_DIR, file_paths)
    #
    # all_chunks = []  # list of RetrievedChunk-shaped records, minus scores
    # for file_path, text in corpus_texts.items():
    #     for chunk_text, start, end in chunk_text_with_offsets(text, 500, 50):
    #         all_chunks.append({"file_path": file_path, "start": start, "end": end, "text": chunk_text})
    #
    # faiss_index = build_faiss_index([c["text"] for c in all_chunks])
    # bm25_index = build_bm25_index([c["text"] for c in all_chunks])

    # --- Step B: define retriever_query_fn using your hybrid retrieval ---
    def retriever_query_fn(query: str, k: int) -> List[RetrievedChunk]:
        # results = your_hybrid_retriever.get_relevant_documents(query, k=k)
        # return [
        #     RetrievedChunk(
        #         file_path=all_chunks[idx]["file_path"],
        #         start=all_chunks[idx]["start"],
        #         end=all_chunks[idx]["end"],
        #         text=all_chunks[idx]["text"],
        #     )
        #     for idx in results
        # ]
        raise NotImplementedError("Wire this up to your retriever before running.")

    results = run_evaluation(
        benchmark_json_path=BENCHMARK_JSON,
        corpus_dir=CORPUS_DIR,
        retriever_query_fn=retriever_query_fn,
        k=10,
        # limit_queries=200,  # uncomment to sanity-check on a small slice first
    )
    return results


if __name__ == "__main__":
    example_main()


# ---------------------------------------------------------------------------
# THESIS REPORTING TEMPLATE
# ---------------------------------------------------------------------------
"""
Suggested paragraph once you have real numbers:

"To enable a scale-matched comparison, the system was additionally evaluated
on the full CUAD subset of the LegalBench-RAG benchmark (Pipitone & Houir
Alami, 2024) — 462 contracts and N queries [fill in exact count from your
run] — using the benchmark's official character-level ground truth. Using
the benchmark's own precision/recall formula, the system achieved
Precision@10 = X and Recall@10 = Y. For comparison against the MAP@10 figure
reported by RAGLite (0.652; 0.719 with Cohere reranking), MAP@10 was also
computed on the same ground truth using a standard any-overlap relevance
rule, yielding Z. Note that the original LegalBench-RAG benchmark does not
itself define a MAP@10 metric; RAGLite's figure was computed independently
using an unpublished relevance-binarization method, so this MAP@10 value
should be read as 'MAP@10 computed on the same official ground truth,'
rather than a guaranteed exact reproduction of RAGLite's internal
methodology. A sensitivity analysis across overlap thresholds [list] showed
MAP@10 values of [list], indicating [stability/sensitivity] to this
choice."
"""
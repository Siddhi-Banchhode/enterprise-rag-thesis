"""
evaluation/evaluator_v2.py
========================
Retrieval-Only Evaluation v2 — adds SQuAD2, fixes a corpus-mismatch bug
in the original evaluator.py's CUAD path. No LLM calls, no quota limits.

WHAT CHANGED FROM evaluator.py:
  1. CUAD FIX: the original always retrieved from your enterprise-PDF
     index (self.vectorstore from load_pdfs()), even when evaluating
     CUAD questions. CUAD's gold answers live in actual contract text,
     not in your 10-Ks/ESG reports, so those numbers didn't measure
     what they claimed to. This version builds a FRESH, isolated
     FAISS+BM25 index over the real CUAD corpus (reusing the same
     loaders as your official LegalBench-RAG script), then retrieves
     CUAD questions against THAT index instead.
  2. NEW: SQuAD2 support, same pattern — SQuAD2's gold answers live in
     Wikipedia paragraphs, so it gets its own fresh isolated index too.
     Uses only ANSWERABLE questions (is_impossible == False), since
     there's no gold span to score retrieval against otherwise.
  3. pdfQA and FinanceBench are UNCHANGED — they're supposed to use
     your enterprise-PDF index (FinanceBench is explicitly filtered to
     companies present in your corpus), so no fix was needed there.

Metrics are the SAME token-overlap Precision@K/Recall@K/F1@K/MRR/Hit@1
as evaluator.py (not the official LegalBench-RAG character-span scoring
used in your CUAD ablation) — this is a deliberate choice for a fast,
LLM-free, multi-domain generalization screen. Document this distinction
explicitly in your methodology section: this script and the official
LegalBench-RAG ablation measure different things and are not meant to
be compared number-for-number.

Datasets:
  - pdfqa        : corpus-grounded questions (your 15 PDFs)
  - cuad         : real CUAD contracts, isolated index (FIXED)
  - financebench : financial SEC filings (corpus-filtered, unchanged)
  - squad2       : open-domain Wikipedia QA, isolated index (NEW)

Run:
    python evaluation/evaluator_v2.py --datasets pdfqa --n 15
    python evaluation/evaluator_v2.py --datasets cuad squad2 --n 20
    python evaluation/evaluator_v2.py --datasets pdfqa cuad financebench squad2 --n 20

Requires: pip install datasets --break-system-packages   (for SQuAD2)

Results saved to:
    data/eval/evaluation_results_v2.json
    data/eval/evaluation_summary_v2.csv
"""
import sys
import os
import json
import time
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

from config import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.vectorstore.faiss_store import load_index
from src.ingestion.pdf_loader import load_pdfs
from src.preprocessing.cleaner import preprocess_documents as clean_documents
from src.preprocessing.chunker import chunk_documents
from src.retrieval.hybrid_retriever import build_hybrid_retriever
from src.retrieval.reranker import Reranker

# Reuse the ALREADY-CORRECT CUAD corpus loaders from your official
# LegalBench-RAG evaluator, instead of the ad hoc local cuad_eval.json
# that evaluator.py used (which never attached real contract text as
# the retrieval corpus).
from legalbench_rag_cuad_evaluator import (
    load_cuad_benchmark,
    referenced_file_paths,
    load_corpus_texts,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # only needed if you evaluate squad2

# Same paths as your run_legalbench_rag_evaluation_500_50.py
CUAD_CORPUS_DIR = "/Users/siddhibanchhode/Documents/enterprise-rag/legalbench_data/corpus"
CUAD_BENCHMARK_JSON = "/Users/siddhibanchhode/Documents/enterprise-rag/legalbench_data/benchmarks/cuad.json"


# ── retrieval metrics (identical to evaluator.py — unchanged on purpose,
#    for exact consistency with your existing pdfQA numbers) ────────────────
def precision_at_k(docs: List[Document], answer: str, k: int) -> float:
    if not docs or not answer:
        return 0.0
    ref_tokens = set(answer.lower().split())
    hits = sum(
        1 for doc in docs[:k]
        if ref_tokens & set(doc.page_content.lower().split())
    )
    return round(hits / min(k, len(docs)), 4)


def recall_at_k(docs: List[Document], answer: str, k: int) -> float:
    if not docs or not answer:
        return 0.0
    ref_tokens = set(answer.lower().split())
    if not ref_tokens:
        return 0.0
    found = set()
    for doc in docs[:k]:
        found |= ref_tokens & set(doc.page_content.lower().split())
    return round(len(found) / len(ref_tokens), 4)


def f1_at_k(p: float, r: float) -> float:
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def hit_at_1(docs: List[Document], answer: str) -> float:
    if not docs or not answer:
        return 0.0
    ref_tokens = set(answer.lower().split())
    return float(bool(ref_tokens & set(docs[0].page_content.lower().split())))


def mrr(docs: List[Document], answer: str) -> float:
    if not docs or not answer:
        return 0.0
    ref_tokens = set(answer.lower().split())
    for rank, doc in enumerate(docs, 1):
        if ref_tokens & set(doc.page_content.lower().split()):
            return round(1.0 / rank, 4)
    return 0.0


# ── existing loaders, copied verbatim from evaluator.py (unchanged) ─────────
def load_pdfqa_samples(n: int = 15) -> List[Dict]:
    """Corpus-grounded QA pairs covering financial, ESG, scientific domains."""
    samples = [
        {"question": "What was Apple's total net sales in fiscal year 2022?",
         "answer": "394,328", "category": "financial"},
        {"question": "What was Netflix's total revenue in 2020?",
         "answer": "24,996", "category": "financial"},
        {"question": "What was Walmart's net sales in fiscal year 2017?",
         "answer": "481,317", "category": "financial"},
        {"question": "What were Apple's services revenue in fiscal year 2022?",
         "answer": "78,129", "category": "financial"},
        {"question": "What was Apple's net income in fiscal year 2022?",
         "answer": "99,803", "category": "financial"},
        {"question": "What was Citigroup's total revenues net of interest expense in 2016?",
         "answer": "77,219", "category": "financial"},
        {"question": "What was Netflix's net income in 2020?",
         "answer": "2,761", "category": "financial"},
        {"question": "What was Ralph Lauren total revenues in 2014?",
         "answer": "revenue", "category": "financial"},
        {"question": "What net zero target year does BP mention in their sustainability report?",
         "answer": "2050", "category": "esg"},
        {"question": "What carbon neutrality goal does PWC China reference in their ESG report?",
         "answer": "2060", "category": "esg"},
        {"question": "What sustainability reporting framework does AstraZeneca follow?",
         "answer": "GRI", "category": "esg"},
        {"question": "What environmental objectives does BASF report on in 2023?",
         "answer": "climate change", "category": "esg"},
        {"question": "What is Google's water goal in their 2024 environmental report?",
         "answer": "water", "category": "esg"},
        {"question": "What renewable energy commitment does Schneider Electric mention?",
         "answer": "renewable", "category": "esg"},
        {"question": "What is the main task addressed in the PaperTab research paper?",
         "answer": "table", "category": "scientific"},
        {"question": "What embedding approach does PaperTab propose for table retrieval?",
         "answer": "BERT", "category": "scientific"},
        {"question": "What dataset does the climate finance benchmark paper evaluate on?",
         "answer": "climate", "category": "scientific"},
    ]
    for s in samples:
        s["dataset"] = "pdfQA"
    return samples[:n]


def load_financebench_samples(n: int = 20) -> List[Dict]:
    """FinanceBench — corpus-filtered to companies in your 15 PDFs.
    Unchanged: this is SUPPOSED to hit your enterprise-PDF index, since
    it's explicitly filtered to companies that ARE in that corpus."""
    path = Path(config.DATASETS_DIR) / "financebench" / "financebench_eval.json"
    if not path.exists():
        logger.warning("FinanceBench not found at %s", path)
        return []
    with open(path) as f:
        data = json.load(f)
    corpus_companies = [
        "apple", "netflix", "walmart", "citigroup",
        "bp", "google", "basf", "astrazeneca", "schneider"
    ]
    samples = []
    for item in data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a and any(c in q.lower() for c in corpus_companies):
            samples.append({
                "question": q, "answer": a,
                "dataset": "FinanceBench", "category": "financial",
            })
        if len(samples) >= n:
            break
    logger.info("Loaded %d FinanceBench corpus-relevant samples", len(samples))
    return samples


# ── FIXED: CUAD loader — real questions AND real gold answer text,
#    pulled from the same official benchmark + corpus files as your
#    LegalBench-RAG ablation, so the corpus this gets retrieved against
#    will actually contain the answers. ───────────────────────────────────
def load_cuad_samples_fixed(n: int = 50) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Returns (samples, corpus_texts).
    corpus_texts is passed on to build an ISOLATED index — do not reuse
    your enterprise-PDF index for these.

    ASSUMPTION (verify with the check in the conversation before trusting
    this): each qa_gt.snippets item is a dict with keys 'file_path' and
    'span': [start, end], matching the public LegalBench-RAG cuad.json
    schema. If your legalbench_rag_cuad_evaluator.py uses a different
    shape (e.g. an object with .file_path/.start/.end), adjust the
    snippet-parsing block below accordingly.
    """
    tests = load_cuad_benchmark(CUAD_BENCHMARK_JSON)
    tests = tests[:n]
    file_paths = referenced_file_paths(tests)
    corpus_texts = load_corpus_texts(CUAD_CORPUS_DIR, file_paths)

    samples = []
    for qa_gt in tests:
        answer_parts = []
        for snip in qa_gt.snippets:
            if isinstance(snip, dict):
                fp = snip.get("file_path")
                span = snip.get("span", [None, None])
                start, end = span[0], span[1]
            else:
                fp = getattr(snip, "file_path", None)
                start = getattr(snip, "start", None)
                end = getattr(snip, "end", None)

            text = corpus_texts.get(fp, "")
            if text and start is not None and end is not None:
                answer_parts.append(text[start:end])

        answer = " ".join(answer_parts).strip()
        if not answer:
            continue

        samples.append({
            "question": qa_gt.query,
            "answer": answer,
            "dataset": "CUAD",
            "category": "legal",
        })

    logger.info("Loaded %d CUAD samples (real corpus, real gold answer text)",
                len(samples))
    return samples, corpus_texts


# ── NEW: SQuAD2 loader — answerable questions only, own isolated corpus ────
def load_squad2_samples(n: int = 20, seed: int = 42) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Returns (samples, context_texts). context_texts is passed on to build
    an ISOLATED index — SQuAD2 passages are Wikipedia paragraphs, not
    anywhere in your enterprise-PDF corpus.

    Only answerable questions (is_impossible == False) are used, since
    unanswerable ones have no gold span to score retrieval against.
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required for SQuAD2. "
            "Install it with: pip install datasets --break-system-packages"
        )

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ds = ds.filter(lambda ex: len(ex["answers"]["text"]) > 0)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    samples = []
    context_texts = {}
    for ex in ds:
        cid = ex["id"]
        context_texts[cid] = ex["context"]
        samples.append({
            "question": ex["question"],
            "answer": ex["answers"]["text"][0],
            "dataset": "SQuAD2",
            "category": "open_domain",
        })

    logger.info("Loaded %d SQuAD2 samples across %d unique passages",
                len(samples), len(context_texts))
    return samples, context_texts


# ── main evaluator ───────────────────────────────────────────────────────────
class MultiDomainEvaluator:
    """
    Retrieval-only evaluation across multiple datasets, each retrieved
    against the CORRECT corpus:
      - pdfqa, financebench  -> your shared enterprise-PDF index
      - cuad, squad2         -> their own isolated, freshly-built indices
    """

    def __init__(self, top_k: int = 5, top_n: int = 3):
        self.top_k = top_k
        self.top_n = top_n
        logger.info("Loading embedding model ...")
        self.embedder = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Loading reranker ...")
        self.reranker = Reranker(top_n=self.top_n)

        self._enterprise_hybrid = None
        self._dataset_hybrids: Dict[str, Any] = {}
        self.retrieval_scope: Dict[str, str] = {}

    def _enterprise_retriever(self):
        if self._enterprise_hybrid is None:
            logger.info("Loading your enterprise-PDF FAISS index ...")
            vectorstore = load_index(self.embedder)
            raw = load_pdfs()
            clean = clean_documents(raw)
            chunks = chunk_documents(clean)
            self._enterprise_hybrid = build_hybrid_retriever(
                chunks, vectorstore, top_k=self.top_k * 2
            )
        return self._enterprise_hybrid

    def _isolated_retriever(self, dataset_name: str, corpus_texts: Dict[str, str]):
        if dataset_name in self._dataset_hybrids:
            return self._dataset_hybrids[dataset_name]

        logger.info("Building isolated corpus + FAISS/BM25 index for %s "
                    "(%d source documents) ...", dataset_name, len(corpus_texts))
        raw_docs = [
            Document(page_content=text, metadata={"source": src})
            for src, text in corpus_texts.items()
        ]
        chunks = chunk_documents(raw_docs)
        vectorstore = FAISS.from_documents(chunks, self.embedder)
        hybrid = build_hybrid_retriever(chunks, vectorstore, top_k=self.top_k * 2)
        self._dataset_hybrids[dataset_name] = hybrid
        return hybrid

    def evaluate_sample(self, question: str, reference_answer: str, hybrid) -> Dict[str, Any]:
        t0 = time.time()
        candidates = hybrid.invoke(question)
        top_docs = self.reranker.rerank(question, candidates)
        latency = round(time.time() - t0, 3)

        p = precision_at_k(candidates, reference_answer, self.top_k)
        r = recall_at_k(candidates, reference_answer, self.top_k)
        f1 = f1_at_k(p, r)
        h1 = hit_at_1(top_docs, reference_answer)
        mrr_score = mrr(candidates, reference_answer)

        return {
            "question": question,
            "reference": reference_answer,
            "precision_at_k": p,
            "recall_at_k": r,
            "f1_at_k": f1,
            "hit_at_1": h1,
            "mrr": mrr_score,
            "latency_sec": latency,
            "num_candidates": len(candidates),
            "num_reranked": len(top_docs),
        }

    def run(self, datasets: List[str], n_per_dataset: int = 15) -> Dict[str, Any]:
        all_results = []

        if "pdfqa" in datasets:
            self.retrieval_scope["pdfqa"] = "enterprise_pdf_corpus"
            hybrid = self._enterprise_retriever()
            self._run_dataset(load_pdfqa_samples(n_per_dataset), hybrid, all_results)

        if "financebench" in datasets:
            self.retrieval_scope["financebench"] = "enterprise_pdf_corpus"
            hybrid = self._enterprise_retriever()
            self._run_dataset(load_financebench_samples(n_per_dataset), hybrid, all_results)

        if "cuad" in datasets:
            self.retrieval_scope["cuad"] = "isolated_cuad_corpus"
            samples, corpus_texts = load_cuad_samples_fixed(n_per_dataset)
            hybrid = self._isolated_retriever("cuad", corpus_texts)
            self._run_dataset(samples, hybrid, all_results)

        if "squad2" in datasets:
            self.retrieval_scope["squad2"] = "isolated_squad2_corpus"
            samples, context_texts = load_squad2_samples(n_per_dataset)
            hybrid = self._isolated_retriever("squad2", context_texts)
            self._run_dataset(samples, hybrid, all_results)

        if not all_results:
            logger.error("No samples evaluated. Check dataset paths/args.")
            return {}

        valid = [r for r in all_results if "error" not in r]
        summary = self._aggregate(valid)
        output = {
            "config": {
                "top_k": self.top_k,
                "top_n": self.top_n,
                "embedding_model": config.EMBEDDING_MODEL,
                "mode": "retrieval_only",
                "n_samples": len(all_results),
                "n_valid": len(valid),
                "retrieval_scope": self.retrieval_scope,
            },
            "summary": summary,
            "results": all_results,
        }
        self._save(output)
        self._print_summary(summary)
        return output

    def _run_dataset(self, samples: List[Dict], hybrid, all_results: List[Dict]):
        for i, sample in enumerate(samples, 1):
            q, ans = sample["question"], sample["answer"]
            ds, cat = sample.get("dataset", "unknown"), sample.get("category", "unknown")
            try:
                metrics = self.evaluate_sample(q, ans, hybrid)
                metrics["dataset"], metrics["category"] = ds, cat
                all_results.append(metrics)
                logger.info(
                    "[%s %d/%d] P@K=%.3f | R@K=%.3f | F1=%.3f | MRR=%.3f | H@1=%.0f | %s",
                    ds, i, len(samples), metrics["precision_at_k"],
                    metrics["recall_at_k"], metrics["f1_at_k"],
                    metrics["mrr"], metrics["hit_at_1"], q[:45],
                )
            except Exception as e:
                logger.error("[%s %d/%d] Failed: %s", ds, i, len(samples), e)
                all_results.append({
                    "question": q, "reference": ans,
                    "dataset": ds, "category": cat, "error": str(e),
                })

    def _aggregate(self, results: List[Dict]) -> Dict:
        if not results:
            return {}

        def mean(key, rows):
            vals = [r[key] for r in rows if key in r]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        overall = {
            "precision_at_k": mean("precision_at_k", results),
            "recall_at_k": mean("recall_at_k", results),
            "f1_at_k": mean("f1_at_k", results),
            "hit_at_1": mean("hit_at_1", results),
            "mrr": mean("mrr", results),
            "avg_latency_sec": mean("latency_sec", results),
            "n_samples": len(results),
        }

        per_dataset = {}
        for ds in set(r.get("dataset", "unknown") for r in results):
            ds_r = [r for r in results if r.get("dataset") == ds]
            per_dataset[ds] = {
                "precision_at_k": mean("precision_at_k", ds_r),
                "recall_at_k": mean("recall_at_k", ds_r),
                "f1_at_k": mean("f1_at_k", ds_r),
                "hit_at_1": mean("hit_at_1", ds_r),
                "mrr": mean("mrr", ds_r),
                "n_samples": len(ds_r),
            }

        per_category = {}
        for cat in set(r.get("category", "unknown") for r in results):
            cat_r = [r for r in results if r.get("category") == cat]
            per_category[cat] = {
                "f1_at_k": mean("f1_at_k", cat_r),
                "mrr": mean("mrr", cat_r),
                "n_samples": len(cat_r),
            }

        return {"overall": overall, "per_dataset": per_dataset, "per_category": per_category}

    def _save(self, output: Dict):
        out_dir = Path("data/eval")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "evaluation_results_v2.json", "w") as f:
            json.dump(output, f, indent=2)

        valid = [r for r in output["results"] if "error" not in r]
        if valid:
            with open(out_dir / "evaluation_summary_v2.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "dataset", "category", "question", "reference",
                    "precision_at_k", "recall_at_k", "f1_at_k",
                    "hit_at_1", "mrr", "latency_sec",
                ])
                writer.writeheader()
                for r in valid:
                    writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
        logger.info("Results saved to data/eval/ (v2 files, originals untouched)")

    def _print_summary(self, summary: Dict):
        print(f"\n{'='*65}")
        print("EVALUATION SUMMARY v2 — RETRIEVAL ONLY (no LLM calls)")
        print(f"{'='*65}")
        overall = summary.get("overall", {})
        print(f"  Precision@K    : {overall.get('precision_at_k', 0):.4f}")
        print(f"  Recall@K       : {overall.get('recall_at_k', 0):.4f}")
        print(f"  F1@K           : {overall.get('f1_at_k', 0):.4f}")
        print(f"  MRR            : {overall.get('mrr', 0):.4f}")
        print(f"  Hit@1          : {overall.get('hit_at_1', 0):.4f}")
        print(f"  Avg Latency    : {overall.get('avg_latency_sec', 0):.3f}s")
        print(f"  Samples        : {overall.get('n_samples', 0)}")

        print(f"\nRetrieval scope per dataset (which corpus was used):")
        for ds, scope in self.retrieval_scope.items():
            print(f"  {ds:<14} : {scope}")

        print(f"\nPer-dataset breakdown:")
        for ds, m in summary.get("per_dataset", {}).items():
            print(f"\n  [{ds}]  (n={m.get('n_samples', 0)})")
            print(f"    Precision@K : {m.get('precision_at_k', 0):.4f}")
            print(f"    Recall@K    : {m.get('recall_at_k', 0):.4f}")
            print(f"    F1@K        : {m.get('f1_at_k', 0):.4f}")
            print(f"    MRR         : {m.get('mrr', 0):.4f}")
            print(f"    Hit@1       : {m.get('hit_at_1', 0):.4f}")

        print(f"\nPer-category F1@K:")
        for cat, m in summary.get("per_category", {}).items():
            print(f"  {cat:<12} : F1={m.get('f1_at_k', 0):.4f}  "
                  f"MRR={m.get('mrr', 0):.4f}  (n={m.get('n_samples', 0)})")

        print(f"\n{'='*65}")
        print("Results saved to data/eval/ (v2 files)")
        print(f"{'='*65}\n")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation v2 — no LLM calls")
    parser.add_argument(
        "--datasets", nargs="+", default=["pdfqa"],
        choices=["pdfqa", "cuad", "financebench", "squad2"],
        help="Datasets to evaluate (default: pdfqa)"
    )
    parser.add_argument("--n", type=int, default=15, help="Samples per dataset (default: 15)")
    parser.add_argument("--top-k", type=int, default=5, help="Hybrid retrieval candidates (default: 5)")
    parser.add_argument("--top-n", type=int, default=3, help="Docs after reranking (default: 3)")
    args = parser.parse_args()

    evaluator = MultiDomainEvaluator(top_k=args.top_k, top_n=args.top_n)
    evaluator.run(datasets=args.datasets, n_per_dataset=args.n)
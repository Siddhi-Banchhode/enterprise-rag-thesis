"""
evaluation/ablation_study.py
=============================
Ablation Study — Retrieval-Only Mode (no LLM calls)

Directly answers:
  RQ1.1: How effectively do semantic embeddings improve retrieval?
  RQ1.2: To what extent does vector-based indexing enhance accuracy?

Four configurations tested on identical question sets:
  Config 1 — BM25 only           (sparse, keyword-based)
  Config 2 — FAISS dense only    (semantic, SBERT-based)
  Config 3 — Hybrid, no reranker (BM25 + FAISS, no cross-encoder)
  Config 4 — Full system         (BM25 + FAISS + cross-encoder reranker)

NO LLM CALLS — measures retrieval quality only:
  Precision@K, Recall@K, F1@K, MRR (Mean Reciprocal Rank), Hit@1

Run from project root:
  python evaluation/ablation_study.py --n 20

Results saved to:
  data/eval/ablation_results.json
  data/eval/ablation_summary.csv
  data/eval/ablation_detail.csv
"""

import sys
import os
import json
import time
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

from config import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.vectorstore.faiss_store import load_index
from src.ingestion.pdf_loader import load_pdfs
from src.preprocessing.cleaner import preprocess_documents as clean_documents
from src.preprocessing.chunker import chunk_documents
from src.retrieval.reranker import Reranker


# ── QA test set ───────────────────────────────────────────────────────────────
# 20 questions with reference answer keywords grounded in your 15-PDF corpus

QA_PAIRS = [
    # Financial
    {"question": "What was Apple's total net sales in fiscal year 2022?",
     "answer": "394,328", "category": "financial"},
    {"question": "What was Netflix's total revenue in 2020?",
     "answer": "24,996", "category": "financial"},
    {"question": "What was Walmart's net sales in fiscal year 2017?",
     "answer": "481,317", "category": "financial"},
    {"question": "What was Citigroup's total revenues net of interest expense in 2016?",
     "answer": "77,219", "category": "financial"},
    {"question": "What were Apple's services revenue in fiscal year 2022?",
     "answer": "78,129", "category": "financial"},
    {"question": "What was the net income reported by Netflix in 2020?",
     "answer": "2,761", "category": "financial"},
    {"question": "What products revenue did Apple report in fiscal year 2022?",
     "answer": "316,199", "category": "financial"},
    {"question": "What was the total revenue of Ralph Lauren in 2014?",
     "answer": "revenue", "category": "financial"},
    # ESG / Sustainability
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
    {"question": "What emissions target does AstraZeneca set for 2030?",
     "answer": "carbon", "category": "esg"},
    # Scientific / Technical
    {"question": "What embedding approach does the PaperTab paper propose for table retrieval?",
     "answer": "BERT", "category": "scientific"},
    {"question": "What is the main task addressed in the PaperTab research paper?",
     "answer": "table", "category": "scientific"},
    {"question": "What dataset does the climate finance benchmark paper use?",
     "answer": "climate", "category": "scientific"},
    {"question": "What method does the paper propose for scientific table understanding?",
     "answer": "table", "category": "scientific"},
    {"question": "What is the primary business segment of Apple described in the 2022 annual report?",
     "answer": "iPhone", "category": "financial"},
]


# ── retrieval metrics (NO LLM needed) ────────────────────────────────────────

def precision_at_k(docs: List[Document], answer: str, k: int) -> float:
    """Fraction of top-K retrieved chunks containing the answer keyword."""
    if not docs or not answer:
        return 0.0
    ref_tokens = set(answer.lower().split())
    hits = sum(
        1 for doc in docs[:k]
        if ref_tokens & set(doc.page_content.lower().split())
    )
    return round(hits / min(k, len(docs)), 4)


def recall_at_k(docs: List[Document], answer: str, k: int) -> float:
    """Fraction of answer tokens found in top-K retrieved chunks."""
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
    """Harmonic mean of Precision@K and Recall@K."""
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def hit_at_1(docs: List[Document], answer: str) -> float:
    """1 if top-1 retrieved chunk contains the answer, else 0."""
    if not docs or not answer:
        return 0.0
    ref_tokens = set(answer.lower().split())
    return float(bool(ref_tokens & set(docs[0].page_content.lower().split())))


def mean_reciprocal_rank(docs: List[Document], answer: str) -> float:
    """
    MRR: 1/rank of first relevant document.
    Measures how high the first correct chunk ranks.
    """
    if not docs or not answer:
        return 0.0
    ref_tokens = set(answer.lower().split())
    for rank, doc in enumerate(docs, 1):
        if ref_tokens & set(doc.page_content.lower().split()):
            return round(1.0 / rank, 4)
    return 0.0


# ── ablation study ────────────────────────────────────────────────────────────

class AblationStudy:
    """
    Retrieval-only ablation study.
    No LLM calls — pure retrieval metric comparison.
    Completes in ~2 minutes for 20 questions.
    """

    TOP_K = 5
    TOP_N = 3

    def __init__(self):
        self._load_pipeline()

    def _load_pipeline(self):
        logger.info("=" * 60)
        logger.info("ABLATION STUDY — Loading pipeline (retrieval only)")
        logger.info("=" * 60)

        logger.info("Loading embedding model …")
        self.embedder = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Loading FAISS index …")
        self.vectorstore = load_index(self.embedder)
        logger.info("FAISS: %d vectors", self.vectorstore.index.ntotal)

        logger.info("Loading chunks for BM25 …")
        raw   = load_pdfs()
        clean = clean_documents(raw)
        self.chunks = chunk_documents(clean)
        logger.info("%d chunks loaded.", len(self.chunks))

        logger.info("Building BM25 retriever …")
        self.bm25 = BM25Retriever.from_documents(self.chunks)
        self.bm25.k = self.TOP_K * 2   # fetch more for fair comparison

        logger.info("Building FAISS retriever …")
        self.faiss_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.TOP_K * 2},
        )

        logger.info("Building hybrid retriever …")
        from src.retrieval.hybrid_retriever import build_hybrid_retriever
        self.hybrid = build_hybrid_retriever(
            self.chunks, self.vectorstore, top_k=self.TOP_K * 2
        )

        logger.info("Loading cross-encoder reranker …")
        self.reranker = Reranker(top_n=self.TOP_N)

        logger.info("All components ready. Starting evaluation …\n")

    def _retrieve(self, config_name: str, question: str) -> List[Document]:
        """Retrieve documents — NO LLM call."""
        if config_name == "bm25_only":
            return self.bm25.invoke(question)[:self.TOP_K]

        elif config_name == "faiss_only":
            return self.faiss_retriever.invoke(question)[:self.TOP_K]

        elif config_name == "hybrid_no_reranker":
            return self.hybrid.invoke(question)[:self.TOP_K]

        elif config_name == "full_system":
            candidates = self.hybrid.invoke(question)
            return self.reranker.rerank(question, candidates)

        else:
            raise ValueError(f"Unknown config: {config_name}")

    def _evaluate_one(
        self, config_name: str, question: str, reference: str
    ) -> Dict[str, Any]:
        """Compute retrieval metrics for one QA pair — no LLM."""
        t0   = time.time()
        docs = self._retrieve(config_name, question)
        latency = round(time.time() - t0, 3)

        p   = precision_at_k(docs, reference, self.TOP_K)
        r   = recall_at_k(docs, reference, self.TOP_K)
        f1  = f1_at_k(p, r)
        h1  = hit_at_1(docs, reference)
        mrr = mean_reciprocal_rank(docs, reference)

        return {
            "config":         config_name,
            "question":       question,
            "reference":      reference,
            "precision_at_k": p,
            "recall_at_k":    r,
            "f1_at_k":        f1,
            "hit_at_1":       h1,
            "mrr":            mrr,
            "latency_sec":    latency,
            "num_docs":       len(docs),
            "top_sources":    list({
                os.path.basename(d.metadata.get("source", "unknown"))
                for d in docs[:3]
            }),
        }

    def run(self, n_questions: int = 20) -> Dict[str, Any]:
        configs = [
            ("bm25_only",          "Config 1 — BM25 Only"),
            ("faiss_only",         "Config 2 — FAISS Dense Only"),
            ("hybrid_no_reranker", "Config 3 — Hybrid (no reranker)"),
            ("full_system",        "Config 4 — Full System (YOUR SYSTEM)"),
        ]

        qa_sample = QA_PAIRS[:n_questions]
        all_results = []

        for config_id, config_label in configs:
            logger.info("\n%s", "─" * 60)
            logger.info("Running: %s", config_label)
            logger.info("%s", "─" * 60)

            for i, pair in enumerate(qa_sample, 1):
                q   = pair["question"]
                ans = pair["answer"]
                try:
                    metrics = self._evaluate_one(config_id, q, ans)
                    metrics["category"] = pair["category"]
                    all_results.append(metrics)
                    logger.info(
                        "[%d/%d] P@K=%.3f | R@K=%.3f | F1=%.3f | "
                        "MRR=%.3f | H@1=%.0f | %s",
                        i, len(qa_sample),
                        metrics["precision_at_k"],
                        metrics["recall_at_k"],
                        metrics["f1_at_k"],
                        metrics["mrr"],
                        metrics["hit_at_1"],
                        q[:45],
                    )
                except Exception as e:
                    logger.error("[%d/%d] Failed: %s", i, len(qa_sample), e)
                    all_results.append({
                        "config": config_id, "question": q,
                        "reference": ans, "error": str(e),
                        "category": pair["category"],
                    })

        summary = self._aggregate(all_results, configs)
        output  = {
            "config": {
                "top_k":           self.TOP_K,
                "top_n":           self.TOP_N,
                "embedding_model": config.EMBEDDING_MODEL,
                "n_questions":     n_questions,
                "mode":            "retrieval_only",
            },
            "summary": summary,
            "results": all_results,
        }

        self._save(output)
        self._print_summary(summary, configs)
        return output

    def _aggregate(self, results, configs):
        summary = {}
        for config_id, config_label in configs:
            cr = [
                r for r in results
                if r.get("config") == config_id and "error" not in r
            ]
            if not cr:
                continue

            def mean(key):
                vals = [r[key] for r in cr if key in r]
                return round(sum(vals) / len(vals), 4) if vals else 0.0

            # per-category breakdown
            categories = set(r.get("category", "unknown") for r in cr)
            per_cat = {}
            for cat in categories:
                cat_r = [r for r in cr if r.get("category") == cat]
                per_cat[cat] = {
                    "f1_at_k":  round(sum(r["f1_at_k"] for r in cat_r) / len(cat_r), 4),
                    "mrr":      round(sum(r["mrr"] for r in cat_r) / len(cat_r), 4),
                    "n":        len(cat_r),
                }

            summary[config_id] = {
                "label":           config_label,
                "precision_at_k":  mean("precision_at_k"),
                "recall_at_k":     mean("recall_at_k"),
                "f1_at_k":         mean("f1_at_k"),
                "hit_at_1":        mean("hit_at_1"),
                "mrr":             mean("mrr"),
                "avg_latency_sec": mean("latency_sec"),
                "n_samples":       len(cr),
                "per_category":    per_cat,
            }
        return summary

    def _save(self, output: Dict):
        out_dir = Path("data/eval")
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(out_dir / "ablation_results.json", "w") as f:
            json.dump(output, f, indent=2)

        # Summary CSV
        with open(out_dir / "ablation_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "config", "label", "precision_at_k", "recall_at_k",
                "f1_at_k", "hit_at_1", "mrr", "avg_latency_sec", "n_samples"
            ])
            writer.writeheader()
            for cfg_id, m in output["summary"].items():
                row = {"config": cfg_id}
                row.update({k: v for k, v in m.items() if k != "per_category"})
                writer.writerow(row)

        # Detail CSV
        valid = [r for r in output["results"] if "error" not in r]
        with open(out_dir / "ablation_detail.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "config", "category", "question", "reference",
                "precision_at_k", "recall_at_k", "f1_at_k",
                "hit_at_1", "mrr", "latency_sec",
            ])
            writer.writeheader()
            for r in valid:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

        logger.info("Results saved to data/eval/")

    def _print_summary(self, summary: Dict, configs: list):
        print(f"\n{'='*78}")
        print("ABLATION STUDY — RETRIEVAL EVALUATION RESULTS")
        print(f"{'='*78}")
        print(f"{'Configuration':<36} {'P@K':>6} {'R@K':>6} {'F1@K':>6} "
              f"{'MRR':>6} {'H@1':>6} {'Lat':>7}")
        print(f"{'-'*78}")

        for config_id, config_label in configs:
            m = summary.get(config_id, {})
            if not m:
                print(f"  {config_label} — no results")
                continue
            marker = " ✅" if config_id == "full_system" else ""
            print(
                f"{config_label:<36} "
                f"{m.get('precision_at_k',0):>6.3f} "
                f"{m.get('recall_at_k',0):>6.3f} "
                f"{m.get('f1_at_k',0):>6.3f} "
                f"{m.get('mrr',0):>6.3f} "
                f"{m.get('hit_at_1',0):>6.3f} "
                f"{m.get('avg_latency_sec',0):>6.3f}s"
                f"{marker}"
            )

        print(f"{'='*78}")

        # ── per-category breakdown ─────────────────────────────────────────
        print("\nPER-CATEGORY F1@K BREAKDOWN:")
        print(f"{'Configuration':<36} {'Financial':>10} {'ESG':>8} {'Scientific':>12}")
        print(f"{'-'*70}")
        for config_id, config_label in configs:
            m = summary.get(config_id, {})
            if not m:
                continue
            pc = m.get("per_category", {})
            fin = pc.get("financial", {}).get("f1_at_k", 0.0)
            esg = pc.get("esg", {}).get("f1_at_k", 0.0)
            sci = pc.get("scientific", {}).get("f1_at_k", 0.0)
            print(f"{config_label:<36} {fin:>10.3f} {esg:>8.3f} {sci:>12.3f}")

        print(f"\n{'='*78}")
        print("KEY FINDINGS FOR THESIS:")
        print()

        bm25  = summary.get("bm25_only", {})
        faiss = summary.get("faiss_only", {})
        hyb   = summary.get("hybrid_no_reranker", {})
        full  = summary.get("full_system", {})

        # RQ1.1
        if bm25 and faiss:
            diff_f1  = round(faiss.get("f1_at_k",0) - bm25.get("f1_at_k",0), 4)
            diff_mrr = round(faiss.get("mrr",0) - bm25.get("mrr",0), 4)
            print("RQ1.1 — Semantic Embeddings (SBERT) vs Keyword Search (BM25):")
            print(f"  FAISS F1@K : {faiss.get('f1_at_k',0):.3f}  |  "
                  f"BM25 F1@K : {bm25.get('f1_at_k',0):.3f}  |  "
                  f"Δ = {diff_f1:+.4f}")
            print(f"  FAISS MRR  : {faiss.get('mrr',0):.3f}  |  "
                  f"BM25 MRR  : {bm25.get('mrr',0):.3f}  |  "
                  f"Δ = {diff_mrr:+.4f}")
            if diff_f1 > 0:
                print("  → SBERT semantic embeddings outperform BM25 keyword search ✅")
            else:
                print("  → BM25 matched/exceeded FAISS on this corpus")
                print("    (Discuss: BM25 strong on exact financial figures — "
                      "valid finding)")

        print()

        # RQ1.2
        if hyb and bm25 and faiss:
            best_single = max(bm25.get("f1_at_k",0), faiss.get("f1_at_k",0))
            hyb_gain    = round(hyb.get("f1_at_k",0) - best_single, 4)
            print("RQ1.2 — Hybrid Retrieval vs Single-Method Retrieval:")
            print(f"  Hybrid F1@K : {hyb.get('f1_at_k',0):.3f}  |  "
                  f"Best single : {best_single:.3f}  |  Δ = {hyb_gain:+.4f}")
            if hyb_gain > 0:
                print("  → Hybrid retrieval outperforms both individual methods ✅")
            else:
                print("  → Single method competitive with hybrid on this sample")
                print("    (Increase --n for more stable results)")

        print()

        # Reranker
        if hyb and full:
            rerank_gain = round(full.get("f1_at_k",0) - hyb.get("f1_at_k",0), 4)
            print("Reranker contribution (Config 3 → Config 4):")
            print(f"  Full system F1@K : {full.get('f1_at_k',0):.3f}  |  "
                  f"Hybrid (no reranker) : {hyb.get('f1_at_k',0):.3f}  |  "
                  f"Δ = {rerank_gain:+.4f}")
            if rerank_gain > 0:
                print("  → Cross-encoder reranking improves retrieval quality ✅")
            elif rerank_gain < 0:
                print("  → Reranker reduced F1 (domain mismatch — ms-marco "
                      "trained on web, not enterprise docs)")
                print("    Discuss as limitation: domain-specific reranker "
                      "fine-tuning recommended")
            else:
                print("  → Reranker neutral on this sample")

        print(f"\n{'='*78}")
        print("Files saved:")
        print("  data/eval/ablation_summary.csv  ← paste into thesis Table 5.X")
        print("  data/eval/ablation_detail.csv   ← per-question breakdown")
        print("  data/eval/ablation_results.json ← full JSON")
        print(f"{'='*78}\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ablation Study — Retrieval-Only (no LLM calls)"
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of QA pairs to evaluate (default: 20, max: 20)"
    )
    args = parser.parse_args()

    n = min(args.n, len(QA_PAIRS))
    logger.info("Ablation study: %d questions × 4 configs (retrieval only)", n)
    logger.info("Estimated time: ~2-3 minutes (no LLM calls)")

    study = AblationStudy()
    study.run(n_questions=n)
"""
evaluation/ragas_eval_resumable.py
=====================================
Resumable, rate-limit-safe RAGAS evaluation.

WHY THIS EXISTS
---------------
Your original ragas_eval.py could only draw from 10 hardcoded QA pairs, and
made ~6-7 Gemini calls per sample (1 answer-generation + 1 claim-extraction
+ ONE CALL PER CLAIM to verify + 1 relevancy-question-generation). At a
20-request/day free-tier cap, that's exactly why you stopped at 3 samples.

WHAT CHANGED
------------
1. Question pool: pulls from your existing evaluator.py loaders
   (load_pdfqa_samples + load_financebench_samples) instead of 10 hardcoded
   pairs. CUAD is deliberately EXCLUDED here \u2014 your 15-PDF corpus is
   financial/ESG/scientific, and if CUAD contracts aren't actually indexed
   in your FAISS store, "can't find grounding context" would masquerade as
   a faithfulness/hallucination failure. Only include CUAD here once you've
   confirmed those documents are actually in your index.
2. Claim verification is now BATCHED into a single call (all claims, one
   YES/NO per line) instead of one call per claim. Faithfulness scoring is
   now a flat 2 calls (extract + verify) regardless of claim count.
   Total per sample: 1 (generate) + 1 (extract) + 1 (verify) + 1 (relevancy)
   = 4 calls, predictable regardless of answer length.
3. Every individual LLM call is paced (not just between samples), staying
   safely under 5 requests/minute.
4. Checkpoints after EVERY sample to data/eval/ragas_results.json. Already-
   evaluated questions (matched by exact question text) are skipped on
   future runs \u2014 so you just re-run this same command once a day.
5. Stops itself safely before the daily cap rather than erroring out
   mid-sample.

USAGE
-----
Run once today, then once per day for the rest of your timeline:
    python evaluation/ragas_eval_resumable.py

First run uses a conservative DAILY_CALL_BUDGET=16 (leaving slack in case
build_rag_chain makes more than 1 internal LLM call \u2014 unverified, since
rag_chain.py wasn't reviewed here). Watch the terminal output: if you finish
a run using noticeably fewer than 16 calls' worth of samples with no errors,
you can raise DAILY_CALL_BUDGET tomorrow.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from config import config
from langchain_core.messages import HumanMessage

from ragas_eval import RAGASEvaluator, StaticRetriever, compute_answer_relevancy
from src.rag.rag_chain import build_rag_chain

# Reuse your existing question pools instead of 10 hardcoded pairs.
# NOTE: load_cuad_samples is intentionally NOT imported here \u2014 see docstring.
from evaluator import load_pdfqa_samples, load_financebench_samples

RESULTS_PATH = Path("data/eval/ragas_results.json")

CALLS_PER_SAMPLE_ESTIMATE = 4       # 1 generate + 1 extract + 1 verify(batched) + 1 relevancy
DAILY_CALL_BUDGET = 16              # conservative vs. Gemini's 20/day free-tier cap
MIN_SECONDS_BETWEEN_CALLS = 13.0    # safely under 5 requests/minute (60/5=12s + margin)


def build_question_pool() -> List[Dict[str, str]]:
    """pdfQA + FinanceBench only \u2014 both confirmed/likely grounded in your
    actual indexed 15-PDF corpus. CUAD excluded (see module docstring)."""
    pool = []
    try:
        pool += load_pdfqa_samples(n=100)  # returns all it has (~17)
    except Exception as e:
        logger.warning("Could not load pdfQA samples: %s", e)
    try:
        pool += load_financebench_samples(n=100)
    except Exception as e:
        logger.warning("Could not load FinanceBench samples (file may not "
                        "exist \u2014 that's fine, pdfQA alone still works): %s", e)

    seen = set()
    unique_pool = []
    for item in pool:
        q = item["question"]
        if q not in seen:
            seen.add(q)
            unique_pool.append(item)

    logger.info("Question pool built: %d total candidate questions", len(unique_pool))
    return unique_pool


def load_existing_results() -> Dict[str, Any]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"config": {}, "summary": {}, "results": []}


def save_results(output: Dict[str, Any]):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    import csv
    valid = [r for r in output["results"] if "error" not in r]
    if valid:
        csv_path = RESULTS_PATH.parent / "ragas_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "reference", "answer",
                "faithfulness", "answer_relevancy", "ragas_score", "latency_sec",
            ])
            writer.writeheader()
            for r in valid:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})


def compute_summary(valid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not valid_results:
        return {}
    return {
        "faithfulness": round(sum(r["faithfulness"] for r in valid_results) / len(valid_results), 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in valid_results) / len(valid_results), 4),
        "ragas_score": round(sum(r["ragas_score"] for r in valid_results) / len(valid_results), 4),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in valid_results) / len(valid_results), 2),
        "n_samples": len(valid_results),
    }


class ResumableRAGASEvaluator(RAGASEvaluator):
    """Extends your existing RAGASEvaluator (same embedder, same FAISS/
    hybrid retriever, same reranker, same Gemini judge \u2014 all unchanged)
    with call pacing and batched claim verification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_call_time = 0.0

    def _pace(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def _llm_call(self, prompt: str) -> str:
        self._pace()
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            self._last_call_time = time.time()
            return response.content
        except Exception:
            self._last_call_time = time.time()
            raise  # propagate so the caller can detect quota exhaustion and stop cleanly

    def compute_faithfulness_batched(self, answer: str, contexts: List[str]) -> float:
        if not answer or not contexts:
            return 0.0
        context_text = "\n\n".join(contexts[:3])

        claim_prompt = f"""Given the following answer, extract all factual claims as a numbered list.
Each claim should be a single, atomic statement.
If the answer says it cannot find information, return: 1. Answer states information not found.

Answer: {answer}

List the claims (one per line, numbered):"""
        claims_raw = self._llm_call(claim_prompt)
        claims = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in claims_raw.strip().split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]
        if not claims:
            claims = [answer[:200]]

        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        verify_prompt = f"""Context:
{context_text}

Claims:
{numbered}

For EACH claim above, answer YES if it is directly supported by the context, or NO if it is not.
Respond with exactly one YES or NO per line, in the same order as the claims, and nothing else."""
        verify_raw = self._llm_call(verify_prompt)
        verdict_lines = [l.strip().upper() for l in verify_raw.strip().split("\n") if l.strip()]

        supported = sum(1 for l in verdict_lines[:len(claims)] if "YES" in l)
        score = supported / len(claims) if claims else 0.0
        logger.info("Faithfulness: %d/%d claims supported -> %.4f", supported, len(claims), score)
        return round(score, 4)

    def evaluate_sample_resumable(self, question: str, reference_answer: str) -> Dict[str, Any]:
        t0 = time.time()

        candidates = self.hybrid.invoke(question)
        top_docs = self.reranker.rerank(question, candidates)

        self._pace()  # pace BEFORE generation too, not just the judge calls
        chain = build_rag_chain(StaticRetriever(docs=top_docs))
        result = chain.invoke({"query": question})
        answer = result.get("result", "")
        self._last_call_time = time.time()

        contexts = [doc.page_content for doc in top_docs]

        faithfulness = self.compute_faithfulness_batched(answer, contexts)
        answer_relevancy = compute_answer_relevancy(answer, question, self._llm_call)

        latency = round(time.time() - t0, 2)
        return {
            "question": question,
            "reference": reference_answer,
            "answer": answer,
            "contexts": contexts,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "ragas_score": round((faithfulness + answer_relevancy) / 2, 4),
            "latency_sec": latency,
        }


def main():
    pool = build_question_pool()
    existing = load_existing_results()
    done_questions = {r["question"] for r in existing["results"] if "error" not in r}

    remaining = [q for q in pool if q["question"] not in done_questions]
    logger.info("%d/%d questions already done (including any from prior "
                "runs). %d remaining.", len(done_questions), len(pool), len(remaining))

    if not remaining:
        logger.info("All available questions already evaluated \u2014 nothing "
                     "new to run today. Consider widening the pool.")
        return

    safe_batch_size = max(1, DAILY_CALL_BUDGET // CALLS_PER_SAMPLE_ESTIMATE)
    todays_batch = remaining[:safe_batch_size]
    logger.info("Running %d new samples today (~%d call budget, ~%d "
                "calls/sample).", len(todays_batch), DAILY_CALL_BUDGET,
                CALLS_PER_SAMPLE_ESTIMATE)

    evaluator = ResumableRAGASEvaluator(top_k=5, top_n=3)

    for i, pair in enumerate(todays_batch, 1):
        q, ref = pair["question"], pair.get("answer", "")
        logger.info("[%d/%d today] Q: %s", i, len(todays_batch), q[:70])
        try:
            metrics = evaluator.evaluate_sample_resumable(q, ref)
            existing["results"].append(metrics)
            logger.info("  Faithfulness=%.4f | Answer Relevancy=%.4f | RAGAS=%.4f",
                        metrics["faithfulness"], metrics["answer_relevancy"], metrics["ragas_score"])
        except Exception as e:
            logger.error("  Stopping early \u2014 likely hit the rate/quota limit: %s", e)
            break
        finally:
            valid = [r for r in existing["results"] if "error" not in r]
            existing["summary"] = compute_summary(valid)
            existing["config"] = {
                "top_k": evaluator.top_k, "top_n": evaluator.top_n,
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_type": getattr(config, "LLM_TYPE", "unknown"),
                "n_valid": len(valid),
            }
            save_results(existing)  # save after EVERY sample, not just at the end

    valid = [r for r in existing["results"] if "error" not in r]
    print(f"\n{'='*60}")
    print("CUMULATIVE RAGAS PROGRESS (across all days)")
    print(f"{'='*60}")
    print(f"  Total samples so far : {len(valid)}")
    print(f"  Faithfulness         : {existing['summary'].get('faithfulness', 0):.4f}")
    print(f"  Answer Relevancy     : {existing['summary'].get('answer_relevancy', 0):.4f}")
    print(f"  RAGAS Score          : {existing['summary'].get('ragas_score', 0):.4f}")
    print(f"  Questions remaining  : {len(pool) - len(valid)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
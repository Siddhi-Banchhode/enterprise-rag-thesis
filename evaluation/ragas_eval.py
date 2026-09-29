"""
evaluation/ragas_eval.py
=========================
RAGAS Evaluation — Faithfulness + Answer Relevancy

RAGAS evaluates the quality of RAG-generated answers using LLM-based metrics:
  - Faithfulness      : Is the answer grounded in the retrieved context?
                        (1.0 = fully grounded, 0.0 = hallucinated)
  - Answer Relevancy  : Is the answer relevant to the question?
                        (1.0 = fully relevant, 0.0 = irrelevant)

These two metrics directly address RQ1.3:
  "How effectively can RAG reduce hallucinations and improve contextual
   reliability of responses?"

Run from project root:
    python evaluation/ragas_eval.py

Results saved to:
    data/eval/ragas_results.json
    data/eval/ragas_summary.csv
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── imports ───────────────────────────────────────────────────────────────────
from config import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.vectorstore.faiss_store import load_index
from src.ingestion.pdf_loader import load_pdfs
from src.preprocessing.cleaner import preprocess_documents as clean_documents
from src.preprocessing.chunker import chunk_documents
from src.retrieval.hybrid_retriever import build_hybrid_retriever
from src.retrieval.reranker import Reranker
from src.rag.rag_chain import build_rag_chain


# ── static retriever wrapper ──────────────────────────────────────────────────

class StaticRetriever(BaseRetriever):
    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs


# ── RAGAS metric implementations ──────────────────────────────────────────────

def compute_faithfulness(
    answer: str,
    contexts: List[str],
    question: str,
    llm_fn,
) -> float:
    """
    Faithfulness: fraction of answer claims that are supported by the context.

    Uses an LLM to extract claims from the answer and verify each against
    the retrieved context. This mirrors the official RAGAS faithfulness metric.

    Parameters
    ----------
    answer : str
        Generated answer from the RAG chain.
    contexts : list of str
        Retrieved document chunks used to generate the answer.
    question : str
        Original user question.
    llm_fn : callable
        Function that takes a prompt string and returns a string response.

    Returns
    -------
    float : faithfulness score between 0.0 and 1.0
    """
    if not answer or not contexts:
        return 0.0

    context_text = "\n\n".join(contexts[:3])

    # Step 1: Extract claims from the answer
    claim_prompt = f"""Given the following answer, extract all factual claims as a numbered list.
Each claim should be a single, atomic statement.
If the answer says it cannot find information, return: 1. Answer states information not found.

Answer: {answer}

List the claims (one per line, numbered):"""

    claims_raw = llm_fn(claim_prompt)
    claims = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in claims_raw.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    if not claims:
        claims = [answer[:200]]

    # Step 2: Verify each claim against the context
    supported = 0
    for claim in claims:
        verify_prompt = f"""Context:
{context_text}

Claim: {claim}

Is this claim directly supported by the context above?
Answer with only YES or NO."""

        verdict = llm_fn(verify_prompt).strip().upper()
        if "YES" in verdict:
            supported += 1

    score = supported / len(claims) if claims else 0.0
    logger.info(
        "Faithfulness: %d/%d claims supported → %.4f",
        supported, len(claims), score,
    )
    return round(score, 4)


def compute_answer_relevancy(
    answer: str,
    question: str,
    llm_fn,
    n_questions: int = 3,
) -> float:
    """
    Answer Relevancy: semantic similarity between the original question
    and questions generated from the answer.

    A relevant answer should contain enough information to reconstruct
    the original question. This mirrors the official RAGAS metric.

    Parameters
    ----------
    answer : str
        Generated answer from the RAG chain.
    question : str
        Original user question.
    llm_fn : callable
        Function that takes a prompt string and returns a string response.
    n_questions : int
        Number of questions to generate from the answer.

    Returns
    -------
    float : answer relevancy score between 0.0 and 1.0
    """
    if not answer or not question:
        return 0.0

    # Generate questions from the answer
    gen_prompt = f"""Given the following answer, generate {n_questions} different questions 
that this answer could be responding to. Make them specific and varied.

Answer: {answer}

Generate exactly {n_questions} questions (one per line, numbered):"""

    gen_raw = llm_fn(gen_prompt)
    generated_questions = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in gen_raw.strip().split("\n")
        if line.strip() and (line.strip()[0].isdigit() or "?" in line)
    ][:n_questions]

    if not generated_questions:
        return 0.0

    # Compute cosine similarity between original and generated questions
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    orig_emb = model.encode(question, normalize_embeddings=True)
    gen_embs = model.encode(generated_questions, normalize_embeddings=True)

    similarities = [float(np.dot(orig_emb, g)) for g in gen_embs]
    score = float(np.mean(similarities)) if similarities else 0.0

    logger.info(
        "Answer Relevancy: %.4f (mean of %d generated questions)",
        score, len(generated_questions),
    )
    return round(max(0.0, score), 4)


# ── RAGAS evaluator ───────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    RAGAS-style evaluation: Faithfulness + Answer Relevancy.

    Uses Gemini (via LangChain) as the judge LLM, matching your thesis stack.
    """

    def __init__(self, top_k: int = 5, top_n: int = 3):
        self.top_k = top_k
        self.top_n = top_n
        self._setup_pipeline()

    def _setup_pipeline(self):
        logger.info("Loading embedding model …")
        self.embedder = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Loading FAISS index …")
        self.vectorstore = load_index(self.embedder)

        logger.info("Loading chunks …")
        raw   = load_pdfs()
        clean = clean_documents(raw)
        self.chunks = chunk_documents(clean)

        logger.info("Building hybrid retriever …")
        self.hybrid = build_hybrid_retriever(
            self.chunks, self.vectorstore, top_k=self.top_k
        )

        logger.info("Loading reranker …")
        self.reranker = Reranker(top_n=self.top_n)

        logger.info("Setting up Gemini LLM judge …")
        from langchain_google_genai import ChatGoogleGenerativeAI
        self._llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        logger.info("Pipeline ready.")

    def _llm_call(self, prompt: str) -> str:
        """Call the judge LLM with a plain text prompt."""
        from langchain_core.messages import HumanMessage
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return ""

    def evaluate_sample(
        self, question: str, reference_answer: str
    ) -> Dict[str, Any]:
        """Run one sample through the full pipeline and compute RAGAS metrics."""
        t0 = time.time()

        # 1. Retrieve
        candidates = self.hybrid.invoke(question)
        top_docs   = self.reranker.rerank(question, candidates)

        # 2. Generate answer
        chain  = build_rag_chain(StaticRetriever(docs=top_docs))
        result = chain.invoke({"query": question})
        answer = result.get("result", "")

        # 3. Extract context strings
        contexts = [doc.page_content for doc in top_docs]

        # 4. RAGAS metrics
        faithfulness = compute_faithfulness(
            answer, contexts, question, self._llm_call
        )
        answer_relevancy = compute_answer_relevancy(
            answer, question, self._llm_call
        )

        latency = round(time.time() - t0, 2)

        return {
            "question":          question,
            "reference":         reference_answer,
            "answer":            answer,
            "contexts":          contexts,
            "faithfulness":      faithfulness,
            "answer_relevancy":  answer_relevancy,
            "ragas_score":       round((faithfulness + answer_relevancy) / 2, 4),
            "latency_sec":       latency,
        }

    def run(self, n_samples: int = 10) -> Dict[str, Any]:
        """
        Run RAGAS evaluation over pdfQA samples.

        Parameters
        ----------
        n_samples : int
            Number of QA pairs to evaluate.

        Returns
        -------
        dict with per-sample results and aggregate summary.
        """
        # QA pairs grounded in your 15-PDF corpus
        qa_pairs = [
            {
                "question": "What was Apple's total net sales in fiscal year 2022?",
                "reference": "Apple's total net sales in fiscal year 2022 were $394,328 million.",
            },
            {
                "question": "What is the net zero target year mentioned in the BP sustainability report?",
                "reference": "BP aims to achieve net zero by 2050.",
            },
            {
                "question": "What was Walmart's total revenue in fiscal year 2017?",
                "reference": "Walmart's net sales were $481,317 million in fiscal year 2017.",
            },
            {
                "question": "What sustainability reporting framework does AstraZeneca use?",
                "reference": "AstraZeneca follows the GRI sustainability reporting framework.",
            },
            {
                "question": "What was Netflix's total revenue in 2020?",
                "reference": "Netflix's total revenue in 2020 was $24,996 million.",
            },
            {
                "question": "What is the primary topic of the PaperTab research paper?",
                "reference": "PaperTab focuses on table retrieval and understanding in research papers.",
            },
            {
                "question": "What carbon neutrality goal does PWC China mention in their ESG report?",
                "reference": "PWC China references China's carbon neutrality goal of 2060.",
            },
            {
                "question": "What was Citigroup's total revenue net of interest expense in 2016?",
                "reference": "Citigroup's total revenues net of interest expense were $77,219 million in 2016.",
            },
            {
                "question": "What environmental objectives does BASF report on in 2023?",
                "reference": "BASF reports on six environmental objectives including climate change mitigation.",
            },
            {
                "question": "What does Google report about water consumption in their 2024 environmental report?",
                "reference": "Google reports on water consumption and replenishment goals in their 2024 environmental report.",
            },
        ][:n_samples]

        logger.info("Running RAGAS evaluation on %d samples …", len(qa_pairs))
        results = []

        for i, pair in enumerate(qa_pairs, 1):
            q   = pair["question"]
            ref = pair["reference"]
            logger.info("[%d/%d] Q: %s", i, len(qa_pairs), q[:60])

            try:
                metrics = self.evaluate_sample(q, ref)
                results.append(metrics)
                logger.info(
                    "  Faithfulness=%.4f | Answer Relevancy=%.4f | RAGAS=%.4f",
                    metrics["faithfulness"],
                    metrics["answer_relevancy"],
                    metrics["ragas_score"],
                )
            except Exception as e:
                logger.error("  Failed: %s", e)
                results.append({
                    "question": q, "reference": ref, "error": str(e)
                })

            # Respect Gemini rate limits
            time.sleep(2.0)

        # ── aggregate ──────────────────────────────────────────────────────
        valid = [r for r in results if "error" not in r]
        summary = {}
        if valid:
            summary = {
                "faithfulness":     round(sum(r["faithfulness"] for r in valid) / len(valid), 4),
                "answer_relevancy": round(sum(r["answer_relevancy"] for r in valid) / len(valid), 4),
                "ragas_score":      round(sum(r["ragas_score"] for r in valid) / len(valid), 4),
                "avg_latency_sec":  round(sum(r["latency_sec"] for r in valid) / len(valid), 2),
                "n_samples":        len(valid),
            }

        output = {
            "config": {
                "top_k":           self.top_k,
                "top_n":           self.top_n,
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_type":        config.LLM_TYPE,
                "n_samples":       len(qa_pairs),
                "n_valid":         len(valid),
            },
            "summary": summary,
            "results": results,
        }

        self._save(output)
        self._print_summary(summary)
        return output

    def _save(self, output: Dict):
        """Save results to JSON and CSV."""
        out_dir = Path("data/eval")
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "ragas_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("RAGAS results saved to %s", json_path)

        import csv
        csv_path = out_dir / "ragas_summary.csv"
        valid = [r for r in output["results"] if "error" not in r]
        if valid:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "question", "reference", "answer",
                    "faithfulness", "answer_relevancy", "ragas_score", "latency_sec"
                ])
                writer.writeheader()
                for r in valid:
                    writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
            logger.info("CSV saved to %s", csv_path)

    def _print_summary(self, summary: Dict):
        print(f"\n{'='*60}")
        print("RAGAS EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Faithfulness      : {summary.get('faithfulness', 0):.4f}")
        print(f"  Answer Relevancy  : {summary.get('answer_relevancy', 0):.4f}")
        print(f"  RAGAS Score       : {summary.get('ragas_score', 0):.4f}")
        print(f"  Avg Latency       : {summary.get('avg_latency_sec', 0):.2f}s")
        print(f"  Samples           : {summary.get('n_samples', 0)}")
        print(f"\n  Interpretation:")
        f = summary.get('faithfulness', 0)
        ar = summary.get('answer_relevancy', 0)
        print(f"  - Faithfulness {f:.2f} → ", end="")
        if f >= 0.8:
            print("Excellent — answers are well-grounded in context")
        elif f >= 0.6:
            print("Good — most claims are supported by context")
        elif f >= 0.4:
            print("Moderate — some hallucination present")
        else:
            print("Low — significant hallucination detected")
        print(f"  - Answer Relevancy {ar:.2f} → ", end="")
        if ar >= 0.8:
            print("Excellent — answers are highly relevant to questions")
        elif ar >= 0.6:
            print("Good — answers are mostly on-topic")
        else:
            print("Moderate — answers may drift from the question")
        print(f"\n{'='*60}")
        print("Results saved to data/eval/ragas_results.json")
        print(f"{'='*60}\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS Evaluation Pipeline")
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of samples to evaluate (default: 5)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Hybrid retrieval candidates (default: 5)"
    )
    parser.add_argument(
        "--top-n", type=int, default=3,
        help="Docs after reranking (default: 3)"
    )
    args = parser.parse_args()

    evaluator = RAGASEvaluator(top_k=args.top_k, top_n=args.top_n)
    evaluator.run(n_samples=args.n)
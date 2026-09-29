"""
evaluation/run_rag_vs_no_rag_hallucination.py
================================================

RQ3 CONTROLLED PAIRED EXPERIMENT
--------------------------------

Research objective:
Determine whether supplying retrieved documentary evidence reduces factual
hallucination and improves factual correctness compared with the same LLM
operating without retrieval.

Condition A — No RAG
    Question -> Gemini 2.5 Flash -> Answer

Condition B — RAG
    Question -> FAISS dense retrieval -> Top-10 evidence
             -> Gemini 2.5 Flash -> Answer

CONTROLLED:
    - Same questions
    - Same reference answers
    - Same Gemini model
    - Same temperature
    - Same answer prompt structure
    - Same factuality judge
    - Same evaluation procedure

INDEPENDENT VARIABLE:
    Retrieved documentary evidence supplied: YES / NO

Final retrieval configuration from RQ4:
    Chunk size      : 2000
    Chunk overlap   : 200
    Embedding model : all-mpnet-base-v2
    Retrieval       : FAISS-only
    Top K           : 10
    BM25 weight     : 0.0
    Reranker        : None
    Query strategy  : Original query

Primary RQ3 metrics:
    - Hallucination rate
    - Correct-answer rate
    - Partial-correct rate
    - Abstention rate
    - Absolute hallucination reduction
    - Relative hallucination reduction

Methodological safeguards:
    - Paired experimental design
    - Same reference ground truth for both conditions
    - Judge blinded to RAG / No-RAG condition
    - Randomized answer order for judging
    - Raw retrieved source metadata retained for failure analysis

Input questions:
    Existing successful samples from:
        data/eval/ragas_results.json

Output:
    data/eval/rag_vs_no_rag_hallucination_results.json
    data/eval/rag_vs_no_rag_hallucination_summary.csv

Run:
    python3 evaluation/run_rag_vs_no_rag_hallucination.py
"""

import sys
import os
import json
import time
import csv
import random
import logging

from pathlib import Path
from typing import List, Dict, Any, Optional


# ============================================================================
# PROJECT PATH
# ============================================================================

sys.path.append(
    str(Path(__file__).parent.parent)
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from config import config

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

from src.vectorstore.faiss_store import load_index


# ============================================================================
# PATHS
# ============================================================================

SOURCE_RESULTS_PATH = Path(
    "data/eval/ragas_results.json"
)

RESULTS_PATH = Path(
    "data/eval/rag_vs_no_rag_hallucination_results.json"
)

CSV_PATH = Path(
    "data/eval/rag_vs_no_rag_hallucination_summary.csv"
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

TOP_K = 10

# Three Gemini calls per sample:
# 1. No-RAG generation
# 2. RAG generation
# 3. Blinded paired judge
CALLS_PER_SAMPLE = 3

# Conservative Gemini request budget.
DAILY_CALL_BUDGET = 15

# Minimum delay between Gemini API calls.
MIN_SECONDS_BETWEEN_CALLS = 13.0

# Reproducible randomization of judge answer order.
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================================
# CONTROLLED ANSWER PROMPT
# ============================================================================

ANSWER_PROMPT = """
You are answering a factual question.

Question:
{question}

Documentary evidence:
{context}

Instructions:
- Answer the question accurately and concisely.
- If documentary evidence is provided, use it as the primary evidence.
- Do not invent unsupported facts.
- If the documentary evidence is insufficient, say that the answer cannot
  be determined from the available evidence.
- If no documentary evidence is provided, answer using your existing
  knowledge.
- Do not explain the evaluation process.

Answer:
"""


# ============================================================================
# BLINDED FACTUALITY / HALLUCINATION JUDGE
# ============================================================================

JUDGE_PROMPT = """
You are evaluating two candidate answers to the SAME factual question.

Evaluate each answer independently against the supplied reference answer.

You are NOT told how either answer was generated.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

For EACH answer, assign exactly ONE classification.

CORRECT
The answer is factually consistent with the reference answer.
Equivalent wording, formatting, units, or reasonable rounding are acceptable.

PARTIALLY_CORRECT
The main answer contains some correct information but is incomplete,
imprecise, or contains a minor factual error.

HALLUCINATED
The answer asserts a factual answer that conflicts with the reference answer,
introduces unsupported factual details, or invents a specific answer.

ABSTAINED
The answer explicitly states that the requested information cannot be
determined, cannot be found, or that insufficient information is available.

Important rules:
- Judge factual correctness, not writing quality.
- Do not penalize harmless wording differences.
- A confident incorrect numerical answer is HALLUCINATED.
- An explicit refusal because information is unavailable is ABSTAINED,
  not HALLUCINATED.
- Evaluate Answer A and Answer B independently.
- Do not infer how an answer was generated.

Return valid JSON ONLY:

{{
  "answer_a": {{
    "classification": "CORRECT|PARTIALLY_CORRECT|HALLUCINATED|ABSTAINED",
    "reason": "brief explanation"
  }},
  "answer_b": {{
    "classification": "CORRECT|PARTIALLY_CORRECT|HALLUCINATED|ABSTAINED",
    "reason": "brief explanation"
  }}
}}
"""


# ============================================================================
# VALID CLASSIFICATIONS
# ============================================================================

CLASSES = [
    "CORRECT",
    "PARTIALLY_CORRECT",
    "HALLUCINATED",
    "ABSTAINED",
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_json_response(text: str) -> str:
    """
    Remove Markdown code fences if Gemini surrounds JSON with them.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]

    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def load_question_pool() -> List[Dict[str, str]]:
    """
    Reuse successful questions from the existing RAGAS-style evaluation.

    This provides a paired question set for the RQ3 experiment.
    """

    if not SOURCE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{SOURCE_RESULTS_PATH} does not exist."
        )

    with open(
        SOURCE_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    questions = []

    for row in data.get("results", []):

        if "error" in row:
            continue

        question = str(
            row.get("question", "")
        ).strip()

        reference = str(
            row.get("reference", "")
        ).strip()

        if not question:
            continue

        if not reference:

            logger.warning(
                "Skipping question because reference answer is empty: %s",
                question[:80],
            )

            continue

        questions.append(
            {
                "question": question,
                "reference": reference,
            }
        )

    # ----------------------------------------------------------------------
    # Deduplicate questions while preserving original order.
    # ----------------------------------------------------------------------

    seen = set()
    unique = []

    for item in questions:

        question = item["question"]

        if question not in seen:

            seen.add(question)
            unique.append(item)

    logger.info(
        "Loaded %d unique paired questions.",
        len(unique),
    )

    return unique


def load_existing_results() -> Dict[str, Any]:
    """
    Load resumable experiment results if they already exist.
    """

    if RESULTS_PATH.exists():

        with open(
            RESULTS_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    return {
        "config": {},
        "summary": {},
        "results": [],
    }


# ============================================================================
# SUMMARY METRICS
# ============================================================================

def count_classes(
    rows: List[Dict[str, Any]],
    condition: str,
) -> Dict[str, int]:

    counts = {
        c: 0
        for c in CLASSES
    }

    for row in rows:

        classification = (
            row.get(condition, {})
            .get("classification", "")
            .upper()
        )

        if classification in counts:
            counts[classification] += 1

    return counts


def compute_transition_counts(
    rows: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Count paired transitions between No-RAG and RAG outcomes.

    Useful for interpreting where RAG helped or hurt.
    """

    transitions = {}

    for row in rows:

        no_class = (
            row.get("no_rag", {})
            .get("classification", "")
            .upper()
        )

        rag_class = (
            row.get("rag", {})
            .get("classification", "")
            .upper()
        )

        if (
            no_class not in CLASSES
            or rag_class not in CLASSES
        ):
            continue

        key = (
            f"{no_class}_TO_{rag_class}"
        )

        transitions[key] = (
            transitions.get(key, 0) + 1
        )

    return transitions


def compute_summary(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:

    valid = [
        r
        for r in rows
        if "error" not in r
        and r.get(
            "no_rag",
            {}
        ).get(
            "classification"
        )
        and r.get(
            "rag",
            {}
        ).get(
            "classification"
        )
    ]

    n = len(valid)

    if n == 0:
        return {}

    no_counts = count_classes(
        valid,
        "no_rag",
    )

    rag_counts = count_classes(
        valid,
        "rag",
    )

    no_hall_rate = (
        no_counts["HALLUCINATED"]
        / n
    )

    rag_hall_rate = (
        rag_counts["HALLUCINATED"]
        / n
    )

    no_correct_rate = (
        no_counts["CORRECT"]
        / n
    )

    rag_correct_rate = (
        rag_counts["CORRECT"]
        / n
    )

    no_partial_rate = (
        no_counts["PARTIALLY_CORRECT"]
        / n
    )

    rag_partial_rate = (
        rag_counts["PARTIALLY_CORRECT"]
        / n
    )

    no_abstain_rate = (
        no_counts["ABSTAINED"]
        / n
    )

    rag_abstain_rate = (
        rag_counts["ABSTAINED"]
        / n
    )

    absolute_hallucination_reduction = (
        no_hall_rate
        - rag_hall_rate
    )

    if no_hall_rate > 0:

        relative_hallucination_reduction = (
            absolute_hallucination_reduction
            / no_hall_rate
        )

    else:

        relative_hallucination_reduction = None

    transitions = (
        compute_transition_counts(
            valid
        )
    )

    return {

        "n_samples":
            n,

        "no_rag": {

            "counts":
                no_counts,

            "correct_rate":
                round(
                    no_correct_rate,
                    4,
                ),

            "partial_correct_rate":
                round(
                    no_partial_rate,
                    4,
                ),

            "hallucination_rate":
                round(
                    no_hall_rate,
                    4,
                ),

            "abstention_rate":
                round(
                    no_abstain_rate,
                    4,
                ),
        },

        "rag": {

            "counts":
                rag_counts,

            "correct_rate":
                round(
                    rag_correct_rate,
                    4,
                ),

            "partial_correct_rate":
                round(
                    rag_partial_rate,
                    4,
                ),

            "hallucination_rate":
                round(
                    rag_hall_rate,
                    4,
                ),

            "abstention_rate":
                round(
                    rag_abstain_rate,
                    4,
                ),
        },

        "correct_rate_absolute_change":
            round(
                rag_correct_rate
                - no_correct_rate,
                4,
            ),

        "hallucination_absolute_reduction":
            round(
                absolute_hallucination_reduction,
                4,
            ),

        "hallucination_relative_reduction":

            (
                round(
                    relative_hallucination_reduction,
                    4,
                )

                if (
                    relative_hallucination_reduction
                    is not None
                )

                else None
            ),

        "paired_transitions":
            transitions,
    }


# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_results(
    output: Dict[str, Any]
):

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # JSON
    # ----------------------------------------------------------------------

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ----------------------------------------------------------------------
    # CSV
    # ----------------------------------------------------------------------

    valid = [
        row
        for row in output["results"]
        if "error" not in row
    ]

    if not valid:
        return

    fieldnames = [

        "question",
        "reference",

        "no_rag_answer",
        "no_rag_classification",
        "no_rag_reason",

        "rag_answer",
        "rag_classification",
        "rag_reason",

        "retrieved_context_count",
        "retrieved_sources",
        "retrieved_pages",

        "judge_order",
        "latency_sec",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in valid:

            evidence = (
                row.get(
                    "rag",
                    {}
                ).get(
                    "retrieved_evidence",
                    []
                )
            )

            sources = [
                str(
                    item.get(
                        "source",
                        ""
                    )
                )
                for item in evidence
            ]

            pages = [
                str(
                    item.get(
                        "page",
                        ""
                    )
                )
                for item in evidence
            ]

            writer.writerow({

                "question":
                    row.get(
                        "question",
                        "",
                    ),

                "reference":
                    row.get(
                        "reference",
                        "",
                    ),

                "no_rag_answer":
                    row.get(
                        "no_rag",
                        {},
                    ).get(
                        "answer",
                        "",
                    ),

                "no_rag_classification":
                    row.get(
                        "no_rag",
                        {},
                    ).get(
                        "classification",
                        "",
                    ),

                "no_rag_reason":
                    row.get(
                        "no_rag",
                        {},
                    ).get(
                        "reason",
                        "",
                    ),

                "rag_answer":
                    row.get(
                        "rag",
                        {},
                    ).get(
                        "answer",
                        "",
                    ),

                "rag_classification":
                    row.get(
                        "rag",
                        {},
                    ).get(
                        "classification",
                        "",
                    ),

                "rag_reason":
                    row.get(
                        "rag",
                        {},
                    ).get(
                        "reason",
                        "",
                    ),

                "retrieved_context_count":
                    len(evidence),

                "retrieved_sources":
                    " | ".join(
                        sources
                    ),

                "retrieved_pages":
                    " | ".join(
                        pages
                    ),

                "judge_order":
                    row.get(
                        "judge_order",
                        "",
                    ),

                "latency_sec":
                    row.get(
                        "latency_sec",
                        "",
                    ),
            })


# ============================================================================
# CONTROLLED EVALUATOR
# ============================================================================

class RAGvsNoRAGEvaluator:

    def __init__(self):

        self._last_call_time = 0.0

        # ------------------------------------------------------------------
        # Embedding model
        # ------------------------------------------------------------------

        logger.info(
            "Loading embedding model: %s",
            config.EMBEDDING_MODEL,
        )

        self.embedder = (
            HuggingFaceEmbeddings(

                model_name=
                    config.EMBEDDING_MODEL,

                model_kwargs={
                    "device": "cpu"
                },

                encode_kwargs={
                    "normalize_embeddings": True
                },
            )
        )

        # ------------------------------------------------------------------
        # Existing optimized FAISS index
        # ------------------------------------------------------------------

        logger.info(
            "Loading FAISS vector store from %s...",
            config.INDEX_PATH,
        )

        self.vectorstore = (
            load_index(
                self.embedder
            )
        )

        # ------------------------------------------------------------------
        # Gemini generator + judge
        # ------------------------------------------------------------------

        logger.info(
            "Loading Gemini model: %s",
            config.GEMINI_MODEL,
        )

        from langchain_google_genai import (
            ChatGoogleGenerativeAI
        )

        self.llm = (
            ChatGoogleGenerativeAI(

                model=
                    config.GEMINI_MODEL,

                temperature=
                    0,

                google_api_key=
                    os.getenv(
                        "GOOGLE_API_KEY"
                    ),
            )
        )

        logger.info(
            "RQ3 evaluation pipeline ready."
        )


    # ----------------------------------------------------------------------
    # API pacing
    # ----------------------------------------------------------------------

    def _pace(self):

        elapsed = (
            time.time()
            - self._last_call_time
        )

        if (
            elapsed
            < MIN_SECONDS_BETWEEN_CALLS
        ):

            wait = (
                MIN_SECONDS_BETWEEN_CALLS
                - elapsed
            )

            logger.info(
                "Waiting %.1fs for Gemini rate limit...",
                wait,
            )

            time.sleep(
                wait
            )


    def llm_call(
        self,
        prompt: str,
    ) -> str:

        self._pace()

        try:

            response = (
                self.llm.invoke(
                    [
                        HumanMessage(
                            content=prompt
                        )
                    ]
                )
            )

            self._last_call_time = (
                time.time()
            )

            return str(
                response.content
            )

        except Exception:

            self._last_call_time = (
                time.time()
            )

            raise


    # ----------------------------------------------------------------------
    # FINAL RQ4 RETRIEVAL CONFIGURATION:
    # FAISS ONLY / TOP 10 / NO RERANKER / ORIGINAL QUERY
    # ----------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
    ):

        docs = (
            self.vectorstore
            .similarity_search(
                question,
                k=TOP_K,
            )
        )

        return docs


    # ----------------------------------------------------------------------
    # Answer generation
    # ----------------------------------------------------------------------

    def generate(
        self,
        question: str,
        contexts: Optional[List[str]],
    ) -> str:

        if contexts:

            context_text = (
                "\n\n".join(

                    f"[Retrieved passage {i}]\n{text}"

                    for i, text
                    in enumerate(
                        contexts,
                        1,
                    )
                )
            )

        else:

            context_text = (
                "[No external documentary "
                "evidence supplied]"
            )

        prompt = (
            ANSWER_PROMPT.format(

                question=
                    question,

                context=
                    context_text,
            )
        )

        return (
            self.llm_call(
                prompt
            ).strip()
        )


    # ----------------------------------------------------------------------
    # BLINDED + RANDOMIZED paired judge
    # ----------------------------------------------------------------------

    def judge(
        self,
        question: str,
        reference: str,
        no_rag_answer: str,
        rag_answer: str,
    ) -> Dict[str, Any]:

        # --------------------------------------------------------------
        # Randomize which condition appears as Answer A / Answer B.
        # --------------------------------------------------------------

        no_rag_is_a = (
            random.choice(
                [True, False]
            )
        )

        if no_rag_is_a:

            answer_a = (
                no_rag_answer
            )

            answer_b = (
                rag_answer
            )

            judge_order = (
                "A=NO_RAG,B=RAG"
            )

        else:

            answer_a = (
                rag_answer
            )

            answer_b = (
                no_rag_answer
            )

            judge_order = (
                "A=RAG,B=NO_RAG"
            )

        prompt = (
            JUDGE_PROMPT.format(

                question=
                    question,

                reference=
                    reference,

                answer_a=
                    answer_a,

                answer_b=
                    answer_b,
            )
        )

        raw = (
            self.llm_call(
                prompt
            )
        )

        cleaned = (
            clean_json_response(
                raw
            )
        )

        try:

            verdict = (
                json.loads(
                    cleaned
                )
            )

        except json.JSONDecodeError as e:

            logger.error(
                "Judge response was not valid JSON:\n%s",
                raw,
            )

            raise ValueError(
                "Gemini judge returned invalid JSON."
            ) from e

        allowed = set(
            CLASSES
        )

        for key in [
            "answer_a",
            "answer_b",
        ]:

            if key not in verdict:

                raise ValueError(
                    f"Judge output missing {key}"
                )

            classification = str(
                verdict[
                    key
                ].get(
                    "classification",
                    "",
                )
            ).upper()

            if classification not in allowed:

                raise ValueError(
                    "Invalid judge classification: "
                    f"{classification}"
                )

            verdict[
                key
            ][
                "classification"
            ] = classification

        # --------------------------------------------------------------
        # Map blinded A/B results back to experimental conditions.
        # --------------------------------------------------------------

        if no_rag_is_a:

            no_rag_verdict = (
                verdict[
                    "answer_a"
                ]
            )

            rag_verdict = (
                verdict[
                    "answer_b"
                ]
            )

        else:

            rag_verdict = (
                verdict[
                    "answer_a"
                ]
            )

            no_rag_verdict = (
                verdict[
                    "answer_b"
                ]
            )

        return {

            "no_rag":
                no_rag_verdict,

            "rag":
                rag_verdict,

            "judge_order":
                judge_order,
        }


    # ----------------------------------------------------------------------
    # Evaluate one paired question
    # ----------------------------------------------------------------------

    def evaluate_pair(
        self,
        question: str,
        reference: str,
    ) -> Dict[str, Any]:

        t0 = time.time()

        # ==============================================================
        # A — NO RAG
        # ==============================================================

        logger.info(
            "Generating NO-RAG answer..."
        )

        no_rag_answer = (
            self.generate(
                question=
                    question,

                contexts=
                    None,
            )
        )


        # ==============================================================
        # B — RAG
        # ==============================================================

        logger.info(
            "Retrieving top-%d FAISS passages...",
            TOP_K,
        )

        docs = (
            self.retrieve(
                question
            )
        )

        contexts = [
            doc.page_content
            for doc in docs
        ]

        # --------------------------------------------------------------
        # Preserve metadata for failure analysis.
        # --------------------------------------------------------------

        retrieved_evidence = []

        for rank, doc in enumerate(
            docs,
            1,
        ):

            retrieved_evidence.append({

                "rank":
                    rank,

                "source":
                    doc.metadata.get(
                        "source"
                    ),

                "page":
                    doc.metadata.get(
                        "page"
                    ),

                "text":
                    doc.page_content,
            })

        logger.info(
            "Retrieved %d passages.",
            len(contexts),
        )

        logger.info(
            "Generating RAG answer..."
        )

        rag_answer = (
            self.generate(
                question=
                    question,

                contexts=
                    contexts,
            )
        )


        # ==============================================================
        # BLINDED JUDGE
        # ==============================================================

        logger.info(
            "Judging paired answers using randomized blinded order..."
        )

        verdict = (
            self.judge(

                question=
                    question,

                reference=
                    reference,

                no_rag_answer=
                    no_rag_answer,

                rag_answer=
                    rag_answer,
            )
        )

        latency = round(
            time.time()
            - t0,
            2,
        )


        return {

            "question":
                question,

            "reference":
                reference,

            "no_rag": {

                "answer":
                    no_rag_answer,

                "classification":
                    verdict[
                        "no_rag"
                    ][
                        "classification"
                    ],

                "reason":
                    verdict[
                        "no_rag"
                    ].get(
                        "reason",
                        "",
                    ),
            },

            "rag": {

                "answer":
                    rag_answer,

                "classification":
                    verdict[
                        "rag"
                    ][
                        "classification"
                    ],

                "reason":
                    verdict[
                        "rag"
                    ].get(
                        "reason",
                        "",
                    ),

                "retrieved_evidence":
                    retrieved_evidence,
            },

            "judge_order":
                verdict[
                    "judge_order"
                ],

            "latency_sec":
                latency,
        }


# ============================================================================
# TERMINAL SUMMARY
# ============================================================================

def print_summary(
    summary: Dict[str, Any]
):

    if not summary:

        print(
            "No valid results yet."
        )

        return

    no = (
        summary[
            "no_rag"
        ]
    )

    rag = (
        summary[
            "rag"
        ]
    )

    print()
    print("=" * 82)

    print(
        "RQ3 — CONTROLLED RAG VS NO-RAG HALLUCINATION EXPERIMENT"
    )

    print("=" * 82)

    print(
        f"Paired samples: "
        f"{summary['n_samples']}"
    )

    print()

    print(
        f"{'Metric':<30}"
        f"{'No-RAG':>14}"
        f"{'RAG':>14}"
    )

    print("-" * 58)

    for cls in CLASSES:

        print(
            f"{cls.title():<30}"
            f"{no['counts'][cls]:>14}"
            f"{rag['counts'][cls]:>14}"
        )

    print()

    print(
        f"No-RAG correct rate:       "
        f"{no['correct_rate'] * 100:.2f}%"
    )

    print(
        f"RAG correct rate:          "
        f"{rag['correct_rate'] * 100:.2f}%"
    )

    print()

    print(
        f"No-RAG partial rate:       "
        f"{no['partial_correct_rate'] * 100:.2f}%"
    )

    print(
        f"RAG partial rate:          "
        f"{rag['partial_correct_rate'] * 100:.2f}%"
    )

    print()

    print(
        f"No-RAG hallucination rate: "
        f"{no['hallucination_rate'] * 100:.2f}%"
    )

    print(
        f"RAG hallucination rate:    "
        f"{rag['hallucination_rate'] * 100:.2f}%"
    )

    print()

    print(
        f"No-RAG abstention rate:     "
        f"{no['abstention_rate'] * 100:.2f}%"
    )

    print(
        f"RAG abstention rate:        "
        f"{rag['abstention_rate'] * 100:.2f}%"
    )

    print()

    absolute = (
        summary[
            "hallucination_absolute_reduction"
        ]
    )

    print(
        "Absolute hallucination reduction: "
        f"{absolute * 100:.2f} percentage points"
    )

    relative = (
        summary[
            "hallucination_relative_reduction"
        ]
    )

    if relative is not None:

        print(
            "Relative hallucination reduction: "
            f"{relative * 100:.2f}%"
        )

    else:

        print(
            "Relative hallucination reduction: "
            "N/A"
        )

    print()

    print(
        "Correct-answer rate change: "
        f"{summary['correct_rate_absolute_change'] * 100:.2f} "
        "percentage points"
    )

    print()

    transitions = (
        summary.get(
            "paired_transitions",
            {}
        )
    )

    if transitions:

        print(
            "Selected paired transitions:"
        )

        important = [

            "HALLUCINATED_TO_CORRECT",
            "HALLUCINATED_TO_PARTIALLY_CORRECT",
            "HALLUCINATED_TO_ABSTAINED",

            "CORRECT_TO_HALLUCINATED",
            "PARTIALLY_CORRECT_TO_HALLUCINATED",

            "CORRECT_TO_CORRECT",
        ]

        for key in important:

            if key in transitions:

                print(
                    f"  {key}: "
                    f"{transitions[key]}"
                )

    print("=" * 82)

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()

    print("=" * 82)

    print(
        "RQ3 — RAG VS NO-RAG CONTROLLED EXPERIMENT"
    )

    print("=" * 82)

    print(
        "Independent variable: retrieved context supplied YES / NO"
    )

    print(
        "RAG retrieval: FAISS-only | K=10 | no reranker | original query"
    )

    print(
        f"Chunking: {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}"
    )

    print(
        f"Embedding: {config.EMBEDDING_MODEL}"
    )

    print(
        f"Generator/Judge: {config.GEMINI_MODEL}"
    )

    print(
        "Judge: blinded + randomized A/B order"
    )

    print("=" * 82)

    print()


    # ========================================================================
    # LOAD QUESTION POOL
    # ========================================================================

    pool = (
        load_question_pool()
    )


    # ========================================================================
    # LOAD EXISTING RESULTS
    # ========================================================================

    existing = (
        load_existing_results()
    )


    done_questions = {

        r["question"]

        for r
        in existing["results"]

        if (
            "error" not in r
            and r.get("question")
        )
    }


    remaining = [

        pair

        for pair in pool

        if (
            pair["question"]
            not in done_questions
        )
    ]


    logger.info(
        "%d/%d questions already completed; %d remaining.",
        len(done_questions),
        len(pool),
        len(remaining),
    )


    # ========================================================================
    # EXPERIMENT COMPLETE
    # ========================================================================

    if not remaining:

        logger.info(
            "Experiment already complete."
        )

        print_summary(
            existing.get(
                "summary",
                {},
            )
        )

        return


    # ========================================================================
    # DAILY BATCH
    # ========================================================================

    batch_size = max(
        1,
        DAILY_CALL_BUDGET
        // CALLS_PER_SAMPLE,
    )


    todays_batch = (
        remaining[
            :batch_size
        ]
    )


    logger.info(
        "Running %d paired samples (~%d Gemini calls).",
        len(todays_batch),
        len(todays_batch)
        * CALLS_PER_SAMPLE,
    )


    # ========================================================================
    # INITIALIZE EVALUATOR
    # ========================================================================

    evaluator = (
        RAGvsNoRAGEvaluator()
    )


    # ========================================================================
    # RUN EXPERIMENT
    # ========================================================================

    for i, pair in enumerate(
        todays_batch,
        1,
    ):

        question = (
            pair[
                "question"
            ]
        )

        reference = (
            pair[
                "reference"
            ]
        )


        logger.info(
            "[%d/%d] Q: %s",
            i,
            len(todays_batch),
            question[:90],
        )


        try:

            result = (
                evaluator.evaluate_pair(
                    question=
                        question,

                    reference=
                        reference,
                )
            )


            existing[
                "results"
            ].append(
                result
            )


            logger.info(
                "Result | No-RAG=%s | RAG=%s | Judge=%s",
                result[
                    "no_rag"
                ][
                    "classification"
                ],
                result[
                    "rag"
                ][
                    "classification"
                ],
                result[
                    "judge_order"
                ],
            )


        except Exception as e:

            logger.error(
                "Stopping safely because of API/evaluation error: %s",
                e,
            )

            break


        finally:

            # --------------------------------------------------------------
            # Recalculate valid results.
            # --------------------------------------------------------------

            valid = [

                r

                for r
                in existing[
                    "results"
                ]

                if "error"
                not in r
            ]


            # --------------------------------------------------------------
            # Recalculate summary.
            # --------------------------------------------------------------

            existing[
                "summary"
            ] = (
                compute_summary(
                    valid
                )
            )


            # --------------------------------------------------------------
            # Save exact experimental configuration.
            # --------------------------------------------------------------

            existing[
                "config"
            ] = {

                "experiment":
                    "rag_vs_no_rag_hallucination",

                "research_question":
                    "RQ3",

                "design":
                    "paired_controlled",

                "source_question_file":
                    str(
                        SOURCE_RESULTS_PATH
                    ),

                "chunk_size":
                    config.CHUNK_SIZE,

                "chunk_overlap":
                    config.CHUNK_OVERLAP,

                "embedding_model":
                    config.EMBEDDING_MODEL,

                "embedding_normalization":
                    True,

                "index_path":
                    config.INDEX_PATH,

                "generator_model":
                    config.GEMINI_MODEL,

                "judge_model":
                    config.GEMINI_MODEL,

                "temperature":
                    0,

                "retrieval":
                    "FAISS-only",

                "top_k":
                    TOP_K,

                "bm25_weight":
                    0.0,

                "faiss_weight":
                    1.0,

                "reranker":
                    "none",

                "query_strategy":
                    "original",

                "judge_blinded":
                    True,

                "judge_answer_order_randomized":
                    True,

                "random_seed":
                    RANDOM_SEED,

                "calls_per_sample":
                    CALLS_PER_SAMPLE,

                "min_seconds_between_calls":
                    MIN_SECONDS_BETWEEN_CALLS,

                "n_valid":
                    len(valid),
            }


            # --------------------------------------------------------------
            # Save immediately after each sample.
            # --------------------------------------------------------------

            save_results(
                existing
            )


    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print_summary(
        existing.get(
            "summary",
            {},
        )
    )


    print(
        f"JSON: {RESULTS_PATH}"
    )

    print(
        f"CSV:  {CSV_PATH}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
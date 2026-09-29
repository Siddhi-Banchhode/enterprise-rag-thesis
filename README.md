# Enterprise RAG System for Intelligent Document Querying

**MSc Thesis Project — Data Science – Business Analytics**  
**Author:** Siddhi Vishwanath Banchhode  
**Supervisor:** Prof. Dr. Philipp Unberath
**Institution:** SRH University of Applied Sciences

## Overview

This repository contains the implementation and evaluation framework developed for the MSc thesis:

> **Design and Empirical Evaluation of a Retrieval-Augmented Generation System for Intelligent Document Querying in Digital Enterprises**

The project investigates Retrieval-Augmented Generation (RAG) for natural-language querying of enterprise PDF documents.

The implemented system combines:

- PDF document ingestion and preprocessing
- configurable overlapping text chunking
- Sentence-BERT embeddings
- FAISS vector indexing and dense retrieval
- optional BM25 and hybrid retrieval
- optional cross-encoder reranking
- retrieval-grounded LLM generation
- Streamlit user interface
- FastAPI service
- Docker-based packaging
- empirical retrieval and generation evaluation

A central objective of the thesis is empirical configuration selection rather than architectural complexity.

The **final selected retrieval configuration uses dense-only FAISS retrieval without cross-encoder reranking**.

---

## System Architecture

The implementation follows two principal phases.

### 1. Offline Document Indexing

PDF documents are:

1. loaded at page level;
2. cleaned and normalized;
3. divided into overlapping text chunks;
4. encoded as dense Sentence-BERT vectors;
5. stored together with metadata in a FAISS vector index.

### 2. Online Query Processing

At query time:

1. the natural-language query is encoded into the embedding space;
2. relevant passages are retrieved;
3. retrieved evidence is assembled into the generation context;
4. the LLM generates an evidence-grounded response;
5. source metadata is retained for traceability.

```text
PDF Documents
      ↓
Page-Level Extraction
      ↓
Cleaning & Normalization
      ↓
Overlapping Chunking
      ↓
Sentence-BERT Embeddings
      ↓
FAISS Vector Index
      ↓
Natural-Language Query
      ↓
Dense Retrieval
      ↓
Retrieved Evidence
      ↓
Grounded Prompt
      ↓
Gemini 2.5 Flash
      ↓
Answer + Source Information
```

The architecture additionally supports BM25 retrieval, hybrid retrieval, query adaptation, and cross-encoder reranking for experimental purposes.

---

## Final Evaluated Configuration

| Component | Final Configuration |
|---|---|
| Chunk size | 2,000 characters |
| Chunk overlap | 200 characters |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| Embedding dimension | 768 |
| Embedding normalization | Enabled |
| Vector index | FAISS `IndexFlatL2` |
| Retrieval strategy | Dense-only FAISS |
| Retrieval depth | K = 10 |
| Query formulation | Original query |
| BM25 contribution | 0.0 |
| FAISS contribution | 1.0 |
| Cross-encoder reranker | Not used |
| Generation model | Gemini 2.5 Flash |
| Temperature | 0 |

The final enterprise-document corpus contains:

- **15 PDF documents**
- **1,842 source pages**
- **1,839 retained pages after preprocessing**
- **4,278 chunks in the final index**

> **Important:** The repository contains experimental retrieval components and historical evaluation scripts. These should not be interpreted as components of the final selected retrieval configuration.

---

## Technology Stack

- **Python** — core implementation
- **LangChain** — RAG and retrieval orchestration
- **PyPDFLoader** — page-level PDF extraction
- **RecursiveCharacterTextSplitter** — overlapping text segmentation
- **Sentence Transformers / `all-mpnet-base-v2`** — dense embeddings
- **FAISS / `IndexFlatL2`** — dense-vector retrieval
- **BM25Retriever** — experimental sparse retrieval
- **EnsembleRetriever** — experimental sparse-dense fusion
- **`BAAI/bge-reranker-base`** — optional experimental reranker
- **Gemini 2.5 Flash** — final generation model
- **FastAPI** — programmatic interface
- **Streamlit** — interactive document-querying interface
- **Docker / Docker Compose** — application packaging

---

## Repository Structure

```text
enterprise-rag/
│
├── api/
│   └── fastapi_server.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── eval/                    # Experimental evaluation outputs
│
├── evaluation/                  # Evaluation and ablation scripts
│
├── scripts/
│   └── download_datasets.py
│
├── src/
│   ├── embeddings/
│   │   └── embedder.py
│   ├── ingestion/
│   │   └── pdf_loader.py
│   ├── preprocessing/
│   │   ├── chunker.py
│   │   └── cleaner.py
│   ├── rag/
│   │   ├── prompt_builder.py
│   │   └── rag_chain.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── retriever.py
│   └── vectorstore/
│       └── faiss_store.py
│
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── config.py
├── requirements_main_rag.txt
├── requirements_eval.txt
├── test_pipeline.py
└── README.md
```

---

## Data Excluded from GitHub

Large, external, private, or generated resources are intentionally excluded from the repository through `.gitignore`.

```text
data/datasets/
data/pdfs/
data/faiss_index/
data/faiss_index_backup_500_50/
legalbench_data/
logs/
venv/
venv_eval/
.env
```

The enterprise PDF corpus and benchmark datasets are therefore **not distributed through this repository**.

Selected experimental outputs under `data/eval/` are retained to provide traceability between the implementation and thesis evaluation.

---

## API-Key Configuration

API credentials are **not stored in this repository**.

The implementation reads credentials from environment variables.

Create a local `.env` file in the project root.

### Gemini

```env
GOOGLE_API_KEY=your_google_api_key
LLM_TYPE=gemini
```

### OpenAI Backend

```env
OPENAI_API_KEY=your_openai_api_key
LLM_TYPE=openai
```

Never commit `.env` or hard-code API credentials into source files.

---

## Dependencies

The project contains separate dependency specifications:

```text
requirements_main_rag.txt
requirements_eval.txt
```

The primary application entry points are:

```text
app/streamlit_app.py
api/fastapi_server.py
```

Docker-related configuration is provided through:

```text
Dockerfile
docker-compose.yml
docker-entrypoint.sh
```

The repository also contains `test_pipeline.py` for pipeline-level testing.

---

## Evaluation Framework

The empirical evaluation contains several complementary components.

### 1. Retrieval Configuration Search

Controlled ablation experiments investigate:

- chunk size and overlap
- candidate retrieval depth
- BM25/FAISS weighting
- embedding-model selection
- query adaptation
- cross-encoder reranking

Most configuration-search experiments use a fixed **50-query CUAD screening subset**.

The selected configuration is subsequently evaluated at full benchmark scale.

### 2. Full-Scale LegalBench-RAG / CUAD Evaluation

The principal retrieval evaluation uses **4,042 CUAD queries**.

Metrics include:

- Precision@K
- Recall@K
- MAP@10

The final configuration improved full-scale CUAD:

**MAP@10: 0.102 → 0.210**

This corresponds to:

- **Absolute improvement:** 0.108
- **Relative improvement:** approximately 105.9%

The experiments also showed that additional architectural complexity did not automatically improve retrieval performance.

Under the evaluated conditions, dense-only retrieval was selected over the tested hybrid configurations, while query adaptation and cross-encoder reranking did not improve the principal ranking metric.

### 3. Generation-Quality Evaluation

A custom **RAGAS-style evaluation** measures:

- Faithfulness
- Answer Relevancy

Across **23 successful evaluation samples**:

| Metric | Score |
|---|---:|
| RAGAS-style Faithfulness | 0.5565 |
| RAGAS-style Answer Relevancy | 0.5148 |

### 4. RAG vs. No-RAG Evaluation

The same questions and Gemini 2.5 Flash generator were evaluated with and without retrieved evidence.

| Outcome | No-RAG | RAG |
|---|---:|---:|
| Correct | 47.83% | 30.43% |
| Hallucinated | 47.83% | 39.13% |
| Abstained | 4.35% | 30.43% |

Observed hallucination decreased by **8.70 percentage points**, while correct-answer frequency decreased and abstention increased.

The paired difference was not statistically significant in the evaluated sample:

**McNemar p ≈ 0.7266**

Therefore, these results should **not** be interpreted as evidence that RAG universally improves factual accuracy.

### 5. Cross-Domain Evaluation

| Dataset | Principal Result |
|---|---|
| pdfQA | F1@K = 0.3261 |
| SQuAD2 | F1@K = 0.4459; Hit@1 = 0.8000 |
| FinanceBench | Metric limitations observed for numerical/derived answers |

---

## Evaluation Scripts

The `evaluation/` directory contains both final evaluation code and scripts created during configuration search.

Examples include:

```text
ablation_study.py
run_embedding_ablation.py
run_query_adaptation_ablation.py
run_reranker_evaluation.py
run_legalbench_rag_evaluation.py
run_rag_vs_no_rag_hallucination.py
ragas_eval.py
ragas_eval_resumable.py
```

Files containing names such as:

```text
500_50
backup
reranked
```

represent particular experimental stages or alternative configurations.

They should **not be interpreted as the final deployed retrieval configuration**.

The authoritative final evaluated configuration is documented above.

---

## Evaluation Outputs

Selected experimental results are retained under:

```text
data/eval/
```

These include:

- ablation results
- retrieval evaluation outputs
- evaluation summaries
- RAGAS-style results
- RAG-vs.-No-RAG hallucination results

---

## Dataset Availability

Large benchmark datasets and the enterprise PDF corpus are intentionally not stored in this GitHub repository.

The project uses or evaluates data from:

- LegalBench-RAG / CUAD
- pdfQA
- SQuAD2
- FinanceBench
- a thesis-specific 15-document enterprise PDF corpus

The repository contains:

```text
scripts/download_datasets.py
```

for dataset preparation supported by the implementation.

Users reproducing the experiments are responsible for obtaining the relevant datasets from their original sources and complying with their respective licenses and usage conditions.

---

## Key Findings

1. **Retrieval configuration materially affects performance.**
2. **Architectural complexity does not guarantee improvement.**
3. **Retrieval remains an important bottleneck.**
4. **Grounding and correctness are distinct properties.**
5. **RAG evaluation should be multidimensional.**
6. **Document structure affects retrieval performance.**

---

## Scope and Limitations

This project is a **research prototype** focused on text-based enterprise PDF question answering.

Important limitations include:

- principal retrieval evaluation is concentrated on the legal-domain CUAD benchmark;
- most configuration-search experiments use a 50-query screening subset;
- generation evaluation contains 23 successful samples;
- LLM-based judging can introduce evaluation bias;
- linearized PDF extraction is limited for tables, complex layouts, and numerical relationships;
- external LLM services introduce reproducibility constraints related to API availability, quotas, latency, and provider-side changes.

The system is **not presented as a production-ready enterprise platform**.

---

## Future Work

Potential extensions include:

- layout-aware and multimodal PDF processing
- metadata-aware and source-constrained retrieval
- improved table and numerical reasoning
- adaptive retrieval
- confidence-aware abstention
- larger generation evaluations
- stronger source isolation
- broader enterprise-domain evaluation
- production-oriented security and scalability

---

## Academic Context

This repository accompanies an MSc thesis and is intended to support transparency and reproducibility of the implementation and experimental workflow.

Experimental results should be interpreted within the datasets, configurations, metrics, and sample sizes described in the thesis.

**Siddhi Vishwanath Banchhode. _Design and Empirical Evaluation of a Retrieval-Augmented Generation System for Intelligent Document Querying in Digital Enterprises._ MSc Thesis, SRH University of Applied Sciences.**

---

## Author

**Siddhi Vishwanath Banchhode**  
MSc Data Science – Business Analytics  
SRH University of Applied Sciences

---

## License

No software license is currently specified for this repository.

Unless a license is added, the repository should not be assumed to grant permission for reuse, modification, or redistribution.
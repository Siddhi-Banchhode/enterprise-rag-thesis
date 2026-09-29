# ── Dockerfile — Enterprise RAG System ────────────────────────────────────────
# Thesis: Design and Empirical Evaluation of a RAG System
#         for Intelligent Document Querying in Digital Enterprises
#
# Build:
#   docker build -t enterprise-rag .
#
# Run (Streamlit UI):
#   docker run -p 8501:8501 --env-file .env enterprise-rag streamlit
#
# Run (FastAPI):
#   docker run -p 8000:8000 --env-file .env enterprise-rag api
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── system dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── copy requirements first (layer caching) ───────────────────────────────────
COPY requirements_main_rag.txt ./requirements.txt

# ── install Python dependencies ───────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── copy project files ────────────────────────────────────────────────────────
COPY config.py        ./
COPY src/             ./src/
COPY app/             ./app/
COPY api/             ./api/
COPY evaluation/      ./evaluation/
COPY data/pdfs/       ./data/pdfs/
COPY data/faiss_index/ ./data/faiss_index/

# ── environment ───────────────────────────────────────────────────────────────
# These are overridden at runtime via --env-file .env
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LLM_TYPE=gemini
ENV PDF_DIR=data/pdfs
ENV INDEX_PATH=data/faiss_index
ENV DATASETS_DIR=data/datasets
ENV CHUNK_SIZE=500
ENV CHUNK_OVERLAP=50
ENV TOP_K=5

# ── expose ports ──────────────────────────────────────────────────────────────
# 8501 = Streamlit UI
# 8000 = FastAPI
EXPOSE 8501 8000

# ── entrypoint script ─────────────────────────────────────────────────────────
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["streamlit"]

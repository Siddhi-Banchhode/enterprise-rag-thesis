"""
api/fastapi_server.py
======================
FastAPI REST API — Enterprise RAG System

Exposes the full RAG pipeline as a REST API:
  POST /query       — ask a question, get an answer with sources
  GET  /health      — health check
  GET  /corpus      — corpus statistics
  POST /upload      — upload a PDF to augment the corpus (session-scoped)

Run from project root:
    uvicorn api.fastapi_server:app --host 0.0.0.0 --port 8000 --reload

Test with curl:
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"question": "What was Apple total net sales in FY2022?"}'
"""

import sys
import os
import tempfile
import logging
import time
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── app init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "Intelligent Document Q&A powered by Hybrid Retrieval + "
        "Cross-Encoder Reranking + Gemini 2.5 Flash. "
        "MSc Thesis — SRH Wilhelm Löhe University of Applied Sciences."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    top_n: int = 3
    use_reranker: bool = True

class SourceDocument(BaseModel):
    filename: str
    page: int
    snippet: str
    rerank_score: Optional[float] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    latency_sec: float
    num_candidates: int
    num_reranked: int

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    corpus_chunks: int
    uploaded_chunks: int
    llm_type: str

class CorpusResponse(BaseModel):
    base_pdfs: int
    base_chunks: int
    uploaded_files: List[str]
    uploaded_chunks: int
    total_chunks: int


# ── pipeline state (module-level, loaded once at startup) ─────────────────────

class PipelineState:
    embedder        = None
    vectorstore     = None
    base_chunks     = None
    reranker        = None
    extra_chunks    = []
    uploaded_files  = []
    ready           = False

state = PipelineState()


@app.on_event("startup")
async def startup_event():
    """Load all pipeline components once at server startup."""
    logger.info("Loading RAG pipeline …")

    from config import config
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.vectorstore.faiss_store import load_index
    from src.ingestion.pdf_loader import load_pdfs
    from src.preprocessing.cleaner import preprocess_documents as clean_documents
    from src.preprocessing.chunker import chunk_documents
    from src.retrieval.reranker import Reranker

    state.embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded.")

    state.vectorstore = load_index(state.embedder)
    logger.info("FAISS index loaded: %d vectors", state.vectorstore.index.ntotal)

    raw              = load_pdfs()
    from src.preprocessing.cleaner import preprocess_documents
    clean            = preprocess_documents(raw)
    state.base_chunks = chunk_documents(clean)
    logger.info("Base chunks: %d", len(state.base_chunks))

    state.reranker = Reranker()
    logger.info("Reranker loaded.")

    state.ready = True
    logger.info("Pipeline ready.")


# ── helpers ───────────────────────────────────────────────────────────────────

def get_retriever(top_k: int):
    from src.retrieval.hybrid_retriever import build_hybrid_retriever
    all_chunks = state.base_chunks + state.extra_chunks

    if state.extra_chunks:
        new_texts = [c.page_content for c in state.extra_chunks]
        state.vectorstore.add_texts(
            new_texts,
            metadatas=[c.metadata for c in state.extra_chunks],
        )

    return build_hybrid_retriever(all_chunks, state.vectorstore, top_k=top_k)


def make_static_retriever(docs):
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from typing import List as TList

    class StaticRetriever(BaseRetriever):
        docs: TList[Document]

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> TList[Document]:
            return self.docs

    return StaticRetriever(docs=docs)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms pipeline is loaded and ready."""
    from config import config
    return HealthResponse(
        status="ok" if state.ready else "loading",
        pipeline_ready=state.ready,
        corpus_chunks=len(state.base_chunks) if state.base_chunks else 0,
        uploaded_chunks=len(state.extra_chunks),
        llm_type=config.LLM_TYPE,
    )


@app.get("/corpus", response_model=CorpusResponse)
async def corpus_info():
    """Return corpus statistics."""
    return CorpusResponse(
        base_pdfs=15,
        base_chunks=len(state.base_chunks) if state.base_chunks else 0,
        uploaded_files=state.uploaded_files,
        uploaded_chunks=len(state.extra_chunks),
        total_chunks=(len(state.base_chunks) if state.base_chunks else 0)
                     + len(state.extra_chunks),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question over the enterprise document corpus.

    Returns the generated answer, source documents, and latency.
    """
    if not state.ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()

    try:
        # 1. Hybrid retrieval
        hybrid     = get_retriever(request.top_k)
        candidates = hybrid.invoke(request.question)

        # 2. Reranking
        if request.use_reranker and candidates:
            top_docs = state.reranker.rerank(
                request.question, candidates
            )[:request.top_n]
        else:
            top_docs = candidates[:request.top_n]

        # 3. RAG generation
        from src.rag.rag_chain import build_rag_chain
        chain  = build_rag_chain(make_static_retriever(top_docs))
        result = chain.invoke({"query": request.question})
        answer = result.get("result", "No answer generated.")

        # 4. Format sources
        sources = []
        seen = set()
        for doc in top_docs:
            fname = doc.metadata.get("filename") or os.path.basename(
                doc.metadata.get("source", "unknown")
            )
            page  = doc.metadata.get("page", 0)
            key   = f"{fname}_p{page}"
            if key in seen:
                continue
            seen.add(key)
            pg = page + 1 if isinstance(page, int) else page
            sources.append(SourceDocument(
                filename=fname,
                page=pg,
                snippet=doc.page_content[:200].replace("\n", " "),
                rerank_score=doc.metadata.get("rerank_score"),
            ))

        latency = round(time.time() - t0, 2)
        logger.info(
            "Query answered in %.2fs | candidates=%d | reranked=%d",
            latency, len(candidates), len(top_docs),
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            latency_sec=latency,
            num_candidates=len(candidates),
            num_reranked=len(top_docs),
        )

    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF to augment the corpus for this session.

    The uploaded document is processed through the full ingestion pipeline
    and its chunks are added to the in-memory index.
    """
    if not state.ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if file.filename in state.uploaded_files:
        return {"message": f"{file.filename} already uploaded.", "chunks_added": 0}

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Process through pipeline
        from langchain_community.document_loaders import PyPDFLoader
        from src.preprocessing.cleaner import preprocess_documents
        from src.preprocessing.chunker import chunk_documents

        loader   = PyPDFLoader(tmp_path)
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source"]   = file.filename
            doc.metadata["filename"] = file.filename

        clean  = preprocess_documents(raw_docs)
        chunks = chunk_documents(clean)

        state.extra_chunks.extend(chunks)
        state.uploaded_files.append(file.filename)
        os.unlink(tmp_path)

        logger.info("Uploaded %s → %d chunks added", file.filename, len(chunks))

        return {
            "message":      f"Successfully processed {file.filename}",
            "chunks_added": len(chunks),
            "total_chunks": len(state.base_chunks) + len(state.extra_chunks),
        }

    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
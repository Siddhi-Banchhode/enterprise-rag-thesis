from config import config, validate_config
from src.ingestion.pdf_loader import load_pdfs
from src.preprocessing.cleaner import preprocess_documents
from src.preprocessing.chunker import chunk_documents
from src.embeddings.embedder import get_embeddings
from src.vectorstore.faiss_store import build_index, load_index, index_exists
from src.retrieval.retriever import get_retriever
from src.rag.rag_chain import build_rag_chain

print("Validating config...")
validate_config()

print("\nLoading PDFs...")
docs = load_pdfs()

if not docs:
    from langchain.schema import Document
    docs = [Document(
        page_content="Full-time employees are entitled to 25 days paid "
                     "annual leave per year. Leave must be approved by "
                     "line managers two weeks in advance.",
        metadata={"source": "demo.pdf", "page": 0, "filename": "demo.pdf"}
    )]
    print("Using demo document (no PDFs found in data/pdfs/)")

print("\nPreprocessing...")
docs = preprocess_documents(docs)

print("\nChunking...")
chunks = chunk_documents(docs)

print("\nEmbedding...")
embedder = get_embeddings()

print("\nBuilding/loading index...")
vs = load_index(embedder) if index_exists() else build_index(chunks, embedder)

print("\nBuilding RAG chain...")
retriever = get_retriever(vs)
chain = build_rag_chain(retriever)

print("\nRunning query...")
question = "What is the annual leave entitlement?"
result = chain.invoke({"query": question})

print(f"\n{'='*50}")
print(f"QUESTION: {question}")
print(f"\nANSWER: {result['result']}")
print(f"\nSOURCES:")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata.get('filename')} "
          f"(page {doc.metadata.get('page')})")
print(f"{'='*50}")
print("\n✅ Pipeline working end-to-end!")
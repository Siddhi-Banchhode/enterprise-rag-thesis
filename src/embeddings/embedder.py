# src/embeddings/embedder.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config
from langchain_huggingface import HuggingFaceEmbeddings

_embedder = None

def get_embeddings():
    global _embedder
    if _embedder is not None:
        return _embedder

    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    print("(First run downloads ~420MB — takes a few minutes)")

    _embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded")
    return _embedder


if __name__ == "__main__":
    embedder = get_embeddings()
    vec = embedder.embed_query("What is the annual revenue?")
    print(f"Vector dims  : {len(vec)}")
    print(f"First 5 vals : {[round(x,4) for x in vec[:5]]}")
    print(f"Magnitude    : {sum(x**2 for x in vec)**0.5:.4f}  (should be ~1.0)")
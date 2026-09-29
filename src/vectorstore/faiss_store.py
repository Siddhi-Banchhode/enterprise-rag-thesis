import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config
from langchain_community.vectorstores import FAISS


def build_index(chunks, embeddings):
    print(f"Building FAISS index from {len(chunks)} chunks...")
    vs = FAISS.from_documents(chunks, embeddings)
    Path(config.INDEX_PATH).mkdir(parents=True, exist_ok=True)
    vs.save_local(config.INDEX_PATH)
    print(f"✅ Index saved to {config.INDEX_PATH}")
    return vs


def load_index(embeddings):
    if not Path(config.INDEX_PATH, "index.faiss").exists():
        raise FileNotFoundError("No index found — run build_index() first")
    vs = FAISS.load_local(config.INDEX_PATH, embeddings,
                          allow_dangerous_deserialization=True)
    print(f"✅ Index loaded: {vs.index.ntotal} vectors")
    return vs


def index_exists():
    return Path(config.INDEX_PATH, "index.faiss").exists()

if __name__ == "__main__":
    print("FAISS store module loaded successfully")
    exists = index_exists()
    print("Index exists:", exists)
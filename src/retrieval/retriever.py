# src/retrieval/retriever.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config


def get_retriever(vectorstore, k=None):
    k = k or config.TOP_K
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": config.SIMILARITY_THRESHOLD
        }
    )
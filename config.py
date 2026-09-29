import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY", "")
    PDF_DIR              = os.getenv("PDF_DIR", "data/pdfs")
    INDEX_PATH           = os.getenv("INDEX_PATH", "data/faiss_index")
    DATASETS_DIR         = os.getenv("DATASETS_DIR", "data/datasets")
    CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", 2000))
    CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_MODEL      = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DEVICE     = "cpu"
    EMBEDDING_DIM        = 768
    TOP_K                = 5
    SIMILARITY_THRESHOLD = 0.30
    LLM_TYPE             = os.getenv("LLM_TYPE", "gemini")
    OPENAI_MODEL         = "gpt-4o-mini"
    GEMINI_MODEL         = "gemini-2.5-flash"
    LLM_TEMPERATURE      = 0

config = Config()

def validate_config():
    issues = []
    if config.LLM_TYPE == "openai" and not config.OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY missing in .env")
    if config.LLM_TYPE == "gemini" and not config.GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY missing in .env")
    os.makedirs(config.PDF_DIR, exist_ok=True)
    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    if issues:
        for i in issues: print(f"⚠ {i}")
    else:
        print("✅ Config validated")
    return len(issues) == 0

if __name__ == "__main__":
    validate_config()
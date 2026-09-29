import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_dir=None):
    if pdf_dir is None:
        pdf_dir = config.PDF_DIR
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠ No PDFs in {pdf_dir}")
        return []

    print(f"Loading {len(pdf_files)} PDFs...")
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for p in pages:
                if len(p.page_content.strip()) > 30:
                    p.metadata["filename"] = pdf_path.name
                    all_docs.append(p)
            print(f"  ✓ {pdf_path.name} — {len(pages)} pages")
        except Exception as e:
            print(f"  ✗ {pdf_path.name}: {e}")

    print(f"✅ Total: {len(all_docs)} pages loaded")
    return all_docs


if __name__ == "__main__":
    docs = load_pdfs()
    if docs:
        print(f"\nSample: {docs[0].page_content[:200]}")
        print(f"Metadata: {docs[0].metadata}")
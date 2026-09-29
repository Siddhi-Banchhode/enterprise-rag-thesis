# src/preprocessing/chunker.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=None, chunk_overlap=None):
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    print(f"✅ Created {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


if __name__ == "__main__":
    from langchain.schema import Document
    test_doc = Document(
        page_content="This is a test. " * 50,
        metadata={"source": "test.pdf"}
    )
    chunks = chunk_documents([test_doc])
    print(f"First chunk: {chunks[0].page_content[:100]}")
import re
from langchain.schema import Document


def clean_text(text):
    if not text: return ""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\f", "\n", text)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text.strip()


def preprocess_documents(documents):
    cleaned = []
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        if len(doc.page_content) > 50:
            cleaned.append(doc)
    print(f"✅ Cleaned: {len(cleaned)}/{len(documents)} documents kept")
    return cleaned


if __name__ == "__main__":
    test = Document(page_content="Test   text  with-\nhyphens.", metadata={})
    print(clean_text(test.page_content))
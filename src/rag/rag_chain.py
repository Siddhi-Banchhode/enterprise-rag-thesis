# src/rag/rag_chain.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import config
from src.rag.prompt_builder import get_rag_prompt
from langchain.chains import RetrievalQA


def build_rag_chain(retriever, llm_type=None):
    llm_type = llm_type or config.LLM_TYPE
    llm = _load_llm(llm_type)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_rag_prompt()}
    )
    print(f"✅ RAG chain built with {llm_type.upper()} LLM")
    return chain


def _load_llm(llm_type):
    if llm_type == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )

    elif llm_type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=config.LLM_TEMPERATURE,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif llm_type == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model="llama3")

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}. Choose from: openai, gemini, ollama")


def format_answer(result):
    answer  = result.get("result", "No answer generated")
    sources = result.get("source_documents", [])
    output  = f"\n{'='*55}\n"
    output += f"ANSWER:\n{answer}\n"
    output += f"\n{'─'*55}\n"
    output += "SOURCES:\n"
    seen = set()
    for doc in sources:
        filename = doc.metadata.get("filename", "unknown")
        page     = doc.metadata.get("page", "?")
        key      = f"{filename}_p{page}"
        if key not in seen:
            seen.add(key)
            pg = page + 1 if isinstance(page, int) else page
            output += f"  • {filename} — Page {pg}\n"
            output += f"    {doc.page_content[:120]}...\n\n"
    output += f"{'='*55}\n"
    return output
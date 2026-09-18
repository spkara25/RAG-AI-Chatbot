"""
rag_engine.py
--------------
Core RAG (Retrieval-Augmented Generation) pipeline.

Pipeline stages implemented here:
  1. Load documents (PDF / TXT / DOCX)
  2. Split into chunks
  3. Generate embeddings (LOCAL, free — sentence-transformers)
  4. Store in a FAISS vector store
  5. Retrieve relevant chunks for a question
  6. Generate an answer using an LLM, constrained to the retrieved context
  7. Return the answer + source citations

Design choice for cost control
-------------------------------
Embeddings are computed locally with `sentence-transformers/all-MiniLM-L6-v2`
(runs on CPU, no API key, no cost, no rate limits). This is normally the most
API-call-heavy part of a RAG pipeline (one call per chunk), so doing it for
free is what keeps this app cheap to run repeatedly.

Only the final answer-generation step calls an LLM API, and only ONCE per
question (not once per chunk), using a cheap/fast model by default
(gpt-4o-mini for OpenAI, or gemini-1.5-flash for Gemini — both inexpensive,
and Gemini has a free tier).
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
try:
    # Modern langchain (0.1+) ships this as its own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older langchain versions where it still lived here
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class SourceChunk:
    """A single retrieved chunk, kept lightweight for display in the UI."""
    source: str
    page: Optional[int]
    text: str


@dataclass
class RAGAnswer:
    answer: str
    sources: List[SourceChunk]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str) -> List[Document]:
    """Load a single file into a list of LangChain Document objects."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    loader = loader_cls(file_path)
    return loader.load()


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load multiple files, tagging each Document with its source filename."""
    all_docs: List[Document] = []
    for path in file_paths:
        docs = load_document(path)
        filename = os.path.basename(path)
        for d in docs:
            d.metadata["source"] = filename
        all_docs.extend(docs)
    return all_docs


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into overlapping chunks suited for embedding/retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# --------------------------------------------------------------------------
# Embeddings + Vector store
# --------------------------------------------------------------------------

_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings_singleton = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model once and reuse it (avoids re-downloading)."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_singleton


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Embed all chunks locally and build an in-memory FAISS index."""
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore: FAISS, path: str) -> None:
    vectorstore.save_local(path)


def load_vectorstore(path: str) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# --------------------------------------------------------------------------
# LLM setup (only place that calls a paid/hosted API, and only for the
# final answer — not for embeddings)
# --------------------------------------------------------------------------

def get_llm(provider: str, api_key: str, model: Optional[str] = None):
    """
    Return a LangChain chat model for the chosen provider.
    provider: "openai" or "gemini"
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            model=model or "gpt-4o-mini",   # cheap + fast
            temperature=0,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            # Pinned to a specific stable (non-preview) model rather than a
            # moving alias, for reproducible behavior. NOTE: "gemini-1.5-flash"
            # has been fully retired by Google (calls now 404) — do not use it.
            # "gemini-2.5-flash" is the current stable Flash model as of this
            # writing. If Google retires it later, pass an explicit `model`
            # (e.g. "gemini-3.8-flash") to override this default.
            model=model or "gemini-2.5-flash",
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful knowledge assistant. Answer the user's \
question using ONLY the information in the "Context" section below, which \
was retrieved from the user's own documents.

Rules:
- If the answer is not contained in the context, say clearly that the \
documents don't contain that information. Do NOT make anything up.
- Be concise and directly answer the question first, then add brief \
supporting detail if useful.
- If different chunks disagree, point that out rather than picking one \
silently.
- Do not mention "the context" or "chunks" explicitly to the user; just \
answer naturally as if you'd read the documents.

Context:
{context}
"""


def _format_context(chunks: List[Document]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        src = c.metadata.get("source", "unknown")
        page = c.metadata.get("page")
        page_str = f", page {page + 1}" if isinstance(page, int) else ""
        parts.append(f"[Excerpt {i} — {src}{page_str}]\n{c.page_content}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# End-to-end query
# --------------------------------------------------------------------------

def _extract_text(response) -> str:
    """
    Safely extract plain text from an LLM response, regardless of which
    shape the underlying SDK/LangChain version hands back:
      - AIMessage.content as a plain string (the common case)
      - AIMessage.content as a list of content-part dicts/objects, e.g.
        [{"type": "text", "text": "..."}] — seen with some Gemini responses
      - An object exposing a direct `.text` attribute
      - Anything else -> fall back to str(response)
    Never raises AttributeError/KeyError; returns "" if nothing usable found.
    """
    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # covers {"type": "text", "text": "..."} and similar shapes
                text_piece = item.get("text") or item.get("content") or ""
                if text_piece:
                    parts.append(text_piece)
            else:
                text_piece = getattr(item, "text", None)
                if text_piece:
                    parts.append(text_piece)
        if parts:
            return "".join(parts)

    # Some SDKs expose .text directly on the response itself
    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text:
        return direct_text

    return str(response) if response is not None else ""


def answer_question(
    vectorstore: FAISS,
    question: str,
    llm,
    k: int = 4,
    chat_history: Optional[List[dict]] = None,
) -> RAGAnswer:
    """
    Retrieve top-k relevant chunks and ask the LLM to answer using only them.
    chat_history: optional list of {"role": "user"/"assistant", "content": str}
                  used to give the LLM conversational context (not re-retrieved).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)

    context_str = _format_context(retrieved_docs)
    system_msg = SYSTEM_PROMPT.format(context=context_str)

    messages = [{"role": "system", "content": system_msg}]
    if chat_history:
        # keep last few turns only, to control token usage
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)
    answer_text = _extract_text(response)

    sources = [
        SourceChunk(
            source=d.metadata.get("source", "unknown"),
            page=(d.metadata.get("page") + 1) if isinstance(d.metadata.get("page"), int) else None,
            text=d.page_content[:300].strip() + ("..." if len(d.page_content) > 300 else ""),
        )
        for d in retrieved_docs
    ]

    return RAGAnswer(answer=answer_text, sources=sources)

import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document


@dataclass
class SourceChunk:
    """A single retrieved chunk, kept lightweight for display in the UI."""
    source: str
    page: Optional[int]
    text: str
    score: Optional[float] = None  # 0-1 relevance score, when available


@dataclass
class RAGAnswer:
    answer: str
    sources: List[SourceChunk]
    confidence: Optional[str] = None  # "high" / "medium" / "low", or None if unscored

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
            model=model or "gemini-2.5-flash",
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

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


def retrieve_with_scores(
    vectorstore: FAISS,
    question: str,
    k: int = 4,
    search_type: str = "similarity",
    score_threshold: float = 0.0,
):
    """
    Retrieve chunks for a question, with a relevance score attached to each
    when the search type supports it. This is the "improved retrieval"
    layer: it lets the UI show *how* relevant each source actually is, and
    lets low-relevance noise be filtered out instead of always forcing k
    chunks into the prompt regardless of quality.

    search_type:
      - "similarity": plain nearest-neighbor search, with a 0-1 cosine
        similarity score per chunk, computed directly from FAISS's raw
        distance (see note below) rather than via LangChain's built-in
        relevance-score helper, whose default normalization for FAISS's
        L2 index is not reliably calibrated to a clean 0-1 range and was
        producing scores that were almost always low regardless of query.
      - "mmr": Maximal Marginal Relevance — trades a little top-1 relevance
        for less redundant, more diverse chunks (useful for broad/vague
        questions). Scores aren't available for MMR, so score=None.

    Returns: list of (Document, Optional[float] score) tuples, already
    filtered by score_threshold when scores are available.
    """
    if search_type == "mmr":
        docs = vectorstore.max_marginal_relevance_search(question, k=k, fetch_k=max(k * 4, 20))
        return [(d, None) for d in docs]

    try:
        # FAISS's default index here uses squared L2 (Euclidean) distance.
        # Because our embeddings are L2-normalized (encode_kwargs=
        # {"normalize_embeddings": True} in get_embeddings), there is an
        # exact closed-form relationship between that distance and cosine
        # similarity for unit vectors:
        #     ||a - b||^2 = 2 - 2*cos_sim(a, b)   =>   cos_sim = 1 - d^2/2
        # This gives a correctly calibrated 0-1-ish score without relying
        # on LangChain's distance-strategy-dependent approximation.
        raw = vectorstore.similarity_search_with_score(question, k=k)
    except Exception:
        # Fall back to plain similarity search with no score if the
        # vectorstore doesn't support returning distances for some reason.
        return [(d, None) for d in vectorstore.similarity_search(question, k=k)]

    scored = []
    for doc, distance in raw:
        cos_sim = 1.0 - (float(distance) ** 2) / 2.0
        cos_sim = max(0.0, min(1.0, cos_sim))  # clamp for float noise
        scored.append((doc, cos_sim))

    if score_threshold > 0:
        filtered = [(d, s) for d, s in scored if s >= score_threshold]
        # Never return zero results just because everything was below
        # threshold — better to answer with the best-available (if weak)
        # context than to silently give nothing back.
        return filtered if filtered else scored[:1]

    return scored


def answer_question(
    vectorstore: FAISS,
    question: str,
    llm,
    k: int = 4,
    chat_history: Optional[List[dict]] = None,
    search_type: str = "similarity",
    score_threshold: float = 0.0,
) -> RAGAnswer:
    """
    Retrieve top-k relevant chunks and ask the LLM to answer using only them.
    chat_history: optional list of {"role": "user"/"assistant", "content": str}
                  used to give the LLM conversational context (not re-retrieved).
    """
    scored_docs = retrieve_with_scores(
        vectorstore, question, k=k, search_type=search_type, score_threshold=score_threshold
    )
    retrieved_docs = [d for d, _ in scored_docs]

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
            score=round(float(score), 3) if score is not None else None,
        )
        for d, score in scored_docs
    ]

    # Lightweight retrieval-quality signal (no extra LLM call): a low top
    # relevance score usually means the documents don't actually cover the
    # question, which is worth surfacing to the user even if the model
    # produced a fluent-sounding answer anyway.
    scores_available = [s.score for s in sources if s.score is not None]
    confidence = None
    if scores_available:
        top_score = max(scores_available)
        if top_score >= 0.5:
            confidence = "high"
        elif top_score >= 0.3:
            confidence = "medium"
        else:
            confidence = "low"

    return RAGAnswer(answer=answer_text, sources=sources, confidence=confidence)
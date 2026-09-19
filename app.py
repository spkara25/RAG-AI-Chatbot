"""
app.py
------
Streamlit front-end for the AI Knowledge Assistant.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import hashlib
import logging

# Streamlit's hot-reload file watcher probes every submodule of `transformers`
# (including vision ones that need `torchvision`, which we don't use here) and
# logs a harmless traceback for each. This just silences that specific logger.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st

from rag_engine import (
    load_documents,
    split_documents,
    build_vectorstore,
    get_llm,
    answer_question,
)

st.set_page_config(page_title="DocChat", page_icon="💬", layout="wide")

EXAMPLE_PROMPTS = [
    "Summarize the key points of this document",
    "What are the main risks or limitations mentioned?",
    "List any numbers, dates, or figures that appear",
]

# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        max-width: 860px;
        padding-top: 2rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.25rem 0.5rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #eef3ff;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #f5f5f7;
    }

    div[data-testid="stChatInput"] {
        border-radius: 24px;
    }

    .app-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.1rem;
        margin-bottom: 1.4rem;
    }

    .source-list {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .source-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 8px;
        border-radius: 8px;
    }
    .source-row:hover {
        background: #f5f5f7;
    }
    .source-name {
        font-weight: 600;
        font-size: 0.8rem;
        color: #333;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .source-snippet {
        font-size: 0.8rem;
        color: #8b8b90;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        flex: 1;
        min-width: 0;
    }
    .score-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 1px 8px;
        font-size: 0.7rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    .score-high { background: #dcf5e6; color: #167a3e; }
    .score-medium { background: #fdf1d6; color: #9a6b00; }
    .score-low { background: #fbe3e3; color: #b3261e; }

    .confidence-banner {
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
    .confidence-banner.low {
        background: #fbe3e3;
        color: #8c1d18;
    }

    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #6b7280;
    }
    .example-chip {
        display: inline-block;
        background: #f5f5f7;
        border: 1px solid #e3e3e6;
        border-radius: 999px;
        padding: 6px 14px;
        margin: 4px;
        font-size: 0.85rem;
        color: #333;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def _get_secret(name: str) -> str:
    """Check Streamlit secrets first (for deployed apps), then env vars."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, "")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Settings")

    provider = st.selectbox("Model provider", ["openai", "gemini"], index=0)
    api_key = st.text_input(
        f"{provider.upper()} API key",
        type="password",
        value=_get_secret("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"),
    )
    model_name = st.text_input(
        "Model",
        value="gpt-4o-mini" if provider == "openai" else "gemini-3.6-flash",
    )

    with st.expander("Advanced"):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Chunks retrieved per answer", 1, 10, 4)
        search_type = st.radio(
            "Retrieval strategy",
            ["similarity", "mmr"],
            help=(
                "Similarity: return the most relevant chunks. "
                "MMR (Maximal Marginal Relevance): trades a little top relevance "
                "for less redundant, more diverse chunks — better for broad questions."
            ),
        )
        score_threshold = st.slider(
            "Minimum relevance score",
            0.0, 0.8, 0.0, step=0.05,
            help="Discard retrieved chunks below this relevance score (similarity mode only).",
        )

    st.divider()

    if st.session_state.chat_history:
        transcript_lines = [f"# Chat — {', '.join(st.session_state.processed_files) or 'DocChat'}\n"]
        for turn in st.session_state.chat_history:
            speaker = "You" if turn["role"] == "user" else "Assistant"
            transcript_lines.append(f"**{speaker}:** {turn['content']}\n")
        st.download_button(
            "Export conversation",
            data="\n".join(transcript_lines),
            file_name="conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="app-title">DocChat</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Chat with your documents. Attach files below, then ask anything.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# File attach — auto-indexes on change
# --------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Attach documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    signature = hashlib.md5(
        "".join(f"{f.name}:{f.size}" for f in uploaded_files).encode()
    ).hexdigest()

    if signature != st.session_state.indexed_signature:
        with st.spinner("Reading your documents..."):
            tmp_dir = tempfile.mkdtemp()
            file_paths = []
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                file_paths.append(path)

            docs = load_documents(file_paths)
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vectorstore = build_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.indexed_signature = signature
            st.session_state.chat_history = []

if st.session_state.processed_files:
    st.caption("Ready: " + ", ".join(st.session_state.processed_files))


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        rows = []
        for s in sources:
            page_str = f" · p.{s['page']}" if s.get("page") else ""
            score_html = ""
            if s.get("score") is not None:
                pct = round(s["score"] * 100)
                tier = "high" if s["score"] >= 0.6 else "medium" if s["score"] >= 0.35 else "low"
                score_html = f'<span class="score-badge score-{tier}">{pct}%</span>'

            snippet = " ".join(s["text"].split())  # collapse newlines/extra whitespace to one line
            snippet_attr = snippet.replace('"', "&quot;")

            rows.append(
                f'<div class="source-row" title="{snippet_attr}">'
                f'<span class="source-name">{s["source"]}{page_str}</span>'
                f"{score_html}"
                f'<span class="source-snippet">{snippet}</span>'
                f"</div>"
            )
        st.markdown('<div class="source-list">' + "".join(rows) + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
if not st.session_state.chat_history and st.session_state.vectorstore is not None:
    st.markdown('<div class="empty-state">Ask something about your document to get started.<br><br>', unsafe_allow_html=True)
    chip_html = "".join(f'<span class="example-chip">{p}</span>' for p in EXAMPLE_PROMPTS)
    st.markdown(chip_html + "</div>", unsafe_allow_html=True)

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant" and turn.get("confidence") == "low":
            st.markdown(
                '<div class="confidence-banner low">⚠ The documents may not fully cover this — treat this answer with caution.</div>',
                unsafe_allow_html=True,
            )
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            render_sources(turn["sources"])

question = st.chat_input("Message DocChat...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Attach a document above to get started.")
    elif not api_key:
        st.warning(f"Add your {provider.upper()} API key in Settings first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(provider, api_key, model_name)
                    result = answer_question(
                        st.session_state.vectorstore,
                        question,
                        llm,
                        k=top_k,
                        chat_history=st.session_state.chat_history[:-1],
                        search_type=search_type,
                        score_threshold=score_threshold,
                    )

                    if result.confidence == "low":
                        st.markdown(
                            '<div class="confidence-banner low">⚠ The documents may not fully cover this — treat this answer with caution.</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(result.answer)

                    sources_serialized = [
                        {"source": s.source, "page": s.page, "text": s.text, "score": s.score}
                        for s in result.sources
                    ]
                    if sources_serialized:
                        render_sources(sources_serialized)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": sources_serialized,
                            "confidence": result.confidence,
                        }
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

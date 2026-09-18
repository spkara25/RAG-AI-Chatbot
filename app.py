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

USER_AVATAR = "🙂"
BOT_AVATAR = "💬"

# --------------------------------------------------------------------------
# Look & feel — plain chat-app styling instead of a "tool" look
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        max-width: 820px;
        padding-top: 2rem;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.25rem 0.5rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stChatMessage"]:has(img[alt="🙂"]) {
        background-color: #eef3ff;
    }
    div[data-testid="stChatMessage"]:has(img[alt="💬"]) {
        background-color: #f5f5f7;
    }

    /* Chat input pinned bar */
    div[data-testid="stChatInput"] {
        border-radius: 24px;
    }

    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.1rem;
        margin-bottom: 1.2rem;
    }
    .source-pill {
        display: inline-block;
        background: #eef0f3;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #444;
        margin: 2px 4px 2px 0;
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
    st.session_state.chat_history = []  # list of {"role", "content"}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None

# --------------------------------------------------------------------------
# Sidebar — kept minimal, plain-language settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Settings")

    provider = st.selectbox("Model provider", ["openai", "gemini"], index=0)
    api_key = st.text_input(
        f"{provider.upper()} API key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY", ""),
    )
    model_name = st.text_input(
        "Model",
        value="gpt-4o-mini" if provider == "openai" else "gemini-3.6-flash",
    )

    with st.expander("Advanced"):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Chunks retrieved per answer", 1, 10, 4)

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
# File attach — auto-indexes on change, no separate "build" step
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

# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
for turn in st.session_state.chat_history:
    avatar = USER_AVATAR if turn["role"] == "user" else BOT_AVATAR
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    page_str = f", p. {s['page']}" if s.get("page") else ""
                    st.markdown(f'<span class="source-pill">{s["source"]}{page_str}</span>', unsafe_allow_html=True)
                    st.caption(s["text"])

question = st.chat_input("Message DocChat...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Attach a document above to get started.")
    elif not api_key:
        st.warning(f"Add your {provider.upper()} API key in Settings first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(provider, api_key, model_name)
                    result = answer_question(
                        st.session_state.vectorstore,
                        question,
                        llm,
                        k=top_k,
                        chat_history=st.session_state.chat_history[:-1],
                    )
                    st.markdown(result.answer)
                    sources_serialized = [
                        {"source": s.source, "page": s.page, "text": s.text}
                        for s in result.sources
                    ]
                    if sources_serialized:
                        with st.expander("Sources"):
                            for s in sources_serialized:
                                page_str = f", p. {s['page']}" if s.get("page") else ""
                                st.markdown(f'<span class="source-pill">{s["source"]}{page_str}</span>', unsafe_allow_html=True)
                                st.caption(s["text"])

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result.answer, "sources": sources_serialized}
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
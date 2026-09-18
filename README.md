# 📚 Mini AI Knowledge Assistant (RAG)

A simple Retrieval-Augmented Generation (RAG) app: upload documents, ask
questions, get answers grounded in those documents with source citations.

## How it works (pipeline)

```
Upload files → Load & extract text → Split into chunks → Embed chunks (local)
   → Store in FAISS vector index → User asks a question → Embed question
   → Retrieve top-k similar chunks → Send question + chunks to LLM
   → LLM answers using only that context → Show answer + sources
```

| Stage | Tool used |
|---|---|
| Document loading | LangChain loaders (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`) — supports PDF, TXT, DOCX |
| Chunking | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap by default, configurable in UI) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run **locally on CPU** — free, no API key, no rate limits |
| Vector store | **FAISS** (in-memory, no external service needed) |
| LLM (answer generation) | OpenAI `gpt-4o-mini` or Google `gemini-1.5-flash` (configurable) |
| UI | Streamlit |

### Why local embeddings?
Embedding is normally the most API-call-heavy step of a RAG pipeline (one
call per chunk, for every document you index). Running it locally with
`sentence-transformers` means indexing is completely free and has no rate
limits — **the only paid API call in the whole app is the single LLM call
made per question**, using a cheap model. This keeps the app safe to run
and re-index repeatedly without burning through API credits.

## Setup

```bash
git clone <this-repo>
cd rag-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get an API key for whichever provider you want to use:
- **OpenAI**: https://platform.openai.com/api-keys (paid, gpt-4o-mini is cheap — fractions of a cent per question)
- **Gemini**: https://aistudio.google.com/app/apikey (has a genuinely free tier — good if you want $0 cost)

You can either:
- Paste the key directly into the sidebar when the app is running, **or**
- Copy `.env.example` to `.env` and fill it in, then have the app read
  `os.environ` (the sidebar field pre-fills from env vars if present).

## Run

```bash
streamlit run app.py
```

Then in the browser:
1. Upload one or more PDF/TXT/DOCX files.
2. Click **Build / Rebuild Index**.
3. Type a question in the chat box at the bottom.
4. Expand **📎 Sources** under any answer to see exactly which document/page
   the answer was pulled from.

## Features implemented

Core requirements:
- ✅ PDF / TXT / DOCX documents as knowledge source
- ✅ Content extraction and chunking (configurable chunk size/overlap)
- ✅ Local embedding generation
- ✅ FAISS vector store for retrieval
- ✅ Question answering via LLM, grounded strictly in retrieved context
  (the system prompt explicitly instructs the model to say when the
  documents don't contain the answer, rather than guessing)
- ✅ Simple Streamlit chat interface

Bonus features:
- ✅ **Source citations** — every answer shows which file/page each
  supporting excerpt came from
- ✅ **Conversation history** — follow-up questions carry recent chat
  context (multi-turn chat, kept in `st.session_state`)
- ✅ **Multiple documents** — upload and index several files at once;
  each chunk is tagged with its source file so citations stay accurate
  across documents
- ✅ **Configurable retrieval** — chunk size, overlap, and number of
  retrieved chunks (`k`) are all adjustable from the sidebar without
  touching code

## Project structure

```
rag-assistant/
├── app.py            # Streamlit UI
├── rag_engine.py      # Loading, chunking, embeddings, vector store, LLM logic
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Troubleshooting

- **`Examining the path of transformers.models...` / `ModuleNotFoundError: No module named 'torchvision'`**
  in the terminal: this is harmless. It's Streamlit's hot-reload file watcher
  probing `transformers`' optional vision submodules, which need `torchvision`
  (not installed, and not needed — we only use a text embedding model). It's
  silenced by the logging line at the top of `app.py` and by
  `.streamlit/config.toml` (`fileWatcherType = "poll"`). It never affects the
  app's actual behavior.
- **Gemini model names**: Google renames/retires Flash versions frequently
  (2.5 → 3 → 3.1 → 3.8 and counting). The app defaults to the
  `gemini-flash-latest` alias, which Google keeps pointed at whatever the
  current Flash model is, so you shouldn't need to update this yourself. If
  you want a specific pinned version instead (for reproducibility), type its
  exact name into the "Model" field in the sidebar.

## Notes / limitations

- FAISS index is currently **in-memory only** (rebuilt each session). The
  engine includes `save_vectorstore` / `load_vectorstore` helpers if you
  want to persist an index to disk between runs — wiring a "load saved
  index" button into the UI is a natural next step.
- For very large document sets, consider swapping FAISS for a managed
  store (Pinecone, Chroma with persistence, etc.) — the retrieval
  interface (`as_retriever`) stays the same either way.
- Answer quality depends on chunk size/k tuning for your specific
  documents; the sidebar sliders let you experiment without code changes.

## Deployment

This is a standard Streamlit app, so it deploys as-is to
[Streamlit Community Cloud](https://streamlit.io/cloud) (free tier) —
just point it at `app.py` and set the API key as a secret rather than
typing it into the sidebar each time. It will also run in any container
platform (Render, Railway, Fly.io, etc.) with `streamlit run app.py --server.port $PORT`.

# 💬 DocChat — AI Knowledge Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chat app: attach documents, ask
questions, get answers grounded in those documents with source citations
and relevance scores.

## How it works (pipeline)

```
Attach files → Load & extract text → Split into chunks → Embed chunks (local)
   → Store in FAISS vector index → User asks a question → Embed question
   → Retrieve top-k relevant chunks (with relevance scores) → Send question
   + chunks to LLM → LLM answers using only that context
   → Show answer + sources + confidence
```

| Stage | Tool used |
|---|---|
| Document loading | LangChain loaders (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`) — supports PDF, TXT, DOCX |
| Chunking | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap by default, configurable) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run **locally on CPU** — free, no API key, no rate limits |
| Vector store | **FAISS** (in-memory, no external service needed) |
| Retrieval | Similarity search with normalized relevance scores, or MMR (diversity-aware), with an optional minimum-score cutoff |
| LLM (answer generation) | OpenAI `gpt-4o-mini` or Google `gemini-3.6-flash` (configurable) |
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
- Copy `.env.example` to `.env` and fill it in (read via `os.environ`), **or**
- For a deployed app, use Streamlit secrets — see **Deployment** below.

## Run

```bash
streamlit run app.py
```

Then in the browser:
1. Attach one or more PDF/TXT/DOCX files — indexing happens automatically,
   no separate "build" step.
2. Type a question in the chat box at the bottom.
3. Expand **Sources** under any answer to see which document/page it came
   from, with a relevance-match percentage for each excerpt.
4. If the retrieved context wasn't a strong match for the question, a
   caution banner appears above the answer instead of pretending it's
   confident.

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
  supporting excerpt came from, with a relevance-match % per source
- ✅ **Conversation history** — multi-turn chat with recent context carried
  into follow-up questions, plus a one-click transcript export
- ✅ **Multiple documents** — attach and index several files at once; each
  chunk is tagged with its source file so citations stay accurate across
  documents
- ✅ **Improved retrieval / evaluation** — two selectable retrieval
  strategies (plain similarity vs. MMR for diversity), a normalized
  relevance score shown per retrieved chunk, an adjustable minimum-score
  cutoff to discard weak matches, and a lightweight confidence signal
  (derived from retrieval scores, no extra LLM call) that flags answers
  where the documents likely don't cover the question
- ✅ **Deployment** — deploy-ready for Streamlit Community Cloud out of the
  box (see below), with a secrets template so API keys never need to be
  typed into the sidebar on a shared deployment

## Project structure

```
rag-assistant/
├── app.py                          # Streamlit UI
├── rag_engine.py                   # Loading, chunking, embeddings, scored retrieval, LLM logic
├── requirements.txt
├── .env.example
├── .streamlit/
│   ├── config.toml                 # File-watcher tweak (see Troubleshooting)
│   └── secrets.toml.example        # Template for deployed API keys
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
- **Gemini model names**: Google renames/retires Flash versions frequently.
  If the model in Settings 404s, check the exact name Google's error message
  recommends and paste it into the "Model" field — no code change needed.

## Notes / limitations

- FAISS index is currently **in-memory only** (rebuilt each session). The
  engine includes `save_vectorstore` / `load_vectorstore` helpers if you
  want to persist an index to disk between runs — wiring a "load saved
  index" button into the UI is a natural next step.
- For very large document sets, consider swapping FAISS for a managed
  store (Pinecone, Chroma with persistence, etc.) — the scored-retrieval
  interface stays conceptually the same either way.
- The confidence signal is a simple threshold on retrieval scores, not a
  full evaluation harness (e.g. no held-out QA set, no RAGAS-style
  faithfulness/answer-relevance metrics). It's meant as a cheap, always-on
  sanity check rather than a rigorous eval — a good next step if you want
  to go further.

## Deployment

**Streamlit Community Cloud (free, easiest):**
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app** → pick the repo and
   `app.py` as the entry point.
3. In the app's **Settings → Secrets**, paste in the contents of
   `.streamlit/secrets.toml.example` with your real key(s) filled in.
   The app reads these automatically (`st.secrets`), so users won't need
   to paste an API key into the sidebar at all.
4. Deploy. You'll get a public `*.streamlit.app` URL.

Note: `sentence-transformers` pulls in `torch`, which is a fairly large
dependency (several hundred MB). This installs fine on Streamlit Community
Cloud but can push close to free-tier resource limits on very small
containers elsewhere — worth knowing if you deploy somewhere more
constrained.

**Other platforms:** this is a standard Streamlit app, so it also runs
on any container platform (Render, Railway, Fly.io, etc.) with:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
Set the same secrets as environment variables there instead of
`st.secrets` — the app already falls back to `os.environ` if Streamlit
secrets aren't configured.


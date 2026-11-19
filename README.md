# MCT Course RAG

Retrieval-Augmented Generation assistant for the MBA **MCT** (Managing Change / Transformation) course. It indexes your local lecture notes and prescribed book, answers questions with citations, and can optionally search the internet. Use a **local LLM** via [Ollama](https://ollama.com) or a **cloud** model (OpenAI / Gemini).

> Course PDFs and DOCX files stay on your machine. This repo ships only the app code — do not commit copyrighted textbooks.

## Features

- Index PDF / DOCX / TXT / MD under a configurable folder (defaults to `C:\MBA\MBA\TERM-III\MCT\MCT`)
- Local LLM: Ollama (e.g. `llama3.2`, `qwen2.5:7b`, `gemma3:4b`)
- Cloud LLM: OpenAI or Google Gemini
- Embeddings: Ollama `nomic-embed-text`, local MiniLM, or OpenAI
- Optional web search (DuckDuckGo by default; Tavily if `TAVILY_API_KEY` is set)
- Streamlit chat UI with course + web citations

## Prerequisites

1. **Python 3.11+**
2. **[Ollama](https://ollama.com)** (for local mode) with at least one chat model and embeddings:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

3. Optional cloud keys in `.env` (see `.env.example`)

## Quick start

```bash
cd mct-rag
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env

streamlit run app.py
```

In the sidebar:

1. Confirm **Course materials folder** points at `C:\MBA\MBA\TERM-III\MCT\MCT` (slide handouts + prescribed book).
2. Pick **Local (Ollama)** or a cloud provider.
3. Click **Build / rebuild index** once (first run can take several minutes for large PDFs).
4. Ask questions. Toggle **Also search the internet** when you want external context.

## Project layout

```
mct-rag/
  app.py              # Streamlit UI
  src/
    config.py
    ingest.py         # Load + chunk + Chroma index
    llm.py            # Ollama / OpenAI / Gemini
    rag.py            # Retrieve + generate
    web_search.py     # DuckDuckGo / Tavily
  .env.example
  requirements.txt
```

Vector store files live under `data/chroma/` (gitignored).

## Sharing on GitHub

This repository is meant to share the **application**, not the course materials.

```bash
git add .
git commit -m "Add MCT course RAG with local/cloud LLM and web search"
gh repo create mct-rag --public --source=. --remote=origin --push
```

Or create an empty repo on GitHub, then:

```bash
git remote add origin https://github.com/<you>/mct-rag.git
git push -u origin main
```

Collaborators clone the app, point the sidebar at **their** local copy of MCT materials, and run `streamlit run app.py`.

## Notes

- Rebuilding the index after adding new notes is required for those files to be searchable.
- Keep the same embedding backend for build and query.
- The MCT folder PDFs are scanned images — first index runs OCR (cached under `data/ocr_cache/`) and can take 10–20+ minutes for the full book. Later rebuilds reuse the OCR cache and are much faster.
- Vector store lives under `data/chroma/` (gitignored).

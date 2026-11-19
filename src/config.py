"""Shared paths and defaults for MCT RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "mct_course"

DEFAULT_DOCS_PATH = os.getenv(
    "DEFAULT_DOCS_PATH",
    r"C:\MBA\MBA\TERM-III\MCT\MCT",
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

LOCAL_EMBED_MODEL = "nomic-embed-text"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOCAL_LLM_DEFAULT = "llama3.2:latest"
OPENAI_LLM_DEFAULT = "gpt-4o-mini"
GEMINI_LLM_DEFAULT = "gemini-2.0-flash"
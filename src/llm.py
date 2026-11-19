"""Local (Ollama) and cloud (OpenAI / Gemini) LLM + embedding helpers."""

from __future__ import annotations

import os
from typing import Any

from src.config import (
    FALLBACK_EMBED_MODEL,
    GEMINI_LLM_DEFAULT,
    LOCAL_EMBED_MODEL,
    LOCAL_LLM_DEFAULT,
    OLLAMA_BASE_URL,
    OPENAI_LLM_DEFAULT,
)


def get_embeddings(
    provider: str = "ollama",
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
):
    """Return an embeddings model.

    provider: "ollama" | "openai" | "local"
      - ollama → nomic-embed-text via local Ollama
      - openai → text-embedding-3-small
      - local  → sentence-transformers MiniLM (no API / no Ollama)
    """
    provider = (provider or "ollama").lower()
    base = ollama_base_url or OLLAMA_BASE_URL

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)

    if provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=FALLBACK_EMBED_MODEL)

    # Default: Ollama, with HuggingFace fallback if Ollama is down
    try:
        from langchain_ollama import OllamaEmbeddings

        emb = OllamaEmbeddings(model=LOCAL_EMBED_MODEL, base_url=base)
        # Light probe — will raise if Ollama is unreachable / model missing
        emb.embed_query("ping")
        return emb
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=FALLBACK_EMBED_MODEL)


def get_llm(
    mode: str = "local",
    model: str | None = None,
    temperature: float = 0.2,
    openai_api_key: str | None = None,
    google_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> Any:
    """Return a chat model.

    mode: "local" (Ollama) | "openai" | "gemini"
    """
    mode = (mode or "local").lower()
    base = ollama_base_url or OLLAMA_BASE_URL

    if mode == "openai":
        from langchain_openai import ChatOpenAI

        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required. Set it in the sidebar or .env")
        return ChatOpenAI(
            model=model or OPENAI_LLM_DEFAULT,
            temperature=temperature,
            api_key=key,
        )

    if mode == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API key required. Set it in the sidebar or .env")
        return ChatGoogleGenerativeAI(
            model=model or GEMINI_LLM_DEFAULT,
            temperature=temperature,
            google_api_key=key,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or LOCAL_LLM_DEFAULT,
        temperature=temperature,
        base_url=base,
    )


def list_ollama_models(ollama_base_url: str | None = None) -> list[str]:
    """Best-effort list of local Ollama tags."""
    import json
    import urllib.request

    base = (ollama_base_url or OLLAMA_BASE_URL).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:  # noqa: BLE001
        return []

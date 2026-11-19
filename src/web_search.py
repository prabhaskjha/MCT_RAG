"""Optional internet search for augmenting RAG answers."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class WebHit:
    title: str
    url: str
    snippet: str


def search_web(
    query: str,
    max_results: int = 5,
    tavily_api_key: str | None = None,
) -> list[WebHit]:
    """Search the web. Prefers Tavily when a key is present; else DuckDuckGo."""
    key = tavily_api_key or os.getenv("TAVILY_API_KEY")
    if key:
        return _tavily(query, max_results=max_results, api_key=key)
    return _duckduckgo(query, max_results=max_results)


def _duckduckgo(query: str, max_results: int = 5) -> list[WebHit]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore

    hits: list[WebHit] = []
    with DDGS() as ddgs:
        for row in ddgs.text(query, max_results=max_results):
            hits.append(
                WebHit(
                    title=row.get("title") or "",
                    url=row.get("href") or row.get("link") or "",
                    snippet=row.get("body") or row.get("snippet") or "",
                )
            )
    return hits


def _tavily(query: str, max_results: int = 5, api_key: str = "") -> list[WebHit]:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(max_results=max_results, tavily_api_key=api_key)
        raw = tool.invoke({"query": query})
        hits: list[WebHit] = []
        if isinstance(raw, list):
            for row in raw:
                hits.append(
                    WebHit(
                        title=row.get("title") or "",
                        url=row.get("url") or "",
                        snippet=row.get("content") or "",
                    )
                )
        return hits
    except Exception:  # noqa: BLE001 — fall back
        return _duckduckgo(query, max_results=max_results)


def format_web_context(hits: list[WebHit]) -> str:
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[{i}] {hit.title}\nURL: {hit.url}\n{hit.snippet}".strip()
        )
    return "\n\n".join(parts)

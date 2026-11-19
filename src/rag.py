"""Retrieve course context (+ optional web) and generate an answer."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.web_search import WebHit, format_web_context, search_web

SYSTEM_PROMPT = """You are an MBA teaching assistant for the MCT course
(Managing Change / Transformation — organizational change).

Answer using the provided COURSE CONTEXT first. If WEB CONTEXT is present,
you may use it to supplement recent examples or definitions, and clearly
distinguish course material from web findings.

Rules:
- Be accurate and concise.
- Cite sources by filename (and page when available) for course material.
- If the answer is not in the context, say so rather than inventing details.
- Prefer frameworks from the prescribed book and lecture notes when relevant.
"""


def retrieve_course_docs(vectorstore: Any, query: str, k: int = 5) -> list[Document]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)


def format_course_context(docs: list[Document]) -> str:
    if not docs:
        return "(No course chunks retrieved. Index your MCT folder first.)"
    blocks = []
    for i, doc in enumerate(docs, start=1):
        name = doc.metadata.get("filename") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        page_bit = f", page {page + 1}" if isinstance(page, int) else ""
        blocks.append(f"--- Source {i}: {name}{page_bit} ---\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(
    *,
    llm: Any,
    vectorstore: Any,
    question: str,
    use_web: bool = False,
    k: int = 5,
    tavily_api_key: str | None = None,
) -> dict:
    """Run RAG (+ optional web) and return answer + citations."""
    course_docs = retrieve_course_docs(vectorstore, question, k=k)
    course_ctx = format_course_context(course_docs)

    web_hits: list[WebHit] = []
    web_ctx = ""
    if use_web:
        web_hits = search_web(question, max_results=5, tavily_api_key=tavily_api_key)
        web_ctx = format_web_context(web_hits)

    user_prompt = (
        f"COURSE CONTEXT:\n{course_ctx}\n\n"
        + (f"WEB CONTEXT:\n{web_ctx}\n\n" if web_ctx else "")
        + f"QUESTION:\n{question}\n\n"
        "Write a helpful answer with brief citations."
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    answer = getattr(response, "content", str(response))

    citations = []
    for doc in course_docs:
        citations.append(
            {
                "type": "course",
                "filename": doc.metadata.get("filename")
                or doc.metadata.get("source")
                or "unknown",
                "page": doc.metadata.get("page"),
                "snippet": (doc.page_content or "")[:280],
            }
        )
    for hit in web_hits:
        citations.append(
            {
                "type": "web",
                "title": hit.title,
                "url": hit.url,
                "snippet": hit.snippet[:280],
            }
        )

    return {"answer": answer, "citations": citations, "course_docs": course_docs}
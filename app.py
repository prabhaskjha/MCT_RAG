"""MCT Course RAG — Streamlit UI with local/cloud LLM and optional web search."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on path when launched as `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from src.config import (  # noqa: E402
    DEFAULT_DOCS_PATH,
    GEMINI_LLM_DEFAULT,
    LOCAL_LLM_DEFAULT,
    OLLAMA_BASE_URL,
    OPENAI_LLM_DEFAULT,
)
from src.ingest import build_vectorstore, open_vectorstore  # noqa: E402
from src.llm import get_llm, list_ollama_models  # noqa: E402
from src.rag import answer_question  # noqa: E402

st.set_page_config(
    page_title="MCT Course RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 1100px; }
      .cite { font-size: 0.85rem; color: #444; border-left: 3px solid #c45c26;
              padding: 0.4rem 0.75rem; margin: 0.4rem 0; background: #faf7f4; }
      .webcite { border-left-color: #2a6f97; background: #f4f8fb; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _sidebar() -> dict:
    st.sidebar.title("MCT RAG settings")
    st.sidebar.caption("Managing Change / Transformation course assistant")

    docs_path = st.sidebar.text_input(
        "Course materials folder",
        value=os.getenv("DEFAULT_DOCS_PATH", DEFAULT_DOCS_PATH),
        help="Defaults to C:\\MBA\\MBA\\TERM-III\\MCT\\MCT (lecture handouts + book).",
    )

    st.sidebar.subheader("LLM")
    llm_mode = st.sidebar.radio(
        "Provider",
        options=["local", "openai", "gemini"],
        format_func=lambda x: {
            "local": "Local (Ollama)",
            "openai": "Cloud — OpenAI",
            "gemini": "Cloud — Gemini",
        }[x],
        horizontal=False,
    )

    ollama_url = st.sidebar.text_input("Ollama base URL", value=OLLAMA_BASE_URL)
    ollama_models = list_ollama_models(ollama_url) if llm_mode == "local" else []

    if llm_mode == "local":
        model_options = ollama_models or [
            LOCAL_LLM_DEFAULT,
            "qwen2.5:7b",
            "gemma3:4b",
        ]
        preferred = LOCAL_LLM_DEFAULT if LOCAL_LLM_DEFAULT in model_options else model_options[0]
        model = st.sidebar.selectbox(
            "Ollama model",
            options=model_options,
            index=model_options.index(preferred),
        )
        openai_key = os.getenv("OPENAI_API_KEY", "")
        google_key = os.getenv("GOOGLE_API_KEY", "")
    elif llm_mode == "openai":
        openai_key = st.sidebar.text_input(
            "OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        model = st.sidebar.text_input("OpenAI model", value=OPENAI_LLM_DEFAULT)
        google_key = os.getenv("GOOGLE_API_KEY", "")
    else:
        google_key = st.sidebar.text_input(
            "Google API key",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
        )
        model = st.sidebar.text_input("Gemini model", value=GEMINI_LLM_DEFAULT)
        openai_key = os.getenv("OPENAI_API_KEY", "")

    st.sidebar.subheader("Embeddings")
    embed_provider = st.sidebar.selectbox(
        "Embedding backend",
        options=["ollama", "local", "openai"],
        format_func=lambda x: {
            "ollama": "Ollama (nomic-embed-text)",
            "local": "Local MiniLM (no server)",
            "openai": "OpenAI embeddings",
        }[x],
        help="Use the same backend when indexing and querying.",
    )

    st.sidebar.subheader("Retrieval")
    use_web = st.sidebar.toggle(
        "Also search the internet",
        value=False,
        help="Augment course RAG with DuckDuckGo (or Tavily if TAVILY_API_KEY is set).",
    )
    top_k = st.sidebar.slider("Course chunks (k)", min_value=3, max_value=12, value=5)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.sidebar.divider()
    st.sidebar.caption(
        "Scanned PDFs use OCR on first build (can take 10–20+ min for the book)."
    )
    rebuild = st.sidebar.button("Build / rebuild index", type="primary", use_container_width=True)

    return {
        "docs_path": docs_path,
        "llm_mode": llm_mode,
        "model": model,
        "ollama_url": ollama_url,
        "openai_key": openai_key,
        "google_key": google_key,
        "embed_provider": embed_provider,
        "use_web": use_web,
        "top_k": top_k,
        "temperature": temperature,
        "rebuild": rebuild,
    }


@st.cache_resource(show_spinner=False)
def _cached_store(embed_provider: str, ollama_url: str, openai_key: str):
    return open_vectorstore(
        embed_provider=embed_provider,
        openai_api_key=openai_key or None,
        ollama_base_url=ollama_url,
    )


def main() -> None:
    cfg = _sidebar()

    st.title("MCT Course RAG")
    st.write(
        "Ask questions over your MCT lecture notes and prescribed book. "
        "Choose a **local Ollama** model or a **cloud** LLM, and optionally pull in web results."
    )

    if cfg["rebuild"]:
        status = st.status("Indexing course materials…", expanded=True)

        def progress(msg: str) -> None:
            status.write(msg)

        try:
            store = build_vectorstore(
                docs_path=cfg["docs_path"],
                embed_provider=cfg["embed_provider"],
                openai_api_key=cfg["openai_key"] or None,
                ollama_base_url=cfg["ollama_url"],
                rebuild=True,
                progress=progress,
            )
            _cached_store.clear()
            st.session_state["vectorstore"] = store
            status.update(label="Index ready", state="complete")
            st.success("Vector index built. You can ask questions below.")
        except Exception as exc:  # noqa: BLE001
            status.update(label="Indexing failed", state="error")
            st.error(str(exc))

    store = st.session_state.get("vectorstore")
    if store is None:
        with st.spinner("Opening existing index…"):
            store = _cached_store(
                cfg["embed_provider"],
                cfg["ollama_url"],
                cfg["openai_key"] or "",
            )
            if store is not None:
                st.session_state["vectorstore"] = store

    if store is None:
        st.info(
            "No index yet. Set the materials folder in the sidebar "
            f"(default: `{DEFAULT_DOCS_PATH}`) and click **Build / rebuild index**."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for cite in msg.get("citations") or []:
                _render_citation(cite)

    prompt = st.chat_input("Ask about MCT — change frameworks, cases, merger notes…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if store is None:
            reply = (
                "Index is empty. Click **Build / rebuild index** in the sidebar first."
            )
            st.warning(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            return

        try:
            with st.spinner("Thinking…"):
                llm = get_llm(
                    mode=cfg["llm_mode"],
                    model=cfg["model"],
                    temperature=cfg["temperature"],
                    openai_api_key=cfg["openai_key"] or None,
                    google_api_key=cfg["google_key"] or None,
                    ollama_base_url=cfg["ollama_url"],
                )
                result = answer_question(
                    llm=llm,
                    vectorstore=store,
                    question=prompt,
                    use_web=cfg["use_web"],
                    k=cfg["top_k"],
                    tavily_api_key=os.getenv("TAVILY_API_KEY"),
                )
            st.markdown(result["answer"])
            for cite in result["citations"]:
                _render_citation(cite)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            err = f"Could not answer: {exc}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})


def _render_citation(cite: dict) -> None:
    if cite.get("type") == "web":
        title = cite.get("title") or "Web result"
        url = cite.get("url") or ""
        snippet = cite.get("snippet") or ""
        st.markdown(
            f'<div class="cite webcite"><b>Web:</b> '
            f'<a href="{url}" target="_blank">{title}</a><br/>{snippet}</div>',
            unsafe_allow_html=True,
        )
    else:
        name = cite.get("filename") or "course"
        page = cite.get("page")
        page_bit = f" · p.{page + 1}" if isinstance(page, int) else ""
        snippet = cite.get("snippet") or ""
        st.markdown(
            f'<div class="cite"><b>Course:</b> {name}{page_bit}<br/>{snippet}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

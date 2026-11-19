"""Load course documents and build / refresh the Chroma vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    SUPPORTED_EXTENSIONS,
)
from src.llm import get_embeddings
from src.ocr_pdf import load_pdf_with_ocr

# Never index dependency trees or app internals if they sit under the docs root
_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "chroma",
    "ocr_cache",
}


def _should_skip_dir(path: Path) -> bool:
    parts_lower = [p.lower() for p in path.parts]
    if any(p in _SKIP_DIR_NAMES for p in parts_lower):
        return True
    # Skip the RAG app tree if docs root is a parent folder
    if "mct-rag" in parts_lower and "materials_slides" not in parts_lower:
        # allow data/materials_slides; skip the rest of mct-rag
        idx = parts_lower.index("mct-rag")
        rest = parts_lower[idx + 1 :]
        if rest and rest[0] != "data":
            return True
        if len(rest) >= 2 and rest[0] == "data" and rest[1] != "materials_slides":
            return True
    return False


def _iter_files(docs_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in docs_path.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_dir(path):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name.startswith(".") or path.name.startswith("~"):
            continue
        files.append(path)
    # Smaller files first so lecture handouts become searchable before the book
    return sorted(files, key=lambda p: (p.stat().st_size, str(p).lower()))


def load_documents(
    docs_path: str | Path,
    progress: Callable[[str], None] | None = None,
) -> list:
    root = Path(docs_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Documents folder not found: {root}")

    files = _iter_files(root)
    if not files:
        raise FileNotFoundError(
            f"No PDF/DOCX/TXT/MD files under {root}. "
            "Point the sidebar path at your MCT materials."
        )

    docs = []
    for i, path in enumerate(files, start=1):
        if progress:
            progress(f"Loading ({i}/{len(files)}): {path.name}")
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                loaded = load_pdf_with_ocr(path, progress=progress)
            elif suffix == ".docx":
                loaded = Docx2txtLoader(str(path)).load()
                for doc in loaded:
                    doc.metadata["source"] = str(path)
                    doc.metadata["filename"] = path.name
            else:
                loaded = TextLoader(
                    str(path), encoding="utf-8", autodetect_encoding=True
                ).load()
                for doc in loaded:
                    doc.metadata["source"] = str(path)
                    doc.metadata["filename"] = path.name

            for doc in loaded:
                text = (doc.page_content or "").strip()
                if not text:
                    continue
                doc.page_content = text
                docs.append(doc)
        except Exception as exc:  # noqa: BLE001 — continue indexing other files
            if progress:
                progress(f"Skipped {path.name}: {exc}")

    if not docs:
        raise ValueError(
            f"Loaded 0 text pages from {root}. "
            "Scanned PDFs need OCR — ensure rapidocr-onnxruntime is installed."
        )
    return docs


def build_vectorstore(
    docs_path: str | Path,
    embed_provider: str = "ollama",
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
    rebuild: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Chroma:
    """Chunk documents and persist a Chroma collection."""
    raw_docs = load_documents(docs_path, progress=progress)
    if progress:
        progress(f"Splitting {len(raw_docs)} document pages/sections…")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [
        c
        for c in splitter.split_documents(raw_docs)
        if (c.page_content or "").strip()
    ]
    if not chunks:
        raise ValueError("No non-empty text chunks after splitting.")

    if progress:
        progress(f"Embedding {len(chunks)} chunks…")

    embeddings = get_embeddings(
        provider=embed_provider,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if rebuild and CHROMA_DIR.exists():
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    batch_size = 32
    store: Chroma | None = None
    embedded = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [(c.page_content or "").strip() for c in batch]
        pairs = [(d, t) for d, t in zip(batch, texts) if t]
        if not pairs:
            continue
        keep_docs = [d for d, _ in pairs]

        if store is None:
            store = Chroma.from_documents(
                documents=keep_docs,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
                collection_name=COLLECTION_NAME,
            )
        else:
            store.add_documents(keep_docs)

        embedded += len(keep_docs)
        if progress and (start // batch_size) % 5 == 0:
            progress(f"Embedded {embedded}/{len(chunks)}")

    if store is None or embedded == 0:
        raise ValueError(
            "All embedding batches were empty. "
            "Try Embedding backend = Local MiniLM, or check Ollama nomic-embed-text."
        )

    if progress:
        progress(f"Indexed {embedded} chunks from {docs_path}")
    return store


def open_vectorstore(
    embed_provider: str = "ollama",
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> Chroma | None:
    """Open an existing Chroma DB, or return None if empty/missing."""
    if not CHROMA_DIR.exists():
        return None
    embeddings = get_embeddings(
        provider=embed_provider,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
    )
    store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    try:
        if store._collection.count() == 0:  # noqa: SLF001
            return None
    except Exception:  # noqa: BLE001
        return None
    return store

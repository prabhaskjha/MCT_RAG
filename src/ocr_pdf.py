"""OCR helpers for scanned course PDFs (image-only pages)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from src.config import PROJECT_ROOT

OCR_CACHE_DIR = PROJECT_ROOT / "data" / "ocr_cache"


def _cache_key(pdf_path: Path) -> str:
    st = pdf_path.stat()
    raw = f"{pdf_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cache(key: str) -> dict[str, str] | None:
    path = OCR_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _save_cache(key: str, pages: dict[str, str]) -> None:
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = OCR_CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(pages), encoding="utf-8")


def _get_ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def load_pdf_with_ocr(
    pdf_path: str | Path,
    progress: Callable[[str], None] | None = None,
    dpi: int = 120,
) -> list[Document]:
    """Extract text via PyMuPDF; OCR pages that have no text layer."""
    import io

    import pymupdf
    from PIL import Image

    path = Path(pdf_path)
    key = _cache_key(path)
    cached = _load_cache(key) or {}
    dirty = False

    doc = pymupdf.open(path)
    ocr = None
    out: list[Document] = []
    short = path.name if len(path.name) < 48 else path.name[:45] + "…"

    if progress:
        progress(f"Reading {short} ({doc.page_count} pages)…")

    for i in range(doc.page_count):
        page = doc[i]
        text = (page.get_text("text") or "").strip()

        if not text:
            if str(i) in cached:
                text = cached[str(i)] or ""
            else:
                if ocr is None:
                    if progress:
                        progress(f"Starting OCR for {short}…")
                    ocr = _get_ocr_engine()
                if progress and (i % 5 == 0 or i < 3):
                    progress(f"OCR {short}: page {i + 1}/{doc.page_count}")
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                result, _ = ocr(img)
                if result:
                    text = "\n".join(row[1] for row in result if row and len(row) > 1)
                else:
                    text = ""
                cached[str(i)] = text
                dirty = True
                # Checkpoint every 10 pages so long books survive interruption
                if dirty and i > 0 and i % 10 == 0:
                    _save_cache(key, cached)
                    dirty = False

        text = (text or "").strip()
        if not text:
            continue
        out.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "page": i,
                },
            )
        )

    if cached:
        _save_cache(key, cached)
    if progress:
        progress(f"Extracted {len(out)} text pages from {short}")
    return out

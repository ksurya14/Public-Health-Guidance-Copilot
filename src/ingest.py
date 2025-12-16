from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.utils import Document, clean_text, get_project_root, setup_logger

logger = setup_logger("ingest")

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        logger.warning("PyMuPDF not available; skipping PDF extraction for %s", path)
        return ""
    text_parts = []
    try:
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to extract text from %s: %s", path, exc)
        return ""
    return "\n".join(text_parts)


def load_documents(raw_dir: Optional[Path] = None, fallback_dir: Optional[Path] = None) -> List[Document]:
    root = get_project_root()
    raw_dir = raw_dir or (root / "data" / "raw")
    fallback_dir = fallback_dir or (root / "data" / "sample_corpus")

    paths = [p for p in sorted(raw_dir.glob("*")) if p.is_file()]
    if not paths:
        logger.info("No files in %s; using fallback corpus %s", raw_dir, fallback_dir)
        paths = [p for p in sorted(fallback_dir.glob("*")) if p.is_file()]

    documents: List[Document] = []
    for idx, path in enumerate(paths):
        text = ""
        if path.suffix.lower() in [".txt", ".md"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(path)
        else:
            logger.debug("Skipping unsupported file %s", path)
            continue

        text = clean_text(text)
        if not text:
            logger.warning("Empty text after cleaning for %s", path)
            continue

        title = path.stem.replace("_", " ").title()
        doc_id = f"DOC{idx:03d}"
        documents.append(Document(doc_id=doc_id, title=title, source_path=str(path), text=text))

    logger.info("Loaded %d documents", len(documents))
    return documents


if __name__ == "__main__":
    docs = load_documents()
    for d in docs:
        print(d.doc_id, d.title, len(d.text))
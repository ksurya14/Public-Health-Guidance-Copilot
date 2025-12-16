from __future__ import annotations

import re
from typing import List

from src.utils import Chunk, Document, setup_logger

logger = setup_logger("chunking")


def chunk_fixed(doc: Document, chunk_size: int = 700, overlap: int = 120) -> List[Chunk]:
    text = doc.text
    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end]
        chunk_id = f"{doc.doc_id}_c{idx}"
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_id=chunk_id,
                text=chunk_text,
                section=None,
                start=start,
                end=end,
                source_path=doc.source_path,
                title=doc.title,
            )
        )
        idx += 1
        if end == len(text):
            break
        start = max(0, end - overlap)

    logger.debug("Chunked %s into %d fixed chunks", doc.doc_id, len(chunks))
    return chunks


def split_into_sections(text: str) -> List[str]:
    # Split on blank lines or simple heading patterns
    sections = re.split(r"\n\s*\n|\n(?=Title:|Section |Chapter )", text)
    sections = [s.strip() for s in sections if s.strip()]
    return sections if sections else [text]


def chunk_section_aware(doc: Document, min_length: int = 180, max_length: int = 900) -> List[Chunk]:
    sections = split_into_sections(doc.text)
    chunks: List[Chunk] = []

    offset = 0
    idx = 0
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        if len(sec) > max_length:
            # Break large sections into smaller ones
            tmp_doc = Document(doc_id=doc.doc_id, title=doc.title, source_path=doc.source_path, text=sec)
            sub_chunks = chunk_fixed(tmp_doc, chunk_size=max_length, overlap=80)
            for sc in sub_chunks:
                sc.chunk_id = f"{doc.doc_id}_c{idx}"
                sc.start += offset
                sc.end += offset
                chunks.append(sc)
                idx += 1
        else:
            # Merge tiny sections into previous chunk when possible
            if chunks and len(sec) < min_length:
                prev = chunks[-1]
                prev.text = f"{prev.text}\n\n{sec}"
                prev.end = prev.start + len(prev.text)
            else:
                chunk_id = f"{doc.doc_id}_c{idx}"
                section_label = sec.split("\n")[0][:80]
                chunks.append(
                    Chunk(
                        doc_id=doc.doc_id,
                        chunk_id=chunk_id,
                        text=sec,
                        section=section_label,
                        start=offset,
                        end=offset + len(sec),
                        source_path=doc.source_path,
                        title=doc.title,
                    )
                )
                idx += 1

        offset += len(sec) + 2  # approximate separator length

    logger.debug("Chunked %s into %d section-aware chunks", doc.doc_id, len(chunks))
    return chunks


def chunk_documents(
    documents: List[Document],
    strategy: str = "fixed",
    fixed_size: int = 700,
    overlap: int = 120,
    min_length: int = 180,
    max_length: int = 900,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in documents:
        if strategy == "section":
            all_chunks.extend(chunk_section_aware(doc, min_length=min_length, max_length=max_length))
        else:
            all_chunks.extend(chunk_fixed(doc, chunk_size=fixed_size, overlap=overlap))
    logger.info("Created %d chunks using %s strategy", len(all_chunks), strategy)
    return all_chunks
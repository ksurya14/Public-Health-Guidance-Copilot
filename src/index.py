from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

#import faiss
import numpy as np
from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer

from src.chunking import chunk_documents
from src.ingest import load_documents
from src.utils import Chunk, ensure_dir, get_project_root, load_config, save_json, save_pickle, setup_logger

logger = setup_logger("index")


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25(chunks: List[Chunk]) -> BM25Okapi:
    corpus_tokens = [tokenize(c.text) for c in chunks]
    return BM25Okapi(corpus_tokens)


# def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)
#     return index


# def encode_chunks(chunks: List[Chunk], model_name: str) -> np.ndarray:
#     model = SentenceTransformer(model_name, device="cpu")
#     texts = [c.text for c in chunks]
#     embeddings = model.encode(
#         texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True, batch_size=8
#     )
#     return embeddings


def save_metadata(chunks: List[Chunk], path: Path, config: Dict[str, Any]) -> None:
    data = {
        "chunks": [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "section": c.section,
                "start": c.start,
                "end": c.end,
                "source_path": c.source_path,
                "title": c.title,
            }
            for c in chunks
        ],
        "config": config,
    }
    save_json(data, path)


def index_corpus(config_path: Path | None = None, chunk_strategy: str | None = None) -> None:
    config = load_config(config_path)
    chunk_cfg = config["chunking"]
    strategy = chunk_strategy or chunk_cfg.get("default_strategy", "fixed")

    documents = load_documents(
        raw_dir=get_project_root() / config["paths"]["raw_dir"],
        fallback_dir=get_project_root() / config["paths"]["fallback_corpus"],
    )
    if not documents:
        logger.error("No documents found to index.")
        return

    chunks = chunk_documents(
        documents,
        strategy=strategy,
        fixed_size=chunk_cfg.get("fixed", {}).get("chunk_size", 700),
        overlap=chunk_cfg.get("fixed", {}).get("overlap", 120),
        min_length=chunk_cfg.get("section", {}).get("min_length", 180),
        max_length=chunk_cfg.get("section", {}).get("max_length", 900),
    )

    bm25 = build_bm25(chunks)
    logger.info("Built BM25 over %d chunks", len(chunks))

    model_name = config["models"]["embedding_model"]
    # embeddings = encode_chunks(chunks, model_name)
    # dense_index = build_faiss_index(embeddings)
    # logger.info("Built FAISS index with dim=%d", embeddings.shape[1])

    base_dir = get_project_root() / config["paths"]["index_dir"]
    index_dir = base_dir / strategy
    ensure_dir(index_dir)

    # faiss.write_index(dense_index, str(index_dir / "dense.faiss"))
    save_pickle(bm25, index_dir / "bm25.pkl")
    # np.save(index_dir / "embeddings.npy", embeddings)
    save_metadata(chunks, index_dir / "metadata.json", config)

    logger.info("Saved index to %s", index_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indices for the RAG system")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--chunk_strategy", type=str, default=None)
    args = parser.parse_args()
    index_corpus(args.config, args.chunk_strategy)


if __name__ == "__main__":
    main()
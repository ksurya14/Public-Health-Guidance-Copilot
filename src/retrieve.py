from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.utils import (
    Chunk,
    RetrievedChunk,
    get_project_root,
    load_config,
    load_json,
    load_pickle,
    setup_logger,
)

logger = setup_logger("retrieve")

# FAISS is optional now
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore


def tokenize(text: str) -> List[str]:
    return text.lower().split()


@dataclass
class Retriever:
    bm25: BM25Okapi
    faiss_index: Optional["faiss.Index"]  # type: ignore[name-defined]
    chunks: List[Chunk]
    embedder: Optional[SentenceTransformer]
    reranker: Optional[CrossEncoder]
    config: Dict

    @classmethod
    def load(
        cls,
        config: Optional[Dict] = None,
        use_reranker: Optional[bool] = None,
        chunk_strategy: Optional[str] = None,
    ) -> "Retriever":
        cfg = config or load_config()
        strategy = chunk_strategy or cfg.get("chunking", {}).get("default_strategy", "fixed")
        base_dir = get_project_root() / cfg["paths"]["index_dir"]
        index_dir = base_dir / strategy

        if not (index_dir / "bm25.pkl").exists():
            raise FileNotFoundError(f"No index found at {index_dir}. Run `make index` first.")

        bm25 = load_pickle(index_dir / "bm25.pkl")
        meta = load_json(index_dir / "metadata.json")

        chunk_objs: List[Chunk] = []
        for c in meta["chunks"]:
            chunk_objs.append(
                Chunk(
                    doc_id=c["doc_id"],
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    section=c.get("section"),
                    start=c["start"],
                    end=c["end"],
                    source_path=c["source_path"],
                    title=c["title"],
                )
            )

        # Optional dense retrieval assets
        faiss_index = None
        embedder = None
        dense_path = index_dir / "dense.faiss"

        if faiss is not None and dense_path.exists():
            try:
                faiss_index = faiss.read_index(str(dense_path))
                embedder = SentenceTransformer(cfg["models"]["embedding_model"], device="cpu")
                logger.info("Dense retrieval enabled (found %s)", dense_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Dense retrieval disabled (failed to load FAISS): %s", exc)
                faiss_index = None
                embedder = None
        else:
            logger.info("Dense retrieval disabled (dense.faiss not found). Using BM25-only.")

        # Optional reranker
        reranker_obj: Optional[CrossEncoder] = None
        enable_rerank = use_reranker if use_reranker is not None else cfg["retrieval"].get("use_reranker", True)
        reranker_model = cfg.get("models", {}).get("reranker_model")

        if enable_rerank and reranker_model:
            try:
                reranker_obj = CrossEncoder(reranker_model, device="cpu")
                logger.info("Loaded reranker %s", reranker_model)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not load reranker (%s); proceeding without it", exc)
                reranker_obj = None

        return cls(
            bm25=bm25,
            faiss_index=faiss_index,
            chunks=chunk_objs,
            embedder=embedder,
            reranker=reranker_obj,
            config=cfg,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        cfg = self.config
        top_k = int(top_k or cfg["retrieval"].get("top_k", 5))

        weight_cfg = weights or cfg["retrieval"].get("hybrid_weights", {"bm25": 0.5, "dense": 0.5})
        bm25_w = float(weight_cfg.get("bm25", 1.0))
        dense_w = float(weight_cfg.get("dense", 0.0))

        # If dense assets are missing, force dense weight to 0
        if self.faiss_index is None or self.embedder is None:
            dense_w = 0.0

        # BM25 scores over all chunks
        bm25_scores = self.bm25.get_scores(tokenize(query))

        # Candidate pool size
        search_k = min(max(top_k * 4, int(cfg["retrieval"].get("rerank_top_k", top_k))), len(self.chunks))

        # Start with BM25 candidates
        top_bm25_indices = np.argsort(bm25_scores)[-search_k:][::-1]
        candidate_ids: Set[int] = {int(i) for i in top_bm25_indices.tolist() if int(i) >= 0}

        dense_map: Dict[int, float] = {}

        # Optional dense retrieval
        if dense_w > 0.0 and self.faiss_index is not None and self.embedder is not None:
            query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            dense_scores, dense_indices = self.faiss_index.search(query_vec, search_k)
            dense_scores = dense_scores[0]
            dense_indices = dense_indices[0]
            dense_map = {int(i): float(s) for i, s in zip(dense_indices, dense_scores) if int(i) >= 0}
            candidate_ids |= {int(i) for i in dense_indices.tolist() if int(i) >= 0}

        # Combine scores (hybrid if dense_w > 0, otherwise BM25-only)
        combined: List[RetrievedChunk] = []
        for cid in candidate_ids:
            if cid < 0 or cid >= len(self.chunks):
                continue
            score = bm25_w * float(bm25_scores[cid]) + dense_w * float(dense_map.get(cid, 0.0))
            combined.append(RetrievedChunk(chunk=self.chunks[cid], score=score))

        combined_sorted = sorted(combined, key=lambda x: x.score, reverse=True)[:search_k]

        # Optional rerank
        do_rerank = use_reranker if use_reranker is not None else cfg["retrieval"].get("use_reranker", True)
        if do_rerank and self.reranker is not None and combined_sorted:
            pairs = [[query, rc.chunk.text] for rc in combined_sorted]
            rerank_scores = self.reranker.predict(pairs)
            for rc, rs in zip(combined_sorted, rerank_scores):
                rc.score = float(rs)
            combined_sorted = sorted(combined_sorted, key=lambda x: x.score, reverse=True)

        return combined_sorted[:top_k]
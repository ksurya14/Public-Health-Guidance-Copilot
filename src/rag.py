from __future__ import annotations

from typing import Dict, List, Optional

from src.retrieve import Retriever
from src.llm import get_backend
from src.utils import RetrievedChunk, load_config, setup_logger

logger = setup_logger("rag")


def format_context(retrieved: List[RetrievedChunk], max_chunks: int = 5) -> str:
    snippets = []
    for rc in retrieved[:max_chunks]:
        cite = f"[{rc.chunk.doc_id}:{rc.chunk.chunk_id}]"
        snippet = rc.chunk.text.replace("\n", " ").strip()
        snippets.append(f"{cite} \"{snippet}\"")
    return "\n\n".join(snippets)


def answer_question(question: str, retrieval_overrides: Optional[Dict] = None) -> Dict:
    config = load_config()
    retrieval_cfg = config.get("retrieval", {})
    gen_cfg = config.get("generation", {})
    chunk_strategy = config.get("chunking", {}).get("default_strategy", "fixed")

    weights = retrieval_cfg.get("hybrid_weights", {"bm25": 0.5, "dense": 0.5})
    top_k = int(retrieval_cfg.get("top_k", 5))
    use_reranker = bool(retrieval_cfg.get("use_reranker", True))

    if retrieval_overrides:
        weights = retrieval_overrides.get("weights", weights)
        top_k = int(retrieval_overrides.get("top_k", top_k))
        use_reranker = bool(retrieval_overrides.get("use_reranker", use_reranker))
        chunk_strategy = retrieval_overrides.get("chunk_strategy", chunk_strategy)

    try:
        retriever = Retriever.load(config=config, use_reranker=use_reranker, chunk_strategy=chunk_strategy)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return {"answer": "Index not found. Please run `make index`.", "contexts": [], "backend": "mock"}

    retrieved = retriever.retrieve(question, top_k=top_k, weights=weights, use_reranker=use_reranker)
    if not retrieved:
        return {"answer": "Not found in provided documents.", "contexts": [], "backend": "mock"}

    context_text = format_context(retrieved, max_chunks=int(gen_cfg.get("max_context_chunks", 5)))
    backend = get_backend(config)
    answer = backend.generate(
        question=question,
        context=context_text,
        system_prompt=gen_cfg.get("system_prompt", "Answer with citations."),
        model=config.get("models", {}).get("llm_model", "gpt-3.5-turbo"),
        temperature=float(gen_cfg.get("temperature", 0.2)),
    )

    return {
        "answer": answer,
        "contexts": [
            {
                "chunk_id": rc.chunk.chunk_id,
                "doc_id": rc.chunk.doc_id,
                "title": rc.chunk.title,
                "source_path": rc.chunk.source_path,
                "text": rc.chunk.text,
                "score": rc.score,
            }
            for rc in retrieved
        ],
        "backend": backend.name,
        "chunk_strategy": chunk_strategy,
    }


if __name__ == "__main__":
    q = "Who should get the flu vaccine and when?"
    response = answer_question(q)
    print(response["answer"])
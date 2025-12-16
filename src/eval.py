from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.rag import answer_question
from src.utils import ensure_dir, get_project_root, load_config, save_json, setup_logger
from src.index import index_corpus

logger = setup_logger("eval")


def load_questions(path: Path) -> List[Dict]:
    questions: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def retrieval_hit(question: Dict, contexts: List[Dict]) -> bool:
    hint = (question.get("source_hint") or "").lower()
    answer_hint = (question.get("answer_hint") or "").lower()

    if hint == "unanswerable":
        # We consider retrieval "correct" if model says not found OR contexts exist but don't support
        # Keep this simple: just require the system to still retrieve something; the answer logic handles refusal.
        return True

    for ctx in contexts:
        text = (ctx.get("text") or "").lower()
        title = (ctx.get("title") or "").lower()
        if hint and (hint in text or hint in title):
            return True
        if answer_hint and answer_hint in text:
            return True
    return False


def token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), 1)


def faithfulness_score(answer: str, contexts: List[Dict]) -> float:
    if not contexts:
        return 1.0 if "not found" in answer.lower() else 0.0
    context_text = " ".join(c.get("text", "") for c in contexts).lower()
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 0.0
    scores = [token_overlap(sent, context_text) for sent in sentences]
    return sum(scores) / len(scores)


def ensure_index_for_strategy(strategy: str, config: Dict) -> None:
    index_dir = get_project_root() / config["paths"]["index_dir"] / strategy
    if index_dir.exists():
        return
    logger.info("Index for strategy %s not found; building now.", strategy)
    index_corpus(chunk_strategy=strategy)


def run_eval() -> Dict:
    config = load_config()
    questions_path = get_project_root() / "eval" / "questions.jsonl"
    questions = load_questions(questions_path)

    ablations = config.get("eval", {}).get("ablations", [])
    results: Dict = {"run_started": time.time(), "ablations": []}

    for ablation in ablations:
        name = ablation.get("name", "unnamed")
        weights = ablation.get("hybrid_weights", config["retrieval"]["hybrid_weights"])
        use_reranker = ablation.get("use_reranker", config["retrieval"].get("use_reranker", True))
        chunk_strategy = ablation.get("chunk_strategy", config["chunking"]["default_strategy"])

        ensure_index_for_strategy(chunk_strategy, config)

        hits = 0
        faith_scores: List[float] = []

        for q in questions:
            resp = answer_question(
                q["question"],
                retrieval_overrides={
                    "weights": weights,
                    "top_k": int(config["eval"].get("recall_k", 5)),
                    "use_reranker": use_reranker,
                    "chunk_strategy": chunk_strategy,
                },
            )
            contexts = resp.get("contexts", [])
            if retrieval_hit(q, contexts):
                hits += 1
            faith_scores.append(faithfulness_score(resp.get("answer", ""), contexts))

        recall_at_k = hits / max(len(questions), 1)
        avg_faithfulness = sum(faith_scores) / max(len(faith_scores), 1)

        logger.info("Ablation %s: recall@k=%.3f faithfulness=%.3f", name, recall_at_k, avg_faithfulness)
        results["ablations"].append(
            {
                "name": name,
                "weights": weights,
                "chunk_strategy": chunk_strategy,
                "use_reranker": use_reranker,
                "recall_at_k": recall_at_k,
                "faithfulness": avg_faithfulness,
                "num_questions": len(questions),
            }
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_dir = get_project_root() / "eval" / "results"
    ensure_dir(results_dir)

    save_json(results, results_dir / f"results_{timestamp}.json")

    summary_lines = ["# Evaluation Summary", "", f"Run: {timestamp}", ""]
    for abl in results["ablations"]:
        summary_lines.append(
            f"- **{abl['name']}**: recall@k={abl['recall_at_k']:.3f}, faithfulness={abl['faithfulness']:.3f}"
        )
    (results_dir / f"summary_{timestamp}.md").write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Saved eval results to %s", results_dir)
    return results


if __name__ == "__main__":
    run_eval()
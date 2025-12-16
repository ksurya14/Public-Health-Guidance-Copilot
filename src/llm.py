from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.utils import setup_logger, ENV_VARS

logger = setup_logger("llm")

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass
class LLMBackend:
    name: str

    def generate(self, question: str, context: str, system_prompt: str, model: str, temperature: float = 0.2) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__(name="openai")
        if OpenAI is None:
            raise ImportError("openai package not available")
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "") or None
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, question: str, context: str, system_prompt: str, model: str, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Answer using ONLY the context. Cite sources as [doc_id:chunk_id] and include short quotes. "
                    'If the answer is not supported, say exactly: "Not found in provided documents."'
                ),
            },
        ]
        try:
            resp = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            return resp.choices[0].message.content or "Not found in provided documents."
        except Exception as exc:  # pragma: no cover
            logger.error("OpenAI backend failed: %s", exc)
            return "Not found in provided documents."


class MockBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__(name="mock")

    def generate(self, question: str, context: str, system_prompt: str, model: str, temperature: float = 0.0) -> str:
        if not context.strip():
            return "Not found in provided documents."
        # Extractive-ish: return first 2 cited snippets (good enough for offline demo)
        lines = [l.strip() for l in context.split("\n") if l.strip()]
        picked = lines[:2]
        return "\n".join(picked) + "\n\nNoted: (mock backend) Answer extracted from retrieved context.\n"


def get_backend(config: dict) -> LLMBackend:
    preferred = config.get("models", {}).get("llm_backend", "auto")
    has_key = bool(os.getenv("OPENAI_API_KEY", ""))
    if preferred in ("openai", "auto") and has_key:
        return OpenAIBackend()
    logger.info("Using mock LLM backend (no OPENAI_API_KEY set)")
    return MockBackend()
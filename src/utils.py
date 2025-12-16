from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def get_project_root() -> Path:
    # src/ -> project root
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or (get_project_root() / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "rag", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_text(text: str) -> str:
    """Normalize whitespace but keep paragraph boundaries for section-aware chunking."""
    lines = [line.strip() for line in text.replace("\r", "").split("\n")]
    paragraphs: List[str] = []
    current: List[str] = []
    for line in lines:
        if line:
            current.append(line)
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    text: str


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    section: Optional[str]
    start: int
    end: int
    source_path: str
    title: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


ENV_VARS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
}
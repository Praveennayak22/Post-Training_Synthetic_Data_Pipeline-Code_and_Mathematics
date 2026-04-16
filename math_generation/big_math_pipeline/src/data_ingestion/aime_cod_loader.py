"""
AIME-COD loader
────────────────
Dataset : KingNish/AIME-COD
HF page : https://huggingface.co/datasets/KingNish/AIME-COD

Observed schema (March 2026):
  ID / id            – unique identifier (e.g., 1983-1)
  Question / question – problem statement
  Answer / answer     – final integer answer
  reasoning / solution – chain-of-draft style reasoning
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .answer_utils import pick_reference_answer
from .base import SeedSample

_DOMAIN_KEYWORD_PRIORITY = [
    ("geometry",             ["triangle", "circle", "polygon", "angle", "area", "perimeter"]),
    ("number_theory",        ["prime", "divisib", "mod", "gcd", "lcm", "integer", "digit"]),
    ("intermediate_algebra", ["log", "exponential", "matrix", "determinant", "complex"]),
    ("algebra",              ["equation", "inequalit", "function", "polynomial", "sequence"]),
]


def _first_non_empty(row: dict, keys: list[str]) -> str | None:
    for key in keys:
        val = row.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return None


def _infer_domain(problem: str) -> str:
    text = problem.lower()
    for dom, keywords in _DOMAIN_KEYWORD_PRIORITY:
        if any(kw in text for kw in keywords):
            return dom
    return "algebra"


def load_aime_cod(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "KingNish/AIME-COD",
    split: str = "train",
    max_samples: int | None = None,
    **_kwargs,
) -> Iterator[SeedSample]:
    logger.info("Loading AIME-COD (split={}, cap={})", split, max_samples)

    load_kwargs: dict = dict(
        path=repo_id,
        split=split,
        trust_remote_code=False,
        cache_dir=cache_dir,
    )
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = load_dataset(**load_kwargs)

    count = 0
    for idx, row in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break

        problem = _first_non_empty(row, ["Question", "question", "problem", "prompt"])
        if not problem:
            continue

        solution = _first_non_empty(row, ["reasoning", "solution", "explanation"])
        answer = pick_reference_answer(
            _first_non_empty(row, ["Answer", "answer", "final_answer"]),
            solution,
        )

        sample_id = _first_non_empty(row, ["ID", "id"]) or f"{split}:{idx}"

        yield SeedSample(
            id=f"aime_cod:{sample_id}",
            source="aime_cod",
            domain=_infer_domain(problem),
            problem=problem,
            solution=solution,
            answer=answer,
            raw={
                "repo_id": repo_id,
                "split": split,
                "row_id": sample_id,
            },
        )
        count += 1

    logger.info("AIME-COD: yielded {} samples", count)

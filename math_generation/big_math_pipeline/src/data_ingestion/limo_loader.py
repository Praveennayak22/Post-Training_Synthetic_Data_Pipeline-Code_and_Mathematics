"""
LIMO loader
───────────
Dataset : GAIR/LIMO  (Apache-2.0)
HF page : https://huggingface.co/datasets/GAIR/LIMO

Observed schema (March 2026):
  question  – problem statement
  solution  – reasoning / explanation
  answer    – final answer text
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample
from .answer_utils import pick_reference_answer

_DOMAIN_KEYWORD_PRIORITY = [
    ("geometry",             ["triangle", "circle", "polygon", "angle", "area", "perimeter"]),
    ("polynomial_roots",     ["polynomial", "vieta", "cubic", "quartic", "roots of"]),
    ("number_theory",        ["prime", "divisib", "mod", "gcd", "lcm", "integer", "digit"]),
    ("intermediate_algebra", ["logarithm", "exponential", "complex", "matrix", "determinant"]),
    ("algebra",              ["equation", "inequalit", "function", "quadratic", "sequence"]),
]


def _infer_domain(problem: str) -> str:
    text = problem.lower()
    for dom, keywords in _DOMAIN_KEYWORD_PRIORITY:
        if any(kw in text for kw in keywords):
            return dom
    return "algebra"


def load_limo(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "GAIR/LIMO",
    split: str = "train",
    max_samples: int | None = None,
    **_kwargs,
) -> Iterator[SeedSample]:
    logger.info("Loading LIMO (split={}, cap={})", split, max_samples)

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

        problem = (row.get("question") or "").strip()
        if not problem:
            continue

        solution = (row.get("solution") or "").strip() or None
        answer = pick_reference_answer(row.get("answer"), solution)

        yield SeedSample(
            id=f"limo:{split}:{idx}",
            source="limo",
            domain=_infer_domain(problem),
            problem=problem,
            solution=solution,
            answer=answer,
            raw={
                "repo_id": repo_id,
                "split": split,
            },
        )
        count += 1

    logger.info("LIMO: yielded {} samples", count)

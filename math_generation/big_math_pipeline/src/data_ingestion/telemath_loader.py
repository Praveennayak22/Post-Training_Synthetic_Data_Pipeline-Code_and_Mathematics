"""
TeleMath loader
───────────────
Dataset : netop/TeleMath  (gated access)
HF page : https://huggingface.co/datasets/netop/TeleMath

Observed schema from dataset card (March 2026):
  question   – telecom mathematical question
  answer     – numerical answer
  category   – broad category label
  tags       – telecom topic tags
  difficulty – basic | advanced

Notes:
  - Access requires accepting the dataset conditions on Hugging Face.
  - No full reference solution is exposed in the public card excerpt.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample
from .answer_utils import pick_reference_answer


def _infer_domain(question: str, difficulty: str | None) -> str:
    text = question.lower()
    if any(token in text for token in ["probability", "ber", "snr", "noise", "signal"]):
        return "intermediate_algebra"
    if difficulty and str(difficulty).strip().lower() == "advanced":
        return "intermediate_algebra"
    return "algebra"


def load_telemath(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "netop/TeleMath",
    split: str = "train",
    max_samples: int | None = None,
    **_kwargs,
) -> Iterator[SeedSample]:
    logger.info("Loading TeleMath (split={}, cap={})", split, max_samples)

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

        answer = pick_reference_answer(row.get("answer"))
        difficulty = row.get("difficulty")

        yield SeedSample(
            id=f"telemath:{split}:{idx}",
            source="telemath",
            domain=_infer_domain(problem, difficulty),
            problem=problem,
            solution=None,
            answer=answer,
            raw={
                "category": row.get("category"),
                "tags": row.get("tags"),
                "difficulty": difficulty,
            },
        )
        count += 1

    logger.info("TeleMath: yielded {} samples", count)
"""
tw-math-reasoning-2k loader
──────────────────────────
Dataset : twinkle-ai/tw-math-reasoning-2k  (MIT)
HF page : https://huggingface.co/datasets/twinkle-ai/tw-math-reasoning-2k

Observed schema (March 2026):
  problem       – original English problem
  level         – difficulty label
  type          – math type/category
  solution      – English solution
  subset        – source subset
  split         – dataset split
  model         – generation model
  problem_zhtw  – Traditional Chinese translation
  think         – Traditional Chinese reasoning trace
  answer        – final answer (often boxed)
  messages      – conversation structure
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample
from .answer_utils import pick_reference_answer

_TYPE_TO_DOMAIN = {
    "prealgebra": "pre_algebra",
    "algebra": "algebra",
    "geometry": "geometry",
    "number theory": "number_theory",
    "number_theory": "number_theory",
    "intermediate algebra": "intermediate_algebra",
    "intermediate_algebra": "intermediate_algebra",
    "counting & probability": "pre_algebra",
    "counting_and_probability": "pre_algebra",
    "precalculus": "intermediate_algebra",
}


def _infer_domain(type_name: str | None) -> str:
    if not type_name:
        return "algebra"
    normalized = type_name.strip().lower()
    return _TYPE_TO_DOMAIN.get(normalized, "algebra")


def load_tw_math_reasoning(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "twinkle-ai/tw-math-reasoning-2k",
    split: str = "train",
    max_samples: int | None = None,
    use_traditional_chinese_problem: bool = False,
    use_traditional_chinese_think: bool = False,
    english_only: bool = True,
    **_kwargs,
) -> Iterator[SeedSample]:
    logger.info("Loading tw-math-reasoning-2k (split={}, cap={}, english_only={})", split, max_samples, english_only)

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

        problem_en = (row.get("problem") or "").strip()
        problem_zh = (row.get("problem_zhtw") or "").strip()
        
        # English-only filter: skip rows without English problem
        if english_only and not problem_en:
            continue
            
        problem = problem_zh if use_traditional_chinese_problem and problem_zh else problem_en
        if not problem:
            continue

        solution_en = (row.get("solution") or "").strip()
        think_zh = (row.get("think") or "").strip()
        solution = think_zh if use_traditional_chinese_think and think_zh else solution_en or think_zh or None

        answer = pick_reference_answer(
            row.get("answer"),
            solution_en,
            think_zh,
        )

        yield SeedSample(
            id=f"tw_math_reasoning:{split}:{idx}",
            source="tw_math_reasoning",
            domain=_infer_domain(row.get("type")),
            problem=problem,
            solution=solution,
            answer=answer,
            raw={
                "level": row.get("level"),
                "type": row.get("type"),
                "subset": row.get("subset"),
                "split": row.get("split"),
                "model": row.get("model"),
                "problem_zhtw": problem_zh,
                "messages": row.get("messages"),
            },
        )
        count += 1

    logger.info("tw-math-reasoning-2k: yielded {} samples", count)
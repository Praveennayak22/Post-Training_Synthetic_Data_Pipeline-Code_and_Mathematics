"""
s1K loader
──────────
Dataset : simplescaling/s1K  (Apache-2.0)
HF page : https://huggingface.co/datasets/simplescaling/s1K

Observed schema (March 2026):
  question               – math question text
  solution               – distilled reference solution
  cot_type               – coarse category (often 'math')
  source_type            – original source provenance
  metadata               – original row metadata (stringified or nested)
  cot                    – optional CoT field (often None)
  thinking_trajectories  – list[str] reasoning traces from Gemini
  attempt                – generated response / completion
"""

from __future__ import annotations

import re
from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample
from .answer_utils import pick_reference_answer, normalize_reference_answer

_DOMAIN_KEYWORD_PRIORITY = [
    ("geometry",             ["triangle", "circle", "polygon", "angle", "area",
                              "perimeter", "sphere", "cone", "arc", "chord"]),
    ("polynomial_roots",     ["polynomial", "vieta", "cubic", "quartic",
                              "roots of", "zero of"]),
    ("number_theory",        ["prime", "divisib", "modular", "gcd", "lcm",
                              "diophantine", "congruence", "digit"]),
    ("intermediate_algebra", ["logarithm", "exponential", "complex number",
                              "matrix", "determinant", "progression"]),
    ("algebra",              ["equation", "inequalit", "function", "quadratic",
                              "system of"]),
    ("pre_algebra",          ["percent", "ratio", "proportion", "decimal",
                              "average", "speed", "distance"]),
]

_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|ans)\s*(?:is|:|=)",
    re.IGNORECASE,
)

def _infer_domain(problem: str) -> str:
    text = problem.lower()
    for dom, keywords in _DOMAIN_KEYWORD_PRIORITY:
        if any(kw in text for kw in keywords):
            return dom
    return "algebra"


def _high_confidence_reference(text: object | None) -> str | None:
    """
    Extract a reference answer only when the source text has explicit signals.

    This keeps Math-Verify active while avoiding false labels from long CoT-only
    fields that do not contain a canonical final answer.
    """
    if text is None:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    candidate = normalize_reference_answer(raw)
    if not candidate:
        return None

    # High-confidence signals for canonical final answer presence.
    if "\\boxed{" in raw:
        return candidate
    if _EXPLICIT_ANSWER_RE.search(raw):
        return candidate

    # Single short line (e.g., "13/6", "x=33") is usually safe.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) == 1 and len(lines[0].split()) <= 6:
        return candidate

    return None


def load_s1k(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "simplescaling/s1K",
    split: str = "train",
    max_samples: int | None = None,
    **_kwargs,
) -> Iterator[SeedSample]:
    logger.info("Loading s1K (split={}, cap={})", split, max_samples)

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

        # Filter for math problems only (skip science, crossword, etc.)
        if (row.get("cot_type") or "").lower() != "math":
            continue

        problem = (row.get("question") or "").strip()
        if not problem:
            continue

        solution = (row.get("solution") or "").strip() or None
        trajectories = row.get("thinking_trajectories") or []
        # Keep Math-Verify enabled while preventing false negatives from
        # free-form reasoning fields without a canonical final answer.
        answer = pick_reference_answer(
            row.get("answer"),
            _high_confidence_reference(solution),
            _high_confidence_reference(row.get("attempt")),
            *(_high_confidence_reference(t) for t in trajectories[:2]),
        )

        yield SeedSample(
            id=f"s1k:{split}:{idx}",
            source="s1k",
            domain=_infer_domain(problem),
            problem=problem,
            solution=solution or (trajectories[0].strip() if trajectories else None),
            answer=answer,
            raw={
                "cot_type": row.get("cot_type"),
                "source_type": row.get("source_type"),
                "metadata": row.get("metadata"),
                "attempt": row.get("attempt"),
            },
        )
        count += 1

    logger.info("s1K: yielded {} samples", count)
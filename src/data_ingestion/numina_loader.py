"""
NuminaMath-CoT loader
─────────────────────
Dataset : AI-MO/NuminaMath-CoT  (Apache-2.0, publicly accessible)
HF page : https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
Size    : ~860 k rows, single 'train' split (+ small 'test')

Real schema (confirmed from dataset viewer, March 2026):
  problem   – competition math problem text
  solution  – full chain-of-thought solution (contains \\boxed{answer})
  messages  – list of {role, content} dicts (user=problem, assistant=solution)
  source    – originating corpus, one of:
              aops_forum | amc_aime | cn_k12 | gsm8k | math |
              olympiads  | orca_math | synthetic_amc | synthetic_math

Note: there is NO standalone 'answer' field — the final answer is extracted
from the last \\boxed{} expression inside 'solution'.

Source row-counts (from dataset card):
  cn_k12 276 591  |  synthetic_math 167 895  |  orca_math 153 334
  olympiads 150 581  |  synthetic_amc 62 111  |  aops_forum 30 201
  math 7 478  |  gsm8k 7 345  |  amc_aime 4 072
"""

from __future__ import annotations

import re
from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample

# Real source names as they appear in the dataset (from dataset card breakdown table)
_SOURCE_TO_DOMAIN: dict[str, str] = {
    "amc_aime":      "number_theory",        # AMC / AIME competition problems
    "aops_forum":    "algebra",              # Art of Problem Solving forum
    "cn_k12":        "algebra",              # Chinese high-school curriculum
    "gsm8k":         "pre_algebra",          # Grade-school math word problems
    "math":          "intermediate_algebra", # MATH benchmark dataset
    "olympiads":     "algebra",              # International olympiad problems
    "orca_math":     "pre_algebra",          # Elementary / middle-school math
    "synthetic_amc": "number_theory",        # Synthetically-augmented AMC
    "synthetic_math":"intermediate_algebra", # Synthetically-augmented MATH
}

# Keyword scan refines the source-based domain assignment (priority order)
_DOMAIN_KEYWORD_PRIORITY = [
    ("geometry",             ["triangle", "circle", "polygon", "angle", "area",
                              "perimeter", "coordinate geometry", "sphere", "cone",
                              "rectangle", "rhombus", "tangent", "arc", "chord"]),
    ("polynomial_roots",     ["polynomial", "vieta", "cubic", "quartic",
                              "factor theorem", "zero of", "roots of the polynomial"]),
    ("number_theory",        ["prime", "divisib", "modular", "gcd", "lcm",
                              "factorial", "diophantine", "congruence", "digit"]),
    ("intermediate_algebra", ["logarithm", "exponential", "complex number",
                              "arithmetic progression", "geometric progression",
                              "matrix", "determinant"]),
    ("algebra",              ["equation", "inequalit", "quadratic", "system of",
                              "simplif"]),
    ("pre_algebra",          ["percent", "ratio", "proportion", "decimal",
                              "word problem", "average", "speed", "distance"]),
]

_BOXED_KEY = r"\boxed{"


def _extract_boxed_answer(solution: str) -> str | None:
    """
    Extract the final \\boxed{} answer from the solution text.
    Uses brace-counting so it handles any depth of nested braces,
    e.g. \\boxed{\\frac{\\sqrt{5}}{2}} or \\boxed{-\\frac{\\sqrt{2}}{2}}.
    Returns the content of the last \\boxed{} found, or None.
    """
    last_start = solution.rfind(_BOXED_KEY)
    if last_start < 0:
        return None
    pos = last_start + len(_BOXED_KEY)
    depth = 1
    chars: list[str] = []
    while pos < len(solution) and depth > 0:
        ch = solution[pos]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                chars.append(ch)
        else:
            chars.append(ch)
        pos += 1
    if depth == 0:
        return "".join(chars).strip()
    return None


def _infer_domain(source: str, problem: str) -> str:
    """Domain label: keyword scan takes priority over source-level default."""
    text = problem.lower()
    for dom, keywords in _DOMAIN_KEYWORD_PRIORITY:
        if any(kw in text for kw in keywords):
            return dom
    return _SOURCE_TO_DOMAIN.get(source.lower(), "algebra")


def load_numina(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    split: str = "train",
    max_samples: int | None = None,
    **_kwargs,                   # absorb any extra keys forwarded from config.yaml
) -> Iterator[SeedSample]:
    """
    Yields SeedSample objects from NuminaMath-CoT.

    Parameters
    ----------
    hf_token:    HuggingFace access token (None = anonymous, works for this public dataset)
    cache_dir:   local path to cache downloaded shards
    split:       dataset split (train / test)
    max_samples: cap the number of samples (None = unlimited)
    """
    logger.info("Loading NuminaMath-CoT (split={}, cap={})", split, max_samples)

    load_kwargs: dict = dict(
        path="AI-MO/NuminaMath-CoT",
        split=split,
        streaming=True,           # stream to avoid downloading all 860k rows at once
        trust_remote_code=False,
    )
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = load_dataset(**load_kwargs)

    count = 0
    for idx, row in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break

        problem  = (row.get("problem")  or "").strip()
        solution = (row.get("solution") or "").strip()
        source   = (row.get("source")   or "unknown").strip()

        if not problem:
            continue   # skip malformed rows

        # No standalone 'answer' field — extract final \boxed{} from solution
        answer = _extract_boxed_answer(solution) if solution else None
        domain = _infer_domain(source, problem)

        yield SeedSample(
            id=f"numina_math:{split}:{idx}",
            source="numina_math",
            domain=domain,
            problem=problem,
            solution=solution or None,
            answer=answer,
            raw={"numina_source": source, "split": split, "row_idx": idx},
        )
        count += 1

    logger.info("NuminaMath-CoT: yielded {:,} samples", count)

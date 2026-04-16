"""
TheoremQA loader
────────────────
Dataset : TIGER-Lab/TheoremQA
HF page : https://huggingface.co/datasets/TIGER-Lab/TheoremQA

Observed schema (March 2026):
  Question     – QA prompt
  Answer       – ground truth answer
  Answer_type  – answer format/type
  Picture      – optional image/context field

Notes:
  - Public split appears as test (800 rows).
  - No full reference solution chain is available in the row schema.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample
from .answer_utils import pick_reference_answer

_DOMAIN_KEYWORD_PRIORITY = [
    ("geometry",             ["triangle", "circle", "polygon", "angle", "area", "perimeter"]),
    ("number_theory",        ["prime", "divisib", "mod", "integer", "factor"]),
    ("intermediate_algebra", ["log", "exponential", "matrix", "determinant", "probability"]),
    ("algebra",              ["equation", "function", "sequence", "polynomial"]),
]


def _infer_domain(question: str) -> str:
    text = question.lower()
    for dom, keywords in _DOMAIN_KEYWORD_PRIORITY:
        if any(kw in text for kw in keywords):
            return dom
    return "algebra"


def load_theoremqa(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "TIGER-Lab/TheoremQA",
    split: str = "test",
    max_samples: int | None = None,
    math_only: bool = True,
    **_kwargs,
) -> Iterator[SeedSample]:
    logger.info("Loading TheoremQA (split={}, cap={}, math_only={})", split, max_samples, math_only)

    load_kwargs: dict = dict(
        path=repo_id,
        split=split,
        trust_remote_code=False,
        cache_dir=cache_dir,
    )
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = load_dataset(**load_kwargs)

    # First pass: try math-only filter
    samples = []
    math_count = 0
    
    for idx, row in enumerate(ds):
        if max_samples is not None and len(samples) >= max_samples:
            break

        # Try to find subject field
        subject = None
        for field in ["Subject", "subject", "domain", "Domain", "category", "Category"]:
            if field in row:
                subject = (row.get(field) or "").strip().lower()
                break
        
        problem = (row.get("Question") or "").strip()
        if not problem:
            continue

        answer = pick_reference_answer(row.get("Answer"))

        # If math_only, check subject; otherwise collect all
        if math_only and subject and subject != "math":
            continue
        
        if math_only and subject == "math":
            math_count += 1

        samples.append({
            "id": f"theoremqa:{split}:{idx}",
            "source": "theoremqa",
            "domain": _infer_domain(problem),
            "problem": problem,
            "answer": answer,
            "subject": subject,
            "answer_type": row.get("Answer_type"),
        })

    # If math filter yielded 0 samples, reload without filter
    if math_only and math_count == 0:
        logger.warning("TheoremQA: math_only=True yielded 0 samples, loading all samples instead")
        samples = []
        for idx, row in enumerate(ds):
            if max_samples is not None and len(samples) >= max_samples:
                break

            problem = (row.get("Question") or "").strip()
            if not problem:
                continue

            answer = pick_reference_answer(row.get("Answer"))

            samples.append({
                "id": f"theoremqa:{split}:{idx}",
                "source": "theoremqa",
                "domain": _infer_domain(problem),
                "problem": problem,
                "answer": answer,
                "subject": (row.get("Subject") or "").strip(),
                "answer_type": row.get("Answer_type"),
            })

    # Yield collected samples
    for sample_dict in samples:
        yield SeedSample(
            id=sample_dict["id"],
            source=sample_dict["source"],
            domain=sample_dict["domain"],
            problem=sample_dict["problem"],
            solution=None,
            answer=sample_dict["answer"],
            raw={
                "subject": sample_dict["subject"],
                "answer_type": sample_dict["answer_type"],
                "picture": None,  # Exclude image data - not JSON serializable
                "split": split,
                "category": "SFT",  # Mark as SFT, not reasoning
            },
        )

    logger.info("TheoremQA: yielded {} samples", len(samples))

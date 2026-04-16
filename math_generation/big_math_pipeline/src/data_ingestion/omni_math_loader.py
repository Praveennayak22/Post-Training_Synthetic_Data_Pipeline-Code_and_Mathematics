"""
Omni-MATH loader
─────────────────
Dataset : KbsdJames/Omni-MATH
GitHub  : https://github.com/KbsdJames/Omni-MATH
Paper   : https://arxiv.org/abs/2410.07985

Description:
  Universal Olympiad-level mathematics benchmark with 4,428 competition-level
  problems across 33+ sub-domains and 10 difficulty levels (1.0–9.5).

Observed schema (March 2026):
  domain      – list of hierarchical category strings (e.g., ["Mathematics -> Algebra -> ..."])
  difficulty  – float difficulty rating (typically 1.0–9.5)
  problem     – problem statement (LaTeX-formatted)
  solution    – detailed solution / reasoning (LaTeX-formatted)
  answer      – final answer (may be symbolic or numeric)
  source      – competition source (e.g., "china_team_selection_test")
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .answer_utils import pick_reference_answer
from .base import SeedSample


def _extract_primary_domain(domain_list: list | None) -> str:
    """
    Extract primary domain from hierarchical category list.
    
    Example:
      ["Mathematics -> Algebra -> Other"]  → "algebra"
      ["Mathematics -> Geometry -> ..."]   → "geometry"
    """
    if not domain_list or not isinstance(domain_list, list) or not domain_list[0]:
        return "algebra"  # safe default
    
    primary = domain_list[0].lower()
    
    # Map second-level categories to standard domains
    if "geometry" in primary:
        return "geometry"
    elif "number theory" in primary or "number_theory" in primary:
        return "number_theory"
    elif "algebra" in primary:
        return "algebra"
    elif "combinatorics" in primary or "combinatorial" in primary:
        return "combinatorics"
    elif "discrete" in primary:
        return "discrete_math"
    elif "trigonometry" in primary:
        return "trigonometry"
    elif "probability" in primary or "statistics" in primary:
        return "probability"
    elif "calculus" in primary:
        return "calculus"
    elif "logic" in primary:
        return "logic"
    
    # Default: extract second part of hierarchy if available
    parts = domain_list[0].split("->")
    if len(parts) > 1:
        second_part = parts[1].strip().lower().replace(" ", "_")
        return second_part
    
    return "algebra"


def load_omni_math(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "KbsdJames/Omni-MATH",
    split: str = "test",
    max_samples: int | None = None,
    **_kwargs,
) -> Iterator[SeedSample]:
    """
    Load Omni-MATH dataset samples as SeedSample objects.
    
    Parameters
    ----------
    hf_token : str, optional
        HuggingFace API token for authentication
    cache_dir : str
        Directory to cache downloaded dataset (default: "data/hf_cache")
    repo_id : str
        HuggingFace dataset repository ID (default: "KbsdJames/Omni-MATH")
    split : str
        Dataset split to load (default: "test"; only split available)
    max_samples : int, optional
        Maximum number of samples to load (None = load all)
    
    Yields
    ------
    SeedSample
        Normalized seed samples ready for downstream processing
    """
    logger.info("Loading Omni-MATH (repo={}, split={}, cap={})", repo_id, split, max_samples)
    
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
        
        # Extract and sanitize fields
        problem = row.get("problem", "").strip() if row.get("problem") else None
        if not problem:
            logger.debug("Skipping row {}: empty problem", idx)
            continue
        
        solution = row.get("solution", "").strip() if row.get("solution") else None
        answer = pick_reference_answer(
            row.get("answer", "").strip() if row.get("answer") else None,
            solution,
        )
        
        domain = _extract_primary_domain(row.get("domain"))
        difficulty = row.get("difficulty", 5.0)  # midpoint default
        source_info = row.get("source", "unknown")
        
        sample_id = f"{source_info}:{idx}"
        
        yield SeedSample(
            id=f"omni_math:{sample_id}",
            source="omni_math",
            domain=domain,
            problem=problem,
            solution=solution,
            answer=answer,
            raw={
                "repo_id": repo_id,
                "split": split,
                "row_id": idx,
                "difficulty": difficulty,
                "source_competition": source_info,
                "domain_hierarchy": row.get("domain"),
            },
        )
        
        count += 1
    
    logger.info("Loaded {} Omni-MATH samples", count)

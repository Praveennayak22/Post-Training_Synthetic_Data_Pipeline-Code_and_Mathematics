"""
DeepMath-103K loader
────────────────────
Dataset : zwhe99/DeepMath-103K
HF page : https://huggingface.co/datasets/zwhe99/DeepMath-103K

Observed schema (April 2026):
  question      – mathematical problem statement
  final_answer  – verifiable final answer
  difficulty    – numerical difficulty score (1-9)
  topic         – hierarchical topic classification
  r1_solutions  – list of three distinct DeepSeek-R1 reasoning paths
  tags          – metadata tags
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .answer_utils import pick_reference_answer
from .base import SeedSample

_TOPIC_TO_DOMAIN = {
    "algebra": "algebra",
    "calculus": "calculus",
    "number theory": "number_theory",
    "number_theory": "number_theory",
    "geometry": "geometry",
    "probability": "probability",
    "discrete mathematics": "discrete_math",
    "discrete_mathematics": "discrete_math",
    "pre-algebra": "pre_algebra",
    "pre_algebra": "pre_algebra",
    "analysis": "calculus",
    "combinatorics": "discrete_math",
    "trigonometry": "geometry",
}


def _infer_domain(topic: str | None) -> str:
    """Map topic field to domain, with fallback."""
    if not topic:
        return "algebra"
    normalized = topic.strip().lower()
    return _TOPIC_TO_DOMAIN.get(normalized, "algebra")


def _extract_solution(r1_solutions: list | None) -> str | None:
    """
    Select the longest (most detailed) DeepSeek-R1 solution from the 3 reasoning paths.
    r1_solutions is typically a list of 3 distinct reasoning paths.
    Returns the solution with maximum detail/length.
    """
    if not r1_solutions or not isinstance(r1_solutions, list):
        return None
    
    # Filter to only valid string solutions
    valid_solutions = [
        sol.strip() for sol in r1_solutions 
        if isinstance(sol, str) and sol.strip()
    ]
    
    if not valid_solutions:
        return None
    
    # Return the longest solution (most detailed reasoning)
    best_solution = max(valid_solutions, key=len)
    return best_solution


def load_deepmath(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "zwhe99/DeepMath-103K",
    split: str = "train",
    max_samples: int | None = None,
    chunk_id: int | None = None,
    total_chunks: int | None = None,
    **_kwargs,
) -> Iterator[SeedSample]:
    """
    Load DeepMath-103K dataset from Hugging Face.
    
    Each sample contains:
    - question: math problem statement
    - final_answer: verifiable numeric/symbolic answer
    - r1_solutions: list of 3 DeepSeek-R1 reasoning paths
    - topic: hierarchical topic classification
    - difficulty: numerical score (1-9)
    
    We select the LONGEST (most detailed) r1_solution as the reference solution
    and set source='deepmath' to mark data as pre-solved (no teacher generation needed).
    
    Parameters
    ----------
    chunk_id : int, optional
        Chunk identifier (0 to total_chunks-1) for parallel processing
    total_chunks : int, optional
        Total number of chunks. If provided, only loads samples for this chunk.
    """
    logger.info("Loading DeepMath-103K (split={}, cap={}, chunk={}/{})", split, max_samples, chunk_id, total_chunks)

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
        # Chunk filtering for parallel processing
        if total_chunks is not None and chunk_id is not None:
            if idx % total_chunks != chunk_id:
                continue
        
        if max_samples is not None and count >= max_samples:
            break

        # Extract fields from row
        problem = row.get("question", "").strip() if row.get("question") else None
        if not problem:
            logger.warning("Row {} has no question; skipping", idx)
            continue

        # Use first DeepSeek-R1 solution as reference solution
        solution = _extract_solution(row.get("r1_solutions"))
        
        # Use final_answer field directly
        answer = row.get("final_answer", "").strip() if row.get("final_answer") else None
        if not answer:
            answer = pick_reference_answer(None, solution)
        
        # Extract topic/domain
        topic = row.get("topic", "algebra")
        domain = _infer_domain(topic)

        sample_id = f"{split}:{idx}"

        yield SeedSample(
            id=f"deepmath:{sample_id}",
            source="deepmath",
            domain=domain,
            problem=problem,
            solution=solution,
            answer=answer,
            raw={
                "repo_id": repo_id,
                "split": split,
                "row_id": sample_id,
                "topic": topic,
                "difficulty": row.get("difficulty"),
                "num_r1_solutions": len(row.get("r1_solutions", [])),
            },
        )

        count += 1

    logger.info("Finished loading {} DeepMath samples", count)

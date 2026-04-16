"""
Math Calculator Tool Dataset loader
────────────────────────────────────
Dataset : archit11/math-calculator-tool-dataset-gpt4o-mini
HF page : https://huggingface.co/datasets/archit11/math-calculator-tool-dataset-gpt4o-mini

Overview:
  Math problems with calculator-based verification and reasoning traces.
  Generated using GPT-4O-mini with tool-use integration for numerical validation.
  Focus on problems where intermediate steps have calculable values.

Format:
  Chat-based format with messages list containing:
  - system: mathematics expert system prompt
  - user: the actual math problem/question  
  - assistant: step-by-step solution with reasoning and tool calls
"""

from __future__ import annotations

import re
from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample


def _extract_from_messages(messages: list[dict]) -> tuple[str | None, str | None, str | None]:
    """
    Extract problem, solution, and answer from messages list.
    Returns: (problem, solution, answer)
    """
    problem = None
    solution = None
    answer = None
    
    if not messages or not isinstance(messages, list):
        return None, None, None
    
    # Extract user message (problem)  
    for msg in messages:
        if msg.get('role') == 'user':
            problem = (msg.get('content') or '').strip()
            # Remove HTML tags if present
            problem = re.sub(r'<[^>]+>', '', problem).strip()
            break
    
    # Extract assistant message (solution)
    for msg in messages:
        if msg.get('role') == 'assistant':
            solution = (msg.get('content') or '').strip()
            break
    
    # Try to extract answer from solution
    if solution:
        # Look for boxed answer pattern
        match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if match:
            answer = match.group(1).strip()
        else:
            # Look for final answer patterns
            match = re.search(r"(?:final\s+)?answer[:\s]+([^\n.]+)", solution, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
    
    return problem, solution, answer


def _infer_domain_from_problem(problem: str) -> str:
    """Infer math domain from problem content."""
    if not problem:
        return "algebra"
    
    lower = problem.lower()
    
    # Domain keyword matching (priority order)
    domain_keywords = [
        ("geometry", ["triangle", "circle", "polygon", "angle", "area", 
                      "perimeter", "coordinate", "sphere", "cone", "rectangle", 
                      "rhombus", "tangent", "arc", "chord"]),
        ("number_theory", ["prime", "divisib", "modular", "gcd", "lcm",
                          "factorial", "diophantine", "congruence", "digit"]),
        ("intermediate_algebra", ["logarithm", "exponential", "complex", 
                                 "progression", "matrix", "determinant"]),
        ("pre_algebra", ["percent", "ratio", "proportion", "decimal",
                        "word problem", "average", "speed", "distance"]),
        ("algebra", ["equation", "inequalit", "quadratic", "system", "polynomial"]),
    ]
    
    for domain, keywords in domain_keywords:
        if any(kw in lower for kw in keywords):
            return domain
    
    return "algebra"


def load_math_calculator_tool(
    hf_token: str | None = None,
    cache_dir: str = "data/hf_cache",
    repo_id: str = "archit11/math-calculator-tool-dataset-gpt4o-mini",
    split: str = "train",
    max_samples: int | None = None,
    **_kwargs,
) -> Iterator[SeedSample]:
    """
    Yields SeedSample objects from math-calculator-tool dataset.
    
    The dataset uses a chat format with system, user, and assistant messages.
    This loader extracts the problem from user messages and solution from assistant messages.

    Parameters
    ----------
    hf_token:    HuggingFace access token (None = anonymous, public dataset)
    cache_dir:   local path to cache downloaded data
    repo_id:     HuggingFace repo identifier
    split:       dataset split (train / test / validation)
    max_samples: cap the number of samples (None = unlimited)
    """
    logger.info(
        "Loading math-calculator-tool dataset (repo={}, split={}, cap={})",
        repo_id, split, max_samples
    )

    load_kwargs: dict = dict(
        path=repo_id,
        split=split,
        trust_remote_code=False,
        cache_dir=cache_dir,
    )
    if hf_token:
        load_kwargs["token"] = hf_token

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as exc:
        logger.error("Failed to load {}: {}", repo_id, exc)
        return

    count = 0
    skipped = 0
    
    for idx, row in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break

        # Extract from messages format
        messages = row.get("messages")
        if not messages:
            skipped += 1
            continue
        
        problem, solution, answer = _extract_from_messages(messages)
        
        if not problem:
            skipped += 1
            continue

        # Infer domain from problem content
        domain = _infer_domain_from_problem(problem)

        yield SeedSample(
            id=f"math_calculator_tool:{split}:{idx}",
            source="math_calculator_tool",
            domain=domain,
            problem=problem,
            solution=solution,
            answer=answer,
            raw={
                "messages_count": len(messages) if messages else 0,
                "has_assistant_message": any(msg.get('role') == 'assistant' for msg in (messages or [])),
            },
        )
        count += 1

    logger.info(
        "math-calculator-tool: yielded {} samples ({} skipped)",
        count, skipped
    )

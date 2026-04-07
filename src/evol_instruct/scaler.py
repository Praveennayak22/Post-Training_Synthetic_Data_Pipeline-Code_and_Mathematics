"""
Evol-Instruct difficulty scaler for math problems.
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Implements WizardMath â€œReinforced Evol-Instructâ€ adapted for our pipeline.
Reference: https://github.com/nlpxucan/WizardLM  (WizardMath directory)
Paper    : https://arxiv.org/abs/2308.09583

WizardMath uses 5 evolution operations:
  1. add_constraint   â€“ add a meaningful constraint that changes the solution path
  2. replace_numbers  â€“ replace simple numbers with complex ones (âˆš2, Ï€, e, fractions)
  3. shift_type       â€“ change the problem type (e.g., algebraic â†’ geometric)
  4. increase_steps   â€“ require more reasoning steps to reach the solution
  5. combine_problems â€“ combine two related problems into one harder problem

Prompt format (official WizardMath)::

    Below is an instruction that describes a task. Write a response that
    appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:

Each mutation sends its instruction to the model in this format.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

import httpx
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.data_ingestion.base import SeedSample

# â”€â”€ Mutation strategy type â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MutationStrategy = Literal[
    "add_constraint",
    "replace_numbers",
    "shift_type",
    "increase_steps",
    "combine_problems",
]

# â”€â”€ Official WizardMath prompt wrapper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# From: https://github.com/nlpxucan/WizardLM  (WizardMath/README.md)
_WIZARDMATH_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:"
)

# â”€â”€ Mutation instructions (the {instruction} slot in WizardMath template) â”€â”€â”€â”€â”€â”€â”€
_MUTATION_INSTRUCTIONS: dict[str, str] = {
    "add_constraint": (
        "I have a math problem below. Please rewrite it to be MORE DIFFICULT by "
        "adding one meaningful mathematical constraint that forces a more sophisticated "
        "solution technique. The constraint must genuinely change the solution path.\n\n"
        "Rules:\n- Keep the problem self-contained and unambiguous.\n"
        "- Do NOT solve the problem.\n"
        "- Output ONLY the rewritten problem statement.\n\nOriginal problem:\n{problem}"
    ),
    "replace_numbers": (
        "I have a math problem below. Please rewrite it to be MORE DIFFICULT by "
        "replacing specific integer values with irrational or complex numbers "
        "(e.g., \u221a2, \u221a3, \u03c0, e, fractions with large denominators).\n\n"
        "Rules:\n- The substitution must keep the problem solvable but require more careful computation.\n"
        "- Do NOT solve the problem.\n"
        "- Output ONLY the rewritten problem statement.\n\nOriginal problem:\n{problem}"
    ),
    "shift_type": (
        "I have a math problem below. Please rewrite it by CHANGING THE PROBLEM TYPE "
        "while preserving the core mathematical idea "
        "(e.g. algebraic \u2192 geometric, discrete \u2192 continuous, computation \u2192 proof).\n\n"
        "Rules:\n- The new problem must require a different but related math area.\n"
        "- Do NOT solve the problem.\n"
        "- Output ONLY the rewritten problem statement.\n\nOriginal problem:\n{problem}"
    ),
    "increase_steps": (
        "I have a math problem below. Please rewrite it to require SIGNIFICANTLY MORE "
        "reasoning steps (add intermediate sub-goals, use induction, or introduce multiple cases).\n\n"
        "Rules:\n- The core mathematical topic must be preserved.\n"
        "- Do NOT solve the problem.\n"
        "- Output ONLY the rewritten problem statement.\n\nOriginal problem:\n{problem}"
    ),
    "combine_problems": (
        "I have a math problem below. Please create a NEW, HARDER problem that COMBINES "
        "it with concepts from {second_domain} mathematics so both areas are essential.\n\n"
        "Rules:\n- Both areas must be substantially present.\n"
        "- The combined problem must remain solvable in principle.\n"
        "- Do NOT solve the problem.\n"
        "- Output ONLY the new problem statement.\n\nOriginal problem:\n{problem}"
    ),
}

_DOMAIN_PAIRS: dict[str, list[str]] = {
    "algebra":              ["geometry", "number_theory", "polynomial_roots"],
    "geometry":             ["algebra", "intermediate_algebra"],
    "number_theory":        ["algebra", "polynomial_roots"],
    "intermediate_algebra": ["algebra", "geometry"],
    "pre_algebra":          ["algebra", "geometry"],
    "polynomial_roots":     ["algebra", "number_theory"],
}

_LEAK_PATTERNS_SCALER: tuple[re.Pattern[str], ...] = (
    re.compile(r"the user wants me to rewrite", re.IGNORECASE),
    re.compile(r"\boriginal problem\b", re.IGNORECASE),
    re.compile(r"rewrite (it )?to be more difficult", re.IGNORECASE),
    re.compile(r"add(?:ing)? one meaningful mathematical constraint", re.IGNORECASE),
    re.compile(r"\bdo not solve\b", re.IGNORECASE),
    re.compile(r"\boutput only\b", re.IGNORECASE),
)


def _clean_mutation_output(text: str) -> str:
    """
    Clean mutation model output by:
    1. Stripping wrapper formats (```markdown, ### headers, etc.)
    2. Removing instruction echos and meta-text
    3. Extracting the actual problem statement
    
    Returns cleaned problem text, or empty string if cleaning fails.
    """
    if not text:
        return ""
    
    # Remove markdown code fences
    text = re.sub(r"^```+\w*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```+$", "", text, flags=re.MULTILINE)
    
    # Remove section headers commonly added by models
    text = re.sub(r"### Instruction:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"### Response:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"### Rewritten Problem:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"### Original Problem:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Rewritten Problem:\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Split by lines and filter out instruction meta-text
    lines = text.split("\n")
    cleaned_lines = []
    skip_until_empty = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip lines that are pure instruction meta-text
        if any(p.search(stripped) for p in _LEAK_PATTERNS_SCALER):
            skip_until_empty = True
            continue
        
        # Skip empty lines after leak pattern lines (separator)
        if skip_until_empty and not stripped:
            skip_until_empty = False
            continue
        
        if stripped:
            skip_until_empty = False
            cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines).strip()
    
    # If result is too short (<10 chars), likely malformed
    if len(result) < 10:
        return ""
    
    return result
# â”€â”€ Output dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class EvolvedSample:
    """A problem that has been mutated by Evol-Instruct."""
    parent_id: str                     # id of the SeedSample it came from
    evolved_id: str                    # unique id for this evolved variant
    source: str                        # inherited from seed
    domain: str | None
    mutation_strategy: str
    mutation_depth: int                # 1 = first mutation, 2 = second, ...
    problem: str                       # evolved problem text
    original_problem: str              # seed / parent problem text
    solution: str | None = None        # filled in by teacher model step
    answer: str | None = None
    reasoning_level: int | None = None
    metadata: dict = field(default_factory=dict, repr=False)


# â”€â”€ API call with retry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@retry(
    retry=retry_if_exception_type((
        httpx.HTTPStatusError,
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    )),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    stop=stop_after_attempt(2),
    reraise=True,
)
def _call_openai_compat(
    endpoint: str,
    api_key: str | None,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    model: str = "glm-5-fp8",
    timeout: float = 90.0,
) -> str:
    """
    Call any OpenAI-compatible chat completions endpoint.
    Sends enable_thinking=false so Evol-Instruct mutations return
    clean rewritten problem text without a reasoning trace.
    """
    base = endpoint.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        target = base
    elif base.endswith("/v1"):
        target = f"{base}/chat/completions"
    else:
        target = f"{base}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(target, headers=headers, json=payload)
        response.raise_for_status()

    data = response.json()
    # OpenAI-compatible response shape, but some backends may emit
    # content=None and put text in reasoning_content or structured parts.
    if "choices" in data and data["choices"]:
        choice = data["choices"][0] or {}
        msg = choice.get("message") or {}
        reasoning = ""
        answer = ""

        if isinstance(msg, dict):
            reasoning = str(msg.get("reasoning_content") or "").strip()
            content = msg.get("content")
            if isinstance(content, str):
                answer = content.strip()
            elif isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str):
                        parts.append(item)
                answer = "\n".join(parts).strip()
            elif content is not None:
                answer = str(content).strip()

        # Some providers return plain completion text at choices[0].text.
        if not answer and not reasoning and choice.get("text"):
            answer = str(choice.get("text")).strip()

        merged = answer or reasoning
        if merged:
            return merged
        raise RuntimeError("Empty model response content in choices[0]")
    return str(data).strip()


# â”€â”€ Main scaler class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class EvolInstructScaler:
    """
    Applies Evol-Instruct mutations to seed SeedSamples.

    Parameters
    ----------
    cfg : full pipeline config dict (from config/config.yaml)
    """

    def __init__(self, cfg: dict) -> None:
        self._api_key    = (os.environ.get("MODEL_API_KEY") or "").strip()
        ecfg             = cfg["evol_instruct"]
        self._endpoint   = os.environ.get("MODEL_ENDPOINT") or cfg["teacher_model"]["endpoint"]
        self._model      = ecfg.get("model", "glm-5-fp8")
        self._temperature   = ecfg["temperature"]
        self._max_new_tokens = ecfg["max_new_tokens"]
        self._num_mutations  = ecfg["num_mutations"]
        self._strategies: list[str] = ecfg["mutation_strategies"]
        self._offline_mode = bool(ecfg.get("offline_mode", False))
        if not self._api_key:
            logger.warning(
                "MODEL_API_KEY not set; Evol-Instruct requests will be sent without Authorization header."
            )

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def evolve(self, seed: SeedSample) -> list[EvolvedSample]:
        """
        Apply all configured mutation strategies to a single seed sample.

        Returns a list of EvolvedSample objects (one per strategy Ã— depth).
        Multiple depths chain mutations: mutation N+1 uses output of mutation N.
        """
        results: list[EvolvedSample] = []
        current_problem = seed.problem

        for depth in range(1, self._num_mutations + 1):
            for strategy in self._strategies:
                evolved_text = self._apply_mutation(
                    problem=current_problem,
                    strategy=strategy,
                    domain=seed.domain or "algebra",
                )
                if evolved_text and evolved_text != current_problem:
                    es = EvolvedSample(
                        parent_id=seed.id,
                        evolved_id=f"{seed.id}::evol::{strategy}::d{depth}",
                        source=seed.source,
                        domain=seed.domain,
                        mutation_strategy=strategy,
                        mutation_depth=depth,
                        problem=evolved_text,
                        original_problem=seed.problem,
                        # The mutation changes the problem, so inherited references
                        # from the seed are no longer valid for strict verification.
                        solution=None,
                        answer=None,
                    )
                    results.append(es)

            # For depth > 1: chain â€” next round mutates the last successful output
            if results:
                current_problem = results[-1].problem

        return results

    def evolve_batch(self, seeds: list[SeedSample]) -> list[EvolvedSample]:
        """Evolve a list of seed samples and return all evolved variants."""
        all_evolved: list[EvolvedSample] = []
        for i, seed in enumerate(seeds):
            logger.info("Evolving seed {}/{}: {}", i + 1, len(seeds), seed.id)
            try:
                evolved = self.evolve(seed)
                all_evolved.extend(evolved)
            except Exception as exc:
                logger.warning("Failed to evolve seed {}: {}", seed.id, exc)
                continue
        logger.info("Total evolved samples: {}", len(all_evolved))
        return all_evolved

    # â”€â”€ private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _apply_mutation(
        self,
        problem: str,
        strategy: str,
        domain: str,
    ) -> str | None:
        instruction_template = _MUTATION_INSTRUCTIONS.get(strategy)
        if instruction_template is None:
            logger.warning("Unknown mutation strategy: {}", strategy)
            return None

        # Pick a complementary domain for combine_problems
        second_domain = "geometry"
        if strategy == "combine_problems":
            candidates = _DOMAIN_PAIRS.get(domain, ["geometry"])
            second_domain = candidates[0]

        # Fast local mode for test runs: avoid remote model calls.
        if self._offline_mode:
            if strategy == "add_constraint":
                return f"{problem}\n\nAdditional constraint: Provide the final answer in simplest exact form."
            if strategy == "replace_numbers":
                return (
                    problem
                    .replace(" 1 ", " 2 ")
                    .replace(" 2 ", " 3 ")
                    .replace(" 3 ", " 5 ")
                )
            if strategy == "shift_type":
                return f"Reframe this as a {second_domain} interpretation of the same core problem:\n{problem}"
            if strategy == "increase_steps":
                return f"Solve this in three explicit stages and justify each stage:\n{problem}"
            if strategy == "combine_problems":
                return f"{problem}\n\nAdditionally, connect your reasoning to one concept from {second_domain}."

        # Fill instruction, then wrap in official WizardMath template
        instruction = instruction_template.format(
            problem=problem,
            second_domain=second_domain,
        )
        prompt = _WIZARDMATH_TEMPLATE.format(instruction=instruction)

        try:
            raw_response = _call_openai_compat(
                endpoint=self._endpoint,
                api_key=self._api_key,
                prompt=prompt,
                temperature=self._temperature,
                max_new_tokens=self._max_new_tokens,
                model=self._model,
            )
            # Clean the mutation output to remove instruction echos and wrappers
            cleaned_response = _clean_mutation_output(raw_response)
            if not cleaned_response:
                logger.warning(
                    "Mutation '{}' output was malformed or empty after cleaning. "
                    "Raw: '{}'",
                    strategy, raw_response[:100] if raw_response else "(empty)",
                )
                return None
            return cleaned_response
        except Exception as exc:
            logger.error(
                "Mutation '{}' failed for problem excerpt '{}...': {}",
                strategy, problem[:60], exc,
            )
            return None


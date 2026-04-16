"""
MCQ â†’ Open-Ended format converter (Big-Math signals approach).
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Mirrors the rule-based *signal* framework from the Big-Math repository
(https://github.com/SynthLabsAI/big-math) which classifies each problem
by its answer type before deciding what transformation (if any) to apply:

  signals/
    mcq_signal.py        â†’ labelled (A)/(B)/(C)/(D) choice problems
    proof_signal.py      â†’ "prove that â€¦", "show that â€¦"
    true_false_signal.py â†’ "True or False: â€¦"
    yes_no_signal.py     â†’ "Is it true that â€¦?  Answer Yes or No."

  reformulation/
    mcq_reformulator.py  â†’ strips choices, rewrites MCQ-framed stem
    tf_reformulator.py   â†’ rewrites as "Determine and justify â€¦"
    yn_reformulator.py   â†’ same

Transformation rules used here
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  MCQ   â†’ strip labelled options, optionally use LLM to clean MCQ phrasing
  Proof â†’ pass through unchanged (already open-ended)
  T/F   â†’ rewrite to "Determine whether X is true or false, with full justification."
  Y/N   â†’ rewrite to "Determine whether X. Justify your answer."
  Other â†’ pass through unchanged (already open-ended numeric/expression problem)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Union

import httpx
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.data_ingestion.base import SeedSample
from src.evol_instruct.scaler import EvolvedSample

ProblemLike = Union[SeedSample, EvolvedSample]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SIGNALS  (mirrors big-math/signals/)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â”€â”€ MCQ signal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Big-Math looks for 2+ labelled options: (A)â€¦(E) or A. B. C. D. patterns.
# We also match single-line multi-choice "A) 3  B) 6  C) 9 â€¦" format.
_MCQ_BLOCK_RE = re.compile(
    r"""
    (?:^|\n)                          # start of line
    \s*
    (?:[A-E][.):]\s*.+?){2,}         # 2+ labelled choices on the SAME line
    |
    (?:^|\n)                          # OR: one choice per line (â‰¥2 lines)
    (?:\s*[\(\[]\s*[A-E]\s*[\)\]]\s*.+\n?){2,}
    |
    (?:^|\n)
    (?:\s*[A-E]\s*[.):\-]\s+.+\n?){2,}
    """,
    re.VERBOSE | re.MULTILINE | re.IGNORECASE,
)

_MCQ_HINT_PHRASES = re.compile(
    r"\b(which of the following|select the (?:best|correct)|choose the (?:best|correct)|"
    r"all of the following|none of the above|each of the following)\b",
    re.IGNORECASE,
)

# â”€â”€ Proof signal (big-math/signals/proof_signal.py analogue) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_PROOF_RE = re.compile(
    r"\b(prove(?: that)?|show(?: that)?|demonstrate(?: that)?|verify(?: that)?|"
    r"establish(?: that)?|disprove|find a proof)\b",
    re.IGNORECASE,
)

# â”€â”€ True/False signal (big-math/signals/true_false_signal.py analogue) â”€â”€â”€â”€â”€â”€â”€
_TRUE_FALSE_RE = re.compile(
    r"\b(true or false|true\/false|state(?: whether)?.*true or false|"
    r"determine.*true or false|is the following statement true|"
    r"is this statement true)\b",
    re.IGNORECASE,
)

# â”€â”€ Yes/No signal (big-math/signals/yes_no_signal.py analogue) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_YES_NO_RE = re.compile(
    r"\b(answer yes or no|answer with yes or no|"
    r"does .+ exist|is it (?:true|possible|necessary|sufficient) that|"
    r"can we|must .+ be)\b[\s\S]{0,80}[?]",
    re.IGNORECASE,
)


def _detect_problem_type(text: str | None) -> str:
    """
    Classify a problem text using Big-Math-style rule-based signals.

    Returns one of: "mcq" | "proof" | "true_false" | "yes_no" | "open"
    """
    if not text:
        return "open"  # safe default for None/empty
    if _MCQ_BLOCK_RE.search(text) or _MCQ_HINT_PHRASES.search(text):
        return "mcq"
    if _PROOF_RE.search(text):
        return "proof"
    if _TRUE_FALSE_RE.search(text):
        return "true_false"
    if _YES_NO_RE.search(text):
        return "yes_no"
    return "open"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  REFORMULATORS  (mirrors big-math/reformulation/)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _strip_mcq_choices(text: str | None) -> str:
    """Remove the labelled choice block and trailing MCQ-framed phrasing."""
    if not text:
        return ""  # safe default for None/empty
    match = _MCQ_BLOCK_RE.search(text)
    if match:
        text = text[: match.start()].rstrip()
    # Remove inline trailing option reference on the last sentence
    text = re.sub(r"\s*[\(\[]\s*[A-E]\s*[\)\]]\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _reformulate_true_false(text: str) -> str:
    """
    Strip the True/False framing and wrap in a 'Determine whether â€¦' instruction.
    Mirrors big-math/reformulation/tf_reformulator.py.
    """
    # Remove "True or False:" prefix if present
    cleaned = re.sub(
        r"^\s*(?:true or false\s*[:\-]?\s*|state whether the following is true or false\s*[:\-]?\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return f"Determine whether the following statement is true or false, and provide a full justification:\n\n{cleaned}"


def _reformulate_yes_no(text: str) -> str:
    """
    Strip Yes/No answer instruction and wrap in an open justification request.
    Mirrors big-math/reformulation/yn_reformulator.py.
    """
    cleaned = re.sub(
        r"\s*[\.\,\;]?\s*(?:answer yes or no|answer with yes or no)[\.!]?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return f"Determine whether the following is true or false, and justify your answer:\n\n{cleaned}"


# â”€â”€ LLM stem rewrite (for MCQ â†’ open-ended phrasing cleanup) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_REWRITE_PROMPT = """\
You are editing a math problem that was originally in multiple-choice format.
The answer options have already been removed. Your task is to rewrite ONLY the \
question stem so that it reads naturally as an open-ended problem â€” the solver \
must derive and state their own answer.

Rules:
- Remove phrases like "which of the following", "select the best answer", etc.
- Replace them with direct instructions like "Find", "Determine", "Prove", or \
"Calculate" as appropriate.
- Do NOT add new mathematical content or change the problem's intent.
- Output ONLY the rewritten question stem. No explanations.

Question stem (options removed):
{stem}

Open-ended rewrite:"""


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _rewrite_stem_via_llm(
    endpoint: str,
    api_key: str | None,
    stem: str,
    model: str = "glm-5-fp8",
    timeout: float = 90.0,
) -> str:
    """Rewrite an MCQ question stem via any OpenAI-compatible chat completions endpoint."""
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
        "messages": [{"role": "user", "content": _REWRITE_PROMPT.format(stem=stem)}],
        "temperature": 0.4,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(target, headers=headers, json=payload)
        resp.raise_for_status()
    data = resp.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"].strip()
    return stem


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Output dataclass
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@dataclass
class ConvertedSample:
    """Holds the open-ended version of a problem alongside the original."""
    source_id: str              # id of the ProblemLike it came from
    source_type: str            # "seed" | "evolved"
    problem_type: str           # "mcq" | "proof" | "true_false" | "yes_no" | "open"
    was_mcq: bool               # True if MCQ choices were detected and stripped
    original_problem: str       # verbatim original text
    problem: str                # open-ended version (may equal original if not MCQ)
    solution: str | None
    answer: str | None
    domain: str | None
    source: str
    reasoning_level: int | None = None
    metadata: dict = field(default_factory=dict, repr=False)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Main converter
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class MCQConverter:
    """
    Applies Big-Math-style signals + reformulations to convert problems to
    open-ended format suitable for SFT/RL training.

    Transformation table
    --------------------
    problem_type  | action
    ------------- | â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mcq           | strip choices; optionally LLM-rewrite MCQ-phrased stem
    proof         | pass through unchanged
    true_false    | _reformulate_true_false()
    yes_no        | _reformulate_yes_no()
    open          | pass through unchanged

    Parameters
    ----------
    cfg             : full pipeline config dict
    use_llm_rewrite : if True, call the HF Inference API to rephrase MCQ stems;
                      False  â†’ apply only regex-based stripping (offline)
    """

    def __init__(self, cfg: dict, use_llm_rewrite: bool = True) -> None:
        self._use_llm       = use_llm_rewrite
        self._keep_original = cfg.get("format_converter", {}).get("keep_original", True)
        # Only require API credentials when LLM rewriting is actually enabled
        if use_llm_rewrite:
            self._api_key  = (os.environ.get("MODEL_API_KEY") or "").strip()
            self._endpoint = os.environ.get("MODEL_ENDPOINT") or cfg["teacher_model"]["endpoint"]
            self._model    = cfg.get("teacher_model", {}).get("model", "deepseek-ai/DeepSeek-R1")
            if not self._api_key:
                logger.warning(
                    "MODEL_API_KEY not set; format-converter rewrite calls will be sent without Authorization header."
                )
        else:
            self._api_key  = ""
            self._endpoint = ""
            self._model    = ""

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def convert(self, sample: ProblemLike) -> ConvertedSample:
        """Convert a single SeedSample or EvolvedSample."""
        original     = sample.problem
        source_type  = "seed" if isinstance(sample, SeedSample) else "evolved"
        
        # Guard: if problem is None or empty, return as-is with 'open' type
        if not original:
            return ConvertedSample(
                source_id=_get_id(sample),
                source_type=source_type,
                problem_type="open",
                was_mcq=False,
                original_problem=original,
                problem=original or "",
                solution=sample.solution,
                answer=getattr(sample, "answer", None),
                raw=getattr(sample, "raw", {}),
            )
        
        problem_type = _detect_problem_type(original)

        if problem_type == "mcq":
            stripped = _strip_mcq_choices(original)
            if self._use_llm and _MCQ_HINT_PHRASES.search(stripped):
                try:
                    open_ended = _rewrite_stem_via_llm(self._endpoint, self._api_key, stripped, model=self._model)
                except Exception as exc:
                    logger.warning("LLM stem rewrite failed for {}: {}", _get_id(sample), exc)
                    open_ended = stripped
            else:
                open_ended = stripped
            was_mcq = True

        elif problem_type == "true_false":
            open_ended = _reformulate_true_false(original)
            was_mcq = False

        elif problem_type == "yes_no":
            open_ended = _reformulate_yes_no(original)
            was_mcq = False

        else:
            # "proof" or "open" â€” pass through unchanged
            open_ended = original
            was_mcq = False

        return ConvertedSample(
            source_id=_get_id(sample),
            source_type=source_type,
            problem_type=problem_type,
            was_mcq=was_mcq,
            original_problem=original,
            problem=open_ended,
            solution=sample.solution,
            answer=sample.answer,
            domain=sample.domain,
            source=sample.source,
        )

    def convert_batch(self, samples: list[ProblemLike]) -> list[ConvertedSample]:
        """Convert a list of samples, logging a type-distribution summary."""
        results: list[ConvertedSample] = []
        counts: dict[str, int] = {}
        for s in samples:
            cs = self.convert(s)
            results.append(cs)
            counts[cs.problem_type] = counts.get(cs.problem_type, 0) + 1
        logger.info(
            "Format conversion done ({} total): {}",
            len(results),
            "  ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        )
        return results


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_id(sample: ProblemLike) -> str:
    if isinstance(sample, SeedSample):
        return sample.id
    return sample.evolved_id


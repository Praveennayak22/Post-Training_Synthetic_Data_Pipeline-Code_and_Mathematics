"""
Math-Verify answer verifier + Rejection Sampling.
───────────────────────────────────────────────────
Uses the `math_verify` library (https://github.com/huggingface/Math-Verify)
to check whether the model's extracted answer matches the reference answer.

Install::
    pip install math-verify[antlr4_13_2]

Correct API (confirmed from README)::
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

    gold   = parse(ref_answer)   # ← gold FIRST
    pred   = parse(model_answer)
    result = verify(gold, pred)  # ← gold first, prediction second (NOT symmetric!)

Order matters: verify(gold, pred) ≠ verify(pred, gold).
See README § FAQ: “Why is verify function not symmetric?”

Rejection Sampling rule (pipeline spec §2.4 step 23):
  Discard any TracedSample where the final answer does NOT match ground truth.

For samples with no reference answer (FineMath passages, pure Evol variants)
we keep the sample but mark `verified = None` (cannot be checked).
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from loguru import logger

# math_verify public API  (install: pip install math-verify[antlr4_13_2])
try:
    from math_verify import verify, parse          # type: ignore[import]
    from math_verify.parser import (               # type: ignore[import]
        LatexExtractionConfig,
        ExprExtractionConfig,
    )
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    _MATH_VERIFY_AVAILABLE = False
    logger.warning(
        "math_verify not installed. "
        "Install with: pip install 'math-verify[antlr4_13_2]'  "
        "Falling back to sympy-based numeric comparison."
    )

from src.teacher_model.generator import TracedSample


# ── Timeout context manager (Unix-style; graceful on Windows) ─────────────────


@contextmanager
def _timeout(seconds: int) -> Iterator[None]:
    """Best-effort timeout; on Windows SIGALRM is unavailable — no-op there."""
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(seconds)
        yield
    except AttributeError:
        # Windows: SIGALRM not available, just yield without timeout
        yield
    finally:
        try:
            signal.alarm(0)
        except AttributeError:
            pass


# ── Output dataclass ──────────────────────────────────────────────────────────


@dataclass
class VerifiedSample:
    """TracedSample with a verification stamp."""
    # ── Core fields (copied from TracedSample) ─────────────────────────────
    source_id: str
    problem: str
    think_trace: str
    raw_answer: str
    extracted_answer: str | None
    reference_answer: str | None
    full_response: str
    domain: str | None
    source: str
    solution: str | None
    reasoning_level: int | None
    # ── Verification fields ────────────────────────────────────────────────
    verified: bool | None           # True=correct, False=wrong, None=no ref
    verification_method: str        # "math_verify" | "sympy_numeric" | "none"
    metadata: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_traced(
        cls,
        ts: TracedSample,
        verified: bool | None,
        method: str,
    ) -> "VerifiedSample":
        return cls(
            source_id=ts.source_id,
            problem=ts.problem,
            think_trace=ts.think_trace,
            raw_answer=ts.raw_answer,
            extracted_answer=ts.extracted_answer,
            reference_answer=ts.reference_answer,
            full_response=ts.full_response,
            domain=ts.domain,
            source=ts.source,
            solution=ts.solution,
            reasoning_level=ts.reasoning_level,
            verified=verified,
            verification_method=method,
            metadata=ts.metadata,
        )


# ── Sympy fallback ────────────────────────────────────────────────────────────


def _normalize_answer(s: str) -> str:
    """
    Normalize an answer string before parsing.
    Strips 'x = ', 'y = ', etc. (competition math often writes 'x = 33')
    so sympy can compare the numeric/symbolic value directly.
    """
    s = s.strip()

    # Strip one layer of common math wrappers.
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            s = s[len(left):-len(right)].strip()

    # Normalize trivial formatting differences.
    s = s.replace("−", "-")
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Handle 'var = value' form: take everything after the last '='
    if "=" in s and not s.startswith("\\"):
        parts = s.split("=")
        rhs = parts[-1].strip()
        if rhs:
            s = rhs
    return s


def _sympy_numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool | None:
    """
    Return True if a and b are mathematically equal, False if not, None if
    comparison cannot be determined (parse error, sympy not installed, etc.).

    Returning None (instead of False) on errors is critical: the verifier
    must not reject samples simply because sympy couldn't parse the answer.
    """
    a = _normalize_answer(a)
    b = _normalize_answer(b)

    # Fast path: exact string match after normalisation
    if a == b:
        return True

    try:
        from sympy import sympify, simplify, N
        from sympy.parsing.latex import parse_latex

        def _parse(s: str):
            try:
                return parse_latex(s)
            except Exception:
                return sympify(s)

        expr_a = _parse(a)
        expr_b = _parse(b)
        diff   = simplify(expr_a - expr_b)
        return abs(complex(N(diff))) < tol
    except Exception:
        # Cannot determine — do NOT treat as wrong (False).
        # The rejection sampler keeps verified=None samples.
        return None


def _parse_with_math_verify(answer: str):
    """
    Parse an answer via math_verify across library versions.

    Newer builds may reject older LatexExtractionConfig kwargs such as `boxed`.
    In that case, fall back to the library defaults instead of disabling
    verification for the sample.
    """
    try:
        return parse(
            answer,
            extraction_config=[
                LatexExtractionConfig(
                    boxed="all",
                    basic_latex=True,
                    units=True,
                    malformed_operators=False,
                ),
                ExprExtractionConfig(),
            ],
        )
    except TypeError:
        return parse(answer)


# ── Main verifier ─────────────────────────────────────────────────────────────


class MathVerifier:
    """
    Verifies model answers against reference answers and applies rejection
    sampling to discard incorrect traces.

    Parameters
    ----------
    cfg      : full pipeline config dict
    """

    def __init__(self, cfg: dict) -> None:
        vcfg = cfg["verification"]
        self._timeout_s = int(vcfg["timeout_seconds"])
        self._tol       = float(vcfg["numeric_precision"])

    # ── public API ────────────────────────────────────────────────────────────

    def verify(self, ts: TracedSample) -> VerifiedSample:
        """
        Verify a single TracedSample.

        - If no reference answer exists  → verified = None
        - If exact string match          → verified = True  (fast path)
        - If math_verify installed       → use it first
        - Otherwise                      → sympy numeric fallback
        - If sympy can't parse           → verified = None  (keep sample)
        """
        # No ground truth available
        if not ts.reference_answer or not ts.extracted_answer:
            return VerifiedSample.from_traced(ts, verified=None, method="none")

        ref   = ts.reference_answer.strip()
        pred  = ts.extracted_answer.strip()

        # ── Fast path: exact string match (handles '\\frac{13}{6}' == '\\frac{13}{6}') ──
        if _normalize_answer(pred) == _normalize_answer(ref):
            return VerifiedSample.from_traced(ts, verified=True, method="exact_match")

        # ── Try math_verify ─────────────────────────────────────────────────
        if _MATH_VERIFY_AVAILABLE:
            try:
                with _timeout(self._timeout_s):
                    # gold (ref) MUST be first arg — verify() is NOT symmetric
                    # See: https://github.com/huggingface/Math-Verify#verify-function
                    gold_parsed = _parse_with_math_verify(ref)
                    pred_parsed = _parse_with_math_verify(pred)
                    result: bool = verify(gold_parsed, pred_parsed)
                return VerifiedSample.from_traced(ts, verified=bool(result), method="math_verify")
            except TimeoutError:
                logger.warning("math_verify timed out for {}", ts.source_id)
            except Exception as exc:
                logger.debug("math_verify error for {}: {}", ts.source_id, exc)

        # ── Sympy numeric fallback ───────────────────────────────────────────
        try:
            with _timeout(self._timeout_s):
                result = _sympy_numeric_equal(pred, ref, self._tol)
            if result is None:
                # sympy couldn't parse — keep the sample as unverifiable
                return VerifiedSample.from_traced(ts, verified=None, method="none")
            return VerifiedSample.from_traced(ts, verified=result, method="sympy_numeric")
        except TimeoutError:
            logger.warning("sympy timed out for {}", ts.source_id)
        except Exception as exc:
            logger.debug("sympy error for {}: {}", ts.source_id, exc)

        return VerifiedSample.from_traced(ts, verified=None, method="none")

    def verify_batch(
        self, samples: list[TracedSample]
    ) -> list[VerifiedSample]:
        """Verify a batch; returns ALL samples (rejection happens separately)."""
        results = [self.verify(ts) for ts in samples]
        correct  = sum(1 for r in results if r.verified is True)
        wrong    = sum(1 for r in results if r.verified is False)
        no_ref   = sum(1 for r in results if r.verified is None)
        logger.info(
            "Verification: {} correct | {} wrong | {} no-reference (total {})",
            correct, wrong, no_ref, len(results),
        )
        return results

    # ── Rejection sampling ────────────────────────────────────────────────────

    @staticmethod
    def rejection_sample(samples: list[VerifiedSample]) -> list[VerifiedSample]:
        """
        Discard samples where verified == False (wrong answer).
        Keep verified == True  (correct) and verified == None (no reference).

        This implements pipeline spec §2.4 step 23.
        """
        accepted = [s for s in samples if s.verified is not False]
        rejected = len(samples) - len(accepted)
        logger.info(
            "Rejection sampling: {} accepted, {} discarded (wrong answer).",
            len(accepted), rejected,
        )
        return accepted

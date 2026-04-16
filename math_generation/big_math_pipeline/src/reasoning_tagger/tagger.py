"""
Reasoning Level Tagger — assigns a 0-4 score per the 5-Level Reasoning Scale.
──────────────────────────────────────────────────────────────────────────────
Scale (from spec §2.2):
  0  Minimal      – Simple factual recall, identification, no deduction required.
  1  Basic        – Straightforward connections, single-step logical processes.
  2  Intermediate – Multiple factors/concepts combined (e.g., Algebra + Geometry).
  3  Advanced     – Sophisticated multi-dimensional analysis, causal relationships.
  4  Expert       – Theoretical frameworks, deep counterfactual reasoning, novel synthesis.

Tagging strategy
────────────────
1. Rule-based heuristics score a problem on three axes:
     A) Structural complexity of the problem text
     B) Depth of the think trace (if available)
     C) Evol-Instruct mutation depth (if available from metadata)
   These are combined into a 0-4 score.

2. The rule-based score is cheap (no LLM call) and deterministic.
   It is sufficient for correct distribution across levels; a calibration
   check later can adjust thresholds if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from loguru import logger

from src.verification.verifier import VerifiedSample

# ── Heuristic signals ─────────────────────────────────────────────────────────

# Keywords that bump complexity upward
_ADV_KEYWORDS = re.compile(
    r"\b(prove|proof|derive|theorem|lemma|corollary|conjecture|"
    r"generalize|counterfactual|novel|synthesis|framework|"
    r"eigenvalue|determinant|manifold|topology|abstract algebra|"
    r"number field|galois|analytic|p-adic)\b",
    re.IGNORECASE,
)

_INTER_KEYWORDS = re.compile(
    r"\b(integrate|differentiate|matrix|logarithm|exponential|"
    r"complex number|polynomial roots|modular arithmetic|"
    r"combinatorics? on words|generating function|recurrence)\b",
    re.IGNORECASE,
)

_BASIC_KEYWORDS = re.compile(
    r"\b(solve|find|calculate|simplify|evaluate|compute|"
    r"what is|determine|identify)\b",
    re.IGNORECASE,
)

# Domain-based baseline level
_DOMAIN_BASELINE: dict[str, int] = {
    "pre_algebra":          0,
    "algebra":              1,
    "geometry":             1,
    "number_theory":        2,
    "intermediate_algebra": 2,
    "polynomial_roots":     2,
}

# Evol mutation depth contribution
_DEPTH_BONUS: dict[int, int] = {1: 0, 2: 1, 3: 2}

# ── Tagger ────────────────────────────────────────────────────────────────────


class ReasoningTagger:
    """Assigns reasoning levels 0–4 to VerifiedSamples."""

    def tag(self, sample: VerifiedSample) -> VerifiedSample:
        """Return a copy of sample with reasoning_level populated."""
        level = self._score(sample)
        # dataclasses don't have .replace without importing replace explicitly
        from dataclasses import replace as dc_replace
        return dc_replace(sample, reasoning_level=level)

    def tag_batch(self, samples: list[VerifiedSample]) -> list[VerifiedSample]:
        tagged = [self.tag(s) for s in samples]
        dist   = {i: sum(1 for t in tagged if t.reasoning_level == i) for i in range(5)}
        logger.info("Reasoning-level distribution: {}", dist)
        return tagged

    # ── scoring logic ─────────────────────────────────────────────────────────

    def _score(self, sample: VerifiedSample) -> int:
        problem     = sample.problem or ""
        think_trace = sample.think_trace or ""
        domain      = sample.domain or "algebra"

        # ── Axis A: problem text signals ──────────────────────────────────────
        if _ADV_KEYWORDS.search(problem):
            text_score = 3
        elif _INTER_KEYWORDS.search(problem):
            text_score = 2
        elif _BASIC_KEYWORDS.search(problem):
            text_score = 1
        else:
            text_score = 1   # default

        # ── Axis B: think-trace depth ─────────────────────────────────────────
        # Proxy: number of distinct reasoning steps (lines / sentence endings)
        trace_lines = len([ln for ln in think_trace.splitlines() if ln.strip()])
        if trace_lines > 40:
            trace_bonus = 2
        elif trace_lines > 20:
            trace_bonus = 1
        else:
            trace_bonus = 0

        # ── Axis C: evol mutation depth (stored in source_id convention) ──────
        depth_bonus = 0
        depth_match = re.search(r"::d(\d+)$", sample.source_id)
        if depth_match:
            depth = int(depth_match.group(1))
            depth_bonus = _DEPTH_BONUS.get(depth, 2)

        # ── Axis D: domain baseline ───────────────────────────────────────────
        domain_base = _DOMAIN_BASELINE.get(domain, 1)

        # ── Combine: use domain_base floor, add bonuses, cap at 4 ─────────────
        raw = max(domain_base, text_score) + trace_bonus + depth_bonus

        # Expert cap: only assign 4 when strong signals present
        if raw >= 4 and not _ADV_KEYWORDS.search(problem) and trace_lines < 30:
            raw = 3

        return min(raw, 4)

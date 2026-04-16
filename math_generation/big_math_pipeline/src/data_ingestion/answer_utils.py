"""
Shared reference-answer extraction utilities for dataset loaders.

Goal:
- Keep Numina-style boxed extraction reliable.
- Normalize heterogeneous answer formats from other datasets so Step 5
  receives cleaner reference answers.
"""

from __future__ import annotations

import re
from typing import Iterable

_BOXED_KEY = r"\boxed{"
_INLINE_DOLLAR_RE = re.compile(r"\$(.+?)\$", re.DOTALL)
_LATEX_PAREN_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(final\s+answer|answer|ans)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
_ANSWER_IS_RE = re.compile(
    r"^\s*(the\s+)?(final\s+)?answer\s+is\s*[:\-]?\s*",
    flags=re.IGNORECASE,
)


def extract_boxed_answer(text: str | None) -> str | None:
    """Extract the content of the last \boxed{...} with brace-depth parsing."""
    if not text:
        return None

    last_start = text.rfind(_BOXED_KEY)
    if last_start < 0:
        return None

    pos = last_start + len(_BOXED_KEY)
    depth = 1
    chars: list[str] = []

    while pos < len(text) and depth > 0:
        ch = text[pos]
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

    return "".join(chars).strip() if depth == 0 else None


def _cleanup_answer_text(value: str) -> str:
    s = value.strip().strip("`").strip()
    s = _ANSWER_PREFIX_RE.sub("", s)
    s = _ANSWER_IS_RE.sub("", s)
    # Remove trivial surrounding punctuation often seen in prose answers.
    s = s.strip().rstrip(".").strip()
    return s


def normalize_reference_answer(text: str | None) -> str | None:
    """
    Normalize a raw answer/solution snippet into a compact reference answer.

    Priority:
    1) last boxed answer
    2) last LaTeX inline parenthesized segment
    3) last inline-dollar segment $...$
    4) cleaned last non-empty line / cleaned full text
    """
    if text is None:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    boxed = extract_boxed_answer(raw)
    if boxed:
        return _cleanup_answer_text(boxed)

    latex_paren = _LATEX_PAREN_RE.findall(raw)
    if latex_paren:
        candidate = _cleanup_answer_text(latex_paren[-1])
        if candidate:
            return candidate

    inline_math = _INLINE_DOLLAR_RE.findall(raw)
    if inline_math:
        candidate = _cleanup_answer_text(inline_math[-1])
        if candidate:
            return candidate

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        return _cleanup_answer_text(lines[-1]) or _cleanup_answer_text(raw)

    return _cleanup_answer_text(raw)


def pick_reference_answer(*candidates: object) -> str | None:
    """
    Return the first non-empty normalized answer from a list of candidates.

    Each candidate may be a scalar value or an iterable of possible strings.
    """
    for candidate in candidates:
        if candidate is None:
            continue

        if isinstance(candidate, (list, tuple)):
            values: Iterable[object] = candidate
        else:
            values = (candidate,)

        for value in values:
            normalized = normalize_reference_answer(None if value is None else str(value))
            if normalized:
                return normalized

    return None

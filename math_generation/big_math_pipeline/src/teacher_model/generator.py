"""
DeepSeek-R1 Teacher Model â€” <think> trace generator.
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Feeds each open-ended problem to DeepSeek-R1 via the HuggingFace Inference
API and collects the full reasoning trace, which contains:

    <think>
    ... step-by-step reasoning ...
    </think>
    \\boxed{final_answer}

The trace is then split into:
  â€¢ think_trace  â€“ the chain-of-thought content inside <think>...</think>
  â€¢ raw_answer   â€“ the text after </think> (usually a \\boxed{} expression)

TracedSamples are passed to Math-Verify in the next step.
"""

from __future__ import annotations

import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx
from loguru import logger

from src.format_converter.converter import ConvertedSample

# â”€â”€ Prompt template â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_TEACHER_PROMPT = """\
Solve the following math problem completely. \
Think step-by-step inside <think>...</think> tags and then state your final answer.

Problem:
{problem}

Begin your response with <think> and end with the final answer on its own line \
in the form \\boxed{{answer}}."""

# â”€â”€ Regex to parse model output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_THINK_RE  = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_BOXED_KEY = r"\boxed{"
_FINAL_ANSWER_RE = re.compile(
    r"(?:^|\\b)(?:final answer|answer|ans|答案|最終答案)\s*(?:is|:|=|是|為|：)?\s*(.+)$",
    re.IGNORECASE,
)


# â”€â”€ Output dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class TracedSample:
    """
    A problem with a DeepSeek-R1 reasoning trace attached.

    Fields
    ------
    source_id      â€“ id carried from ConvertedSample
    problem        â€“ open-ended problem text
    think_trace    â€“ raw content inside <think>...</think>
    raw_answer     â€“ text after </think> (full model output tail)
    extracted_answer â€“ best-effort extracted final answer (boxed or last line)
    reference_answer â€“ ground truth from seed, if available
    full_response  â€“ complete model output for debugging
    domain, source â€“ inherited
    reasoning_level â€“ assigned later by tagger
    verified       â€“ set by Math-Verify step
    """
    source_id: str
    problem: str
    think_trace: str
    raw_answer: str
    extracted_answer: str | None
    reference_answer: str | None
    full_response: str
    domain: str | None
    source: str
    solution: str | None = None
    reasoning_level: int | None = None
    verified: bool | None = None         # filled by Math-Verify step
    metadata: dict = field(default_factory=dict, repr=False)


# â”€â”€ API helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _call_teacher(
    endpoint: str,
    api_key: str | None,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    model: str = "glm-5-fp8",
    enable_thinking: bool = True,
    timeout: float = 45.0,
    max_attempts: int = 1,
    retry_backoff_seconds: float = 2.0,
) -> str:
    """
    Call any OpenAI-compatible chat completions endpoint.
    Supports SGLang's chat_template_kwargs.enable_thinking flag
    (api.tensorstudio.ai) to request full <think> reasoning traces.
    """
    # Normalize endpoint to a concrete chat-completions route.
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
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    data: dict | None = None
    transient_errors = (
        httpx.HTTPStatusError,
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    )

    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(target, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            break
        except transient_errors as exc:
            if attempt >= attempts:
                raise
            wait_s = retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Teacher request failed (attempt {}/{}): {}. Retrying in {:.1f}s",
                attempt,
                attempts,
                exc,
                wait_s,
            )
            time.sleep(wait_s)

    if data is None:
        raise RuntimeError("Teacher response was empty after retry loop")

    # OpenAI-compatible response: reasoning_content holds the <think> trace,
    # content holds the final answer (SGLang/GLM-4 style).
    if "choices" in data and data["choices"]:
        msg = data["choices"][0]["message"]
        reasoning = (msg.get("reasoning_content") or "").strip()
        answer    = (msg.get("content") or "").strip()
        # Reassemble as <think>...</think>\n<answer> so downstream parsers work
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n{answer}"
        return answer
    return str(data).strip()


# â”€â”€ Parse helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _extract_think(text: str) -> str:
    """Return content inside the first <think>...</think> block."""
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_after_think(text: str) -> str:
    """Return everything after </think>."""
    end = text.lower().rfind("</think>")
    if end == -1:
        return text
    return text[end + len("</think>"):].strip()


def _extract_boxed_answer(text: str) -> str | None:
    """
    Extract the last \\boxed{...} from text using brace-counting.
    Handles any nesting depth: \\boxed{\\frac{13}{6}}, \\boxed{\\sqrt{\\frac{a}{b}}}, etc.
    Falls back to the last non-empty line if no \\boxed{} is found.
    """
    last_start = text.rfind(_BOXED_KEY)
    if last_start >= 0:
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
        if depth == 0:
            return "".join(chars).strip()
    return None


def _unwrap_latex_enclosure(text: str) -> str:
    """Remove one layer of common LaTeX/math wrappers."""
    s = text.strip()
    pairs = [
        ("$$", "$$"),
        ("$", "$"),
        (r"\\(", r"\\)"),
        (r"\\[", r"\\]"),
    ]
    for left, right in pairs:
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            return s[len(left):-len(right)].strip()
    return s


def _clean_answer_candidate(text: str | None) -> str | None:
    """Normalize a candidate answer span before verification."""
    if not text:
        return None
    s = text.strip().strip("` ")
    if not s:
        return None

    # Guard against parser artifacts being treated as answers.
    lowered = s.lower()
    if "<think" in lowered or "</think" in lowered:
        return None

    # If the candidate itself still contains a boxed form, keep only the boxed payload.
    boxed = _extract_boxed_answer(s)
    if boxed and boxed != s:
        s = boxed

    # Remove one layer of display wrappers and trailing punctuation noise.
    s = _unwrap_latex_enclosure(s).rstrip(" .;,")
    s = s.strip("$").strip()

    # Common markdown emphasis wrappers.
    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        s = s[2:-2].strip()

    s = s.strip()
    if not s:
        return None
    lowered = s.lower()
    if "<think" in lowered or "</think" in lowered:
        return None
    return s


def _extract_from_final_answer_line(text: str) -> str | None:
    """Extract from lines such as 'Final answer: ...'."""
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        m = _FINAL_ANSWER_RE.search(line)
        if m:
            return _clean_answer_candidate(m.group(1))
    return None


def _looks_like_math_answer(line: str) -> bool:
    """Heuristic filter for short answer-like lines in model tails."""
    if not line:
        return False
    if len(line.split()) > 18:
        return False
    if any(tok in line.lower() for tok in ["therefore", "because", "explain", "proof"]):
        return False
    return bool(re.search(r"[0-9]|\\\\|=|\^|/|\(|\)|\[|\]|\{|\}|%|π|sqrt", line))


def _extract_from_tail_lines(text: str) -> str | None:
    """Try reversed tail lines and return the first plausible math answer."""
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        candidate = _clean_answer_candidate(line)
        if not candidate:
            continue
        if _looks_like_math_answer(candidate):
            return candidate
    return None


def _extract_final_answer(text: str) -> str | None:
    """
    Robust final-answer extraction:
    1) boxed payload from tail section
    2) 'Final answer: ...' line from tail
    3) last non-empty tail line
    4) answer-like last line from think block
    5) boxed payload from full response
    """
    tail = _extract_after_think(text)
    think = _extract_think(text)
    candidates: list[str | None] = [
        _extract_boxed_answer(tail),
        _extract_from_final_answer_line(tail),
        _extract_from_tail_lines(tail),
        _extract_from_final_answer_line(text),
        _extract_from_tail_lines(think),
        _extract_boxed_answer(text),
    ]

    for candidate in candidates:
        cleaned = _clean_answer_candidate(candidate)
        if cleaned:
            return cleaned
    return None


# â”€â”€ Main generator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TeacherGenerator:
    """
    Generates <think> traces for converted math problems using DeepSeek-R1.

    Parameters
    ----------
    cfg : full pipeline config dict
    """

    def __init__(self, cfg: dict) -> None:
        self._api_key        = (os.environ.get("MODEL_API_KEY") or "").strip()
        tcfg                 = cfg["teacher_model"]
        self._endpoint       = os.environ.get("MODEL_ENDPOINT") or tcfg["endpoint"]
        self._model          = tcfg.get("model", "glm-5-fp8")
        self._temperature    = tcfg["temperature"]
        self._max_new_tokens = tcfg["max_new_tokens"]
        self._enable_thinking = tcfg.get("enable_thinking", True)
        self._request_timeout = float(tcfg.get("request_timeout_seconds", 45))
        self._retry_attempts = max(1, int(tcfg.get("retry_attempts", 1)))
        self._retry_backoff_seconds = float(tcfg.get("retry_backoff_seconds", 2.0))
        self._parallel_workers = max(1, int(tcfg.get("parallel_workers", 2)))
        self._max_new_tokens_cap = int(tcfg.get("max_new_tokens_cap", 1000))
        self._fallback_only = bool(tcfg.get("fallback_only", False))
        if not self._api_key:
            logger.warning(
                "MODEL_API_KEY not set; teacher requests will be sent without Authorization header."
            )

    def _fallback_trace(self, sample: ConvertedSample) -> TracedSample | None:
        if not sample.solution or not sample.answer:
            return None
        return TracedSample(
            source_id=sample.source_id,
            problem=sample.problem,
            think_trace=sample.solution,
            raw_answer=str(sample.answer),
            extracted_answer=sample.answer,
            reference_answer=sample.answer,
            full_response=sample.solution,
            domain=sample.domain,
            source=sample.source,
            solution=sample.solution,
            metadata={"teacher_fallback": True},
        )

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate(self, sample: ConvertedSample) -> TracedSample | None:
        """
        Generate a <think> trace for a single problem.

        Returns None if the model call fails after all retries.
        """
        if self._fallback_only:
            ts = self._fallback_trace(sample)
            if ts is None:
                logger.warning(
                    "fallback_only enabled but no fallback data available for {}",
                    sample.source_id,
                )
            return ts

        prompt = _TEACHER_PROMPT.format(problem=sample.problem)
        try:
            full_response = _call_teacher(
                endpoint=self._endpoint,
                api_key=self._api_key,
                prompt=prompt,
                temperature=self._temperature,
                # Hard cap: model total context is 4096 tokens (input + output).
                # Typical traces are 300-600 tokens; lower cap reduces timeout risk.
                max_new_tokens=min(self._max_new_tokens, self._max_new_tokens_cap),
                model=self._model,
                enable_thinking=self._enable_thinking,
                timeout=self._request_timeout,
                max_attempts=self._retry_attempts,
                retry_backoff_seconds=self._retry_backoff_seconds,
            )
        except Exception as exc:
            logger.error("Teacher call failed for {}: {}", sample.source_id, exc)
            # Fallback for reliability in constrained/slow endpoint conditions:
            # use available seed/evolved solution text so pipeline can continue.
            ts = self._fallback_trace(sample)
            if ts is not None:
                logger.warning("Using fallback trace for {}", sample.source_id)
                return ts
            return None

        think_trace       = _extract_think(full_response)
        raw_answer        = _extract_after_think(full_response)
        extracted_answer  = _extract_final_answer(full_response)

        return TracedSample(
            source_id=sample.source_id,
            problem=sample.problem,
            think_trace=think_trace,
            raw_answer=raw_answer,
            extracted_answer=extracted_answer,
            reference_answer=sample.answer,
            full_response=full_response,
            domain=sample.domain,
            source=sample.source,
            solution=sample.solution,
        )

    def generate_batch(
        self, samples: list[ConvertedSample]
    ) -> list[TracedSample]:
        """
        Generate traces for a list of converted problems.

        Skips samples where the model call fails.
        """
        results: list[TracedSample] = []
        failed = 0

        if self._parallel_workers <= 1 or len(samples) <= 1:
            for i, s in enumerate(samples):
                logger.info("Generating trace {}/{}: {}", i + 1, len(samples), s.source_id)
                ts = self.generate(s)
                if ts is None:
                    failed += 1
                else:
                    results.append(ts)
        else:
            logger.info(
                "Generating teacher traces in parallel (workers={} | timeout={}s | retries={})",
                self._parallel_workers,
                self._request_timeout,
                self._retry_attempts,
            )
            with ThreadPoolExecutor(max_workers=self._parallel_workers) as pool:
                future_map = {pool.submit(self.generate, s): s for s in samples}
                done = 0
                for fut in as_completed(future_map):
                    done += 1
                    s = future_map[fut]
                    try:
                        ts = fut.result()
                    except Exception as exc:
                        logger.error("Teacher worker failed for {}: {}", s.source_id, exc)
                        failed += 1
                        continue

                    if ts is None:
                        failed += 1
                    else:
                        results.append(ts)

                    if done == len(samples) or done % self._parallel_workers == 0:
                        logger.info("Teacher progress: {}/{}", done, len(samples))

        logger.info(
            "Teacher traces: {} generated, {} failed out of {}",
            len(results), failed, len(samples),
        )
        return results


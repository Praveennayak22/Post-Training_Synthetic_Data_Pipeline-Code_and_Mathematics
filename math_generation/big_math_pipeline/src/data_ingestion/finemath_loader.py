"""
FineMath loader
───────────────
Dataset : HuggingFaceTB/finemath  (ODC-By v1.0)
HF page : https://huggingface.co/datasets/HuggingFaceTB/finemath

Available configs: finemath-3plus (34B tokens, 21.4M docs)
                   finemath-4plus (9.6B tokens, 6.7M docs, higher quality)
                   infiwebmath-3plus | infiwebmath-4plus

Real schema (confirmed from dataset card):
  text              – page content (Markdown + LaTeX)
  url               – source page URL
  fetch_time        – crawler timestamp (int64)
  content_mime_type – MIME type
  warc_filename     – CommonCrawl WARC source file
  warc_record_offset– WARC offset (int32)
  warc_record_length– WARC record size (int32)
  token_count       – number of Llama tokens (int32)
  char_count        – character count (int32)
  metadata          – additional OpenWebMath metadata
  score             – raw quality score (float64)
  int_score         – integer quality score 0–5 (int64)  ← use this for filtering
  crawl             – CommonCrawl crawl identifier
  snapshot_type     – 'latest' or 'largest'
  language          – document language
  language_score    – LangID probability (float64)

FineMath contains web passages, not standalone Q&A pairs.
We use the 'text' field as the seed 'problem'; Evol-Instruct later
synthesises concrete math problems from these passages.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from loguru import logger

from .base import SeedSample

# Priority-ordered keyword scan (same logic as numina loader)
_DOMAIN_KEYWORD_PRIORITY = [
    ("geometry",             ["triangle", "circle", "polygon", "angle", "area",
                              "coordinate geometry", "sphere", "cone", "rhombus"]),
    ("polynomial_roots",     ["polynomial", "root", "vieta", "cubic", "quartic",
                              "factor theorem", "zeros of"]),
    ("number_theory",        ["prime", "divisib", "modular arithmetic", "gcd",
                              "lcm", "integer", "factorial", "congruence"]),
    ("intermediate_algebra", ["logarithm", "exponential", "complex number",
                              "matrix", "determinant", "arithmetic progression",
                              "geometric progression"]),
    ("algebra",              ["equation", "inequalit", "function", "sequence",
                              "series", "quadratic"]),
    ("pre_algebra",          ["fraction", "percent", "ratio", "decimal",
                              "word problem", "arithmetic"]),
]


def _infer_domain_from_text(text: str) -> str:
    snippet = text[:800].lower()
    for dom, keywords in _DOMAIN_KEYWORD_PRIORITY:
        if any(kw in snippet for kw in keywords):
            return dom
    return "algebra"


def load_finemath(
    hf_token: str,
    cache_dir: str = "data/hf_cache",
    config_name: str = "finemath-4plus",
    split: str = "train",
    max_samples: int | None = None,
    min_text_length: int = 200,
    min_int_score: int = 3,          # use int_score field (0-5); 3+ = quality content
) -> Iterator[SeedSample]:
    """
    Yield SeedSample objects from FineMath.

    Load example from dataset card::

        data = load_dataset("HuggingFaceTB/finemath", "finemath-4plus",
                            split="train", num_proc=8)

    Parameters
    ----------
    hf_token        : HuggingFace token
    cache_dir       : local path to cache downloaded shards
    config_name     : one of finemath-3plus | finemath-4plus |
                      infiwebmath-3plus | infiwebmath-4plus
    split           : dataset split (only 'train' exists)
    max_samples     : cap on rows (None = all)
    min_text_length : skip passages shorter than this (noise filter)
    min_int_score   : keep only rows with int_score >= this value (3 = step-by-step math)
    """
    logger.info(
        "Loading FineMath  config={} split={} cap={} min_int_score={}",
        config_name, split, max_samples or "all", min_int_score,
    )

    ds = load_dataset(
        "HuggingFaceTB/finemath",
        config_name,
        split=split,
        token=hf_token,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )

    count = 0
    for idx, row in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break

        # Apply quality gate using the real int_score field
        int_score = row.get("int_score", 0)
        if int_score < min_int_score:
            continue

        text = (row.get("text") or "").strip()
        if len(text) < min_text_length:
            continue

        # Only keep English documents (language field is real in this dataset)
        if row.get("language", "en") != "en":
            continue

        domain = _infer_domain_from_text(text)

        yield SeedSample(
            id=f"finemath:{config_name}:{split}:{idx}",
            source="finemath",
            domain=domain,
            problem=text,           # full passage as seed
            solution=None,
            answer=None,
            raw={
                "url":        row.get("url", ""),
                "int_score":  int_score,
                "score":      row.get("score", 0.0),
                "token_count":row.get("token_count", 0),
                "language":   row.get("language", "en"),
                "crawl":      row.get("crawl", ""),
            },
        )
        count += 1

    logger.info("FineMath: yielded {} samples", count)

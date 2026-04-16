"""
Shared dataclass that every loader normalises its output into.
────────────────────────────────────────────────────────────────
Fields
  id          – unique identifier  (<dataset_key>:<split>:<row_index>)
  source      – dataset key string, e.g. "numina_math" | "finemath"
                (matches the key used in config.yaml and LOADER_REGISTRY)
  domain      – math domain label (best-effort, may be None)
  problem     – the raw problem / question text
  solution    – reference solution / CoT (may be None for web passages)
  answer      – final numeric / symbolic answer (None if unavailable)
  raw         – key metadata dict from the original dataset row (traceability)
  reasoning_level – populated later by the ReasoningTagger (default None)

Adding a new data source
────────────────────────
Create a loader module that yields SeedSample objects with  source  set to
the dataset config key, then register it in  loader.LOADER_REGISTRY.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeedSample:
    id: str
    source: str
    domain: str | None
    problem: str
    solution: str | None
    answer: str | None
    raw: dict = field(default_factory=dict, repr=False)
    reasoning_level: int | None = None

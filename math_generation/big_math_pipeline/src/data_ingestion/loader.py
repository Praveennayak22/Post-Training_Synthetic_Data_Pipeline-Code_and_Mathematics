"""
Unified seed-data loader — public interface for Step 1.
────────────────────────────────────────────────────────
Registry-driven design: each dataset entry in config.yaml maps to a
loader function via LOADER_REGISTRY.  Adding a new data source in
future requires only:

  1. Write   src/data_ingestion/<name>_loader.py  with a  load_<name>()
             function that yields / returns list[SeedSample].
  2. Register it below in  LOADER_REGISTRY.
  3. Add a matching block under  datasets:  in config/config.yaml.

No changes to this file are needed for new sources.

Config shape expected per dataset entry
────────────────────────────────────────
datasets:
  numina_math:            # key must match a LOADER_REGISTRY entry
    enabled: true         # set false to skip without deleting the block
    split: train
    max_samples: null
    # ... any extra kwargs forwarded verbatim to the loader function
  finemath:
    enabled: true
    config_name: finemath-4plus
    split: train
    max_samples: null
  # future_source:
  #   enabled: false
  #   ...

Usage
─────
    from src.data_ingestion import load_seed_data, SeedSample

    samples: list[SeedSample] = load_seed_data(cfg)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

from loguru import logger

from .base import SeedSample
from .numina_loader import load_numina
from .finemath_loader import load_finemath
from .s1k_loader import load_s1k
from .tw_math_reasoning_loader import load_tw_math_reasoning
from .telemath_loader import load_telemath
from .limo_loader import load_limo
from .theoremqa_loader import load_theoremqa
from .aime_cod_loader import load_aime_cod
from .math_calculator_tool_loader import load_math_calculator_tool
from .omni_math_loader import load_omni_math
from .deepmath_loader import load_deepmath

# ══════════════════════════════════════════════════════════════════════════════
#  LOADER REGISTRY
#  Key   = dataset config key used in config.yaml  (under  datasets:)
#  Value = callable(hf_token, cache_dir, **dataset_cfg_kwargs)
#          that returns list[SeedSample] or Iterator[SeedSample]
#
#  To add a new source:
#    from .my_loader import load_my_source
#    LOADER_REGISTRY["my_source_key"] = load_my_source
# ══════════════════════════════════════════════════════════════════════════════

LOADER_REGISTRY: dict[str, Callable[..., list[SeedSample] | Iterator[SeedSample]]] = {
    "numina_math": load_numina,
    "finemath":    load_finemath,
    "s1k":         load_s1k,
    "tw_math_reasoning": load_tw_math_reasoning,
    "telemath":    load_telemath,
    "limo":        load_limo,
    "theoremqa":   load_theoremqa,
    "aime_cod":    load_aime_cod,
    "math_calculator_tool": load_math_calculator_tool,
    "omni_math":   load_omni_math,
    "deepmath":    load_deepmath,
    # "gsm8k":      load_gsm8k,
    # "math_orca":  load_math_orca,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Public loader
# ══════════════════════════════════════════════════════════════════════════════

def load_seed_data(cfg: dict) -> list[SeedSample]:
    """
    Load seed samples from all *enabled* dataset entries in config.yaml.

    Iterates over every key under  cfg["datasets"]  and calls the matching
    loader from LOADER_REGISTRY.  If a key has  enabled: false  it is skipped.
    If a key has no matching loader a warning is emitted (not a hard failure),
    making it easy to stub out future sources.

    Parameters
    ----------
    cfg : full pipeline config dict (loaded from config/config.yaml)

    Returns
    -------
    Flat list of SeedSample objects, deduplicated by id.
    """
    hf_token  = os.environ["HF_TOKEN"]
    cache_dir = cfg["huggingface"]["cache_dir"]
    datasets_cfg: dict[str, dict] = cfg.get("datasets", {})

    if not datasets_cfg:
        logger.warning("No datasets configured under 'datasets:' in config.yaml.")
        return []

    all_samples: list[SeedSample] = []

    for dataset_key, dataset_opts in datasets_cfg.items():
        # Allow callers to temporarily disable a source without removing it
        if not dataset_opts.get("enabled", True):
            logger.info("Skipping dataset '{}' (enabled: false)", dataset_key)
            continue

        loader_fn = LOADER_REGISTRY.get(dataset_key)
        if loader_fn is None:
            logger.warning(
                "No loader registered for dataset key '{}'. "
                "Add it to LOADER_REGISTRY in loader.py to use it.",
                dataset_key,
            )
            continue

        # Build kwargs: pass everything except the 'enabled' flag itself
        kwargs: dict[str, Any] = {
            k: v for k, v in dataset_opts.items() if k != "enabled"
        }

        logger.info("Loading dataset '{}' with options: {}", dataset_key, kwargs)
        try:
            result = loader_fn(hf_token=hf_token, cache_dir=cache_dir, **kwargs)
            batch  = list(result)
            all_samples.extend(batch)
            logger.info("  └─ loaded {:,} samples from '{}'", len(batch), dataset_key)
        except Exception as exc:
            logger.error(
                "Failed to load dataset '{}': {}. Skipping.", dataset_key, exc
            )

    # Deduplicate by id (safety net when sources overlap)
    seen:   set[str]         = set()
    deduped: list[SeedSample] = []
    for s in all_samples:
        if s.id not in seen:
            seen.add(s.id)
            deduped.append(s)

    dupes = len(all_samples) - len(deduped)
    if dupes:
        logger.info("Removed {:,} duplicate seed samples.", dupes)

    logger.info(
        "Total seed samples loaded: {:,} (from {} dataset(s))",
        len(deduped),
        sum(1 for opts in datasets_cfg.values() if opts.get("enabled", True)),
    )
    return deduped


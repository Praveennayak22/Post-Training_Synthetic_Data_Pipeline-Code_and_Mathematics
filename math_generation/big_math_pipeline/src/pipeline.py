"""
Math Synthetic Data Pipeline — top-level orchestrator.
────────────────────────────────────────────────────────
Runs all steps end-to-end (or a specified subset):

  Step 1  Load seed data (NuminaMath-CoT + FineMath)
  Step 2  Evol-Instruct difficulty scaling
  Step 3  MCQ → open-ended conversion (Big-Math style)
  Step 4  DeepSeek-R1 teacher <think> trace generation
  Step 5  Math-Verify answer validation + rejection sampling
  Step 6  Reasoning-level tagging (0–4)
  Step 7  Serialise final dataset to Parquet

Output filename format: {domain}_{model}_{source}_{count}_{timestamp}.parquet
(Timestamp added to ensure each run produces a unique file with no overwrites)

Usage
─────
  python -m src.pipeline               # run full pipeline
  python -m src.pipeline --steps 1,2   # run only steps 1 and 2
  python -m src.pipeline --dry-run     # load + evolve only, skip API calls
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# ── load .env as early as possible ────────────────────────────────────────────
load_dotenv()

from src.data_ingestion        import load_seed_data
from src.evol_instruct         import EvolInstructScaler
from src.format_converter      import MCQConverter
from src.teacher_model         import TeacherGenerator
from src.verification          import MathVerifier
from src.reasoning_tagger      import ReasoningTagger

# ── helpers ───────────────────────────────────────────────────────────────────


_DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant that helps people find information. "
    "Use clear, accurate language and provide concise, correct answers."
)

_DEFAULT_INSTRUCTION = (
    "Solve the following math question. Show your reasoning in <think>...</think> "
    "and then provide the final answer clearly."
)


_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bthe user wants me to rewrite\b", re.IGNORECASE),
    re.compile(r"\boriginal problem\b", re.IGNORECASE),
    re.compile(r"\brewrite (it )?to be more difficult\b", re.IGNORECASE),
    re.compile(r"\badd(?:ing)? one meaningful mathematical constraint\b", re.IGNORECASE),
    re.compile(r"\bdo not solve\b", re.IGNORECASE),
    re.compile(r"\boutput only\b", re.IGNORECASE),
)


def _normalize_space(text: str | None) -> str:
    return " ".join((text or "").split())


def _sanitize_question_text(text: str | None) -> str:
    """Trim wrappers and normalize whitespace for final Question field."""
    q = (text or "").strip()
    if not q:
        return ""

    # Strip simple markdown fences often produced by model wrappers.
    q = q.replace("```", " ")
    q = q.replace("### Instruction:", " ")
    q = q.replace("### Response:", " ")
    q = _normalize_space(q)
    return q


def _is_question_like(text: str) -> bool:
    """
    Keep only rows that look like actual math questions and reject
    leaked mutation-instruction/meta text.
    """
    if not text:
        return False

    t = text.strip()
    if len(t) < 8:
        return False

    lower = t.lower()
    if any(p.search(lower) for p in _LEAK_PATTERNS):
        return False

    # Soft heuristics for question-like phrasing.
    has_math_verb = any(
        kw in lower
        for kw in (
            "solve",
            "find",
            "compute",
            "evaluate",
            "determine",
            "prove",
            "show that",
            "let ",
            "if ",
            "given",
        )
    )
    has_question_mark = "?" in t
    has_math_tokens = any(ch in t for ch in ("=", "+", "-", "*", "/", "^", "(", ")", "$", "\\"))

    return has_math_verb or has_question_mark or has_math_tokens


def _is_nonempty_answer(answer: str | None) -> bool:
    return bool(_normalize_space(answer))


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dirs(cfg: dict) -> None:
    for key in ("output_dir", "checkpoint_dir"):
        Path(cfg["pipeline"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["huggingface"]["cache_dir"]).mkdir(parents=True, exist_ok=True)


def _save_jsonl(
    records: list[dict],
    path: str,
    *,
    append: bool = True,
    dedupe_key: str = "sample_id",
) -> int:
    """
    Save records as newline-delimited JSON.

    When append=True and the file already exists, new records are appended.
    If dedupe_key is present, records with duplicate keys are skipped.

    Returns
    -------
    int
        Number of records written in this call.
    """
    out_path = Path(path)
    existing_ids: set[str] = set()

    if append and out_path.exists() and dedupe_key:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get(dedupe_key)
                if sid is not None:
                    existing_ids.add(str(sid))

    mode = "a" if append else "w"
    written = 0
    skipped_dupes = 0

    with open(out_path, mode, encoding="utf-8") as f:
        for rec in records:
            if dedupe_key:
                sid = rec.get(dedupe_key)
                if sid is not None:
                    sid = str(sid)
                    if sid in existing_ids:
                        skipped_dupes += 1
                        continue
                    existing_ids.add(sid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    action = "Appended" if append else "Saved"
    logger.info(
        "{} {:,} record(s) → {} (skipped duplicates: {:,})",
        action,
        written,
        path,
        skipped_dupes,
    )
    return written


def _save_parquet(
    records: list[dict],
    cfg: dict,
    output_dir: str,
) -> tuple[str, int]:
    """
    Save records as Apache Parquet with custom naming.
    
    Filename format: {domain}_{model}_{source}_{count}_{timestamp}.parquet
    Each run generates a unique file (timestamp ensures no overwrites).
    
    Returns
    -------
    tuple[str, int]
        (output_path, records_written)
    """
    import collections
    
    if not records:
        logger.warning("No records to save as parquet")
        return "", 0
    
    # Infer domain from most common domain in records
    domains = [rec.get("Domain", "unknown") for rec in records]
    domain_counter = collections.Counter(domains)
    primary_domain = domain_counter.most_common(1)[0][0] if domain_counter else "mixed"
    
    # Get model name from config
    model_name = cfg.get("teacher_model", {}).get("model", "unknown").replace("-", "_").replace(".", "")
    
    # Get source(s) from config - check enabled datasets
    sources = []
    for key, opts in cfg.get("datasets", {}).items():
        if opts.get("enabled", False):
            sources.append(key)
    source_str = "_".join(sources[:2]) if sources else "unknown"  # Limit to 2 sources for readability
    
    # Record count
    count = len(records)
    
    # Add timestamp for uniqueness (each run gets a unique file)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename: {domain}_{model}_{source}_{count}_{timestamp}.parquet
    parquet_filename = f"{primary_domain}_{model_name}_{source_str}_{count}_{timestamp}.parquet"
    out_path = str(Path(output_dir) / parquet_filename)
    
    # Convert records to DataFrame and save
    df = pd.DataFrame(records)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    
    logger.info(
        "Dataset saved as Parquet → {} ({:,} records, domain: {}, model: {})",
        out_path,
        count,
        primary_domain,
        model_name,
    )
    
    return out_path, count


def _cleanup_intermediate_json(cfg: dict) -> None:
    """Remove non-final JSON artifacts so only the final output file remains."""
    removed = 0

    # Main pipeline step checkpoints.
    chk_dir = Path(cfg["pipeline"]["checkpoint_dir"])
    if chk_dir.exists():
        for p in chk_dir.glob("*.json"):
            try:
                p.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Could not remove {}: {}", p, exc)

    # Optional test output folders used by local helper scripts.
    step_folders = (
        "data/step1_seeds",
        "data/step2_evolved",
        "data/step3_converted",
        "data/step4_traced",
        "data/step5_verified",
        "data/step6_tagged",
    )
    for rel in step_folders:
        step_dir = Path(rel)
        if not step_dir.exists():
            continue
        for p in step_dir.glob("*.json"):
            try:
                p.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Could not remove {}: {}", p, exc)

    # Remove summary file when running in only-output mode.
    summary_path = Path(cfg["pipeline"]["output_dir"]) / "pipeline_summary.json"
    if summary_path.exists():
        try:
            summary_path.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Could not remove {}: {}", summary_path, exc)

    logger.info("Cleanup complete: removed {} intermediate JSON file(s).", removed)


def _list_saved_datasets(cfg: dict) -> None:
    """List all saved datasets (both Parquet and JSONL for backward compatibility)."""
    output_dir = Path(cfg["pipeline"]["output_dir"])
    if not output_dir.exists():
        logger.info("Output directory does not exist yet.")
        return
    
    # Look for both parquet and jsonl files
    datasets_parquet = sorted(output_dir.glob("*.parquet"))
    datasets_jsonl = sorted(output_dir.glob("math_sft_rl_dataset_*.jsonl"))
    
    all_datasets = datasets_parquet + datasets_jsonl
    
    if not all_datasets:
        logger.info("No saved datasets found in {}", output_dir)
        return
    
    logger.info("Saved datasets in {}:", output_dir)
    for i, path in enumerate(all_datasets, 1):
        size_kb = path.stat().st_size / 1024
        mod_time = path.stat().st_mtime
        mod_dt = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        file_type = "Parquet" if path.suffix == ".parquet" else "JSONL"
        logger.info("  {}. {} [{}] ({:.1f} KB) — {}", i, path.name, file_type, size_kb, mod_dt)


def _word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.strip().split())


def _reasoning_label(level: int | None) -> str:
    if level is None:
        return "unknown"
    return _REASONING_LABELS.get(level, "unknown").lower()


def _build_final_output_record(record: dict, cfg: dict) -> dict:
    """
    Normalize final output to a spreadsheet-style schema for downstream review.
    Supports dataset-specific model names from config (e.g., deepmath → deepseek-r1).
    
    Handles both:
    - Full traced samples (after steps 4,5,6): has think_trace, extracted_answer, reasoning_level
    - Converted-only samples (when skipping 4,5,6): has answer, solution from step 3
    """
    # Check if source has dataset-specific model name in config
    source = record.get("source", "")
    model_name = None
    if source in cfg.get("datasets", {}):
        dataset_config = cfg["datasets"][source]
        model_name = dataset_config.get("model_name")
    
    # Fall back to default teacher model if not specified
    if not model_name:
        model_name = cfg.get("teacher_model", {}).get("model", "unknown")
    
    system_prompt = _DEFAULT_SYSTEM_PROMPT
    instruction = _DEFAULT_INSTRUCTION
    question = _sanitize_question_text(record.get("problem", ""))
    
    # Handle both traced (from step 4+) and converted-only (from step 3) data
    think = record.get("think_trace", "")
    if not think:
        # For converted-only data: use solution as the reasoning
        think = record.get("solution", "")
    
    # Answer: prefer raw_answer (traced) → extracted_answer (traced) → answer (converted)
    answer = record.get("raw_answer") or record.get("extracted_answer") or record.get("answer", "")

    return {
        "SFT/Reasoning": "Reasoning",
        "System prompt": system_prompt,
        "Instruction": instruction,
        "Question": question,
        "<Think>": think,
        "Answer": answer,
        "Difficulty level": _reasoning_label(record.get("reasoning_level")),
        "Number of words in instruction (preferably no. of tokens)": _word_count(instruction),
        "Number of words in system prompt": _word_count(system_prompt),
        "Number of words in question": _word_count(question),
        "Number of words in answer": _word_count(answer),
        "Task": "QA",
        "Domain": record.get("domain"),
        "Language": "English",
        "Source": record.get("source"),
        "Made by": "math_pipeline",
        "Reasoning/SFT": "reasoning",
        "Model": model_name,
        "sample_id": record.get("source_id"),
        "verified": record.get("verified"),
    }


def _load_step3_checkpoint(cfg: dict) -> list:
    """
    Reconstruct ConvertedSample objects from the step3 checkpoint JSON.
    Used when running --steps 4,5,6,7 without re-running steps 1-3.
    """
    from src.format_converter.converter import ConvertedSample
    path = Path(cfg["pipeline"]["checkpoint_dir"]) / "step3_converted.json"
    if not path.exists():
        logger.warning("Step 3 checkpoint not found at {}; returning empty list.", path)
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    samples: list = []
    for rec in d.get("records", []):
        flat: dict[str, Any] = {k: v for k, v in rec.items() if not k.startswith("_")}
        flat.update(rec.get("_extra", {}))
        try:
            cs = ConvertedSample(
                source_id        = flat.get("source_id", ""),
                source_type      = flat.get("source_type", "seed"),
                problem          = flat.get("problem", ""),
                original_problem = flat.get("original_problem", flat.get("problem", "")),
                problem_type     = flat.get("problem_type", "open"),
                was_mcq          = flat.get("was_mcq", False),
                answer           = flat.get("answer"),
                domain           = flat.get("domain"),
                source           = flat.get("source", ""),
                solution         = flat.get("solution"),
                metadata         = rec.get("_metadata", {}),
            )
            samples.append(cs)
        except Exception as exc:
            logger.warning("Skipped malformed step3 record: {}", exc)
    logger.info("Loaded {} ConvertedSamples from checkpoint: {}", len(samples), path)
    return samples


def _load_step4_checkpoint(cfg: dict) -> list:
    """
    Reconstruct TracedSample objects from the step4 checkpoint JSON.
    Used when running --steps 5,6,7 without re-running step 4.
    """
    from src.teacher_model.generator import TracedSample
    path = Path(cfg["pipeline"]["checkpoint_dir"]) / "step4_traced.json"
    if not path.exists():
        logger.warning("Step 4 checkpoint not found at {}; returning empty list.", path)
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    samples: list = []
    for rec in d.get("records", []):
        # _project() stores non-key fields in '_extra'; merge them back
        flat: dict[str, Any] = {k: v for k, v in rec.items() if not k.startswith("_")}
        flat.update(rec.get("_extra", {}))
        try:
            ts = TracedSample(
                source_id        = flat.get("source_id", ""),
                problem          = flat.get("problem", ""),
                think_trace      = flat.get("think_trace", ""),
                raw_answer       = flat.get("raw_answer", ""),
                extracted_answer = flat.get("extracted_answer"),
                reference_answer = flat.get("reference_answer"),
                full_response    = flat.get("full_response", ""),
                domain           = flat.get("domain"),
                source           = flat.get("source", ""),
                solution         = flat.get("solution"),
                reasoning_level  = flat.get("reasoning_level"),
                verified         = flat.get("verified"),
                metadata         = rec.get("_metadata", {}),
            )
            samples.append(ts)
        except Exception as exc:
            logger.warning("Skipped malformed step4 record: {}", exc)
    logger.info("Loaded {} TracedSamples from checkpoint: {}", len(samples), path)
    return samples


# ── step metadata ─────────────────────────────────────────────────────────────

_STEP_META: dict[int, dict] = {
    1: {
        "name":        "step1_seeds",
        "title":       "Step 1 — Seed Data Ingestion",
        "description": "Raw problem/solution pairs loaded from source datasets.",
        "key_fields":  ["id", "source", "domain", "problem", "solution", "answer"],
    },
    2: {
        "name":        "step2_evolved",
        "title":       "Step 2 — Evol-Instruct Difficulty Scaling",
        "description": "Each seed problem mutated via WizardMath-style operations to increase difficulty.",
        "key_fields":  ["evolved_id", "parent_id", "source", "domain",
                        "mutation_strategy", "mutation_depth",
                        "original_problem", "problem"],
    },
    3: {
        "name":        "step3_converted",
        "title":       "Step 3 — MCQ → Open-Ended Conversion (Big-Math style)",
        "description": "MCQ/True-False/Yes-No problems reformulated as open-ended questions.",
        "key_fields":  ["source_id", "source", "domain", "problem_type", "was_mcq",
                        "original_problem", "problem"],
    },
    4: {
        "name":        "step4_traced",
        "title":       "Step 4 — DeepSeek-R1 Teacher Trace Generation",
        "description": "Each problem solved by the teacher model; <think> trace + final answer captured.",
        "key_fields":  ["source_id", "source", "domain", "problem",
                        "think_trace", "extracted_answer", "reference_answer"],
    },
    5: {
        "name":        "step5_verified",
        "title":       "Step 5 — Math-Verify + Rejection Sampling",
        "description": "Samples with wrong answers discarded; verified=True/None kept.",
        "key_fields":  ["source_id", "source", "domain", "problem",
                        "extracted_answer", "reference_answer",
                        "verified", "verification_method"],
    },
    6: {
        "name":        "step6_tagged",
        "title":       "Step 6 — Reasoning Level Tagging (0–4)",
        "description": "Each sample scored on the 5-level reasoning scale. Ready for SFT/RL.",
        "key_fields":  ["source_id", "source", "domain", "problem",
                        "think_trace", "extracted_answer", "verified",
                        "reasoning_level", "verification_method"],
    },
}

_REASONING_LABELS = {0: "Minimal", 1: "Basic", 2: "Intermediate", 3: "Advanced", 4: "Expert"}


def _to_record(obj: Any) -> dict:
    """Convert a dataclass (or plain dict) to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)


def _project(record: dict, key_fields: list[str]) -> dict:
    """
    Return an ordered dict with key_fields first (human-readable summary),
    followed by a 'full_record' block containing all remaining fields.
    This lets you see the most important transformation at a glance.
    """
    focused: dict = {}
    for k in key_fields:
        if k in record:
            focused[k] = record[k]

    # Add reasoning label if level present
    if "reasoning_level" in focused and focused["reasoning_level"] is not None:
        focused["reasoning_label"] = _REASONING_LABELS.get(focused["reasoning_level"], "?")

    # Remaining fields as an audit trail
    extra = {k: v for k, v in record.items() if k not in key_fields and k != "metadata"}
    if extra:
        focused["_extra"] = extra
    if "metadata" in record and record["metadata"]:
        focused["_metadata"] = record["metadata"]
    return focused


def _save_step_json(cfg: dict, step: int, objects: list) -> str:
    """
    Write a pretty-printed JSON file for *step* under the checkpoint directory.

    Format::

        {
          "step":        2,
          "title":       "Step 2 — Evol-Instruct Difficulty Scaling",
          "description": "…",
          "timestamp":   "2026-03-08T12:00:00",
          "total":       30,
          "records": [
            {
              "evolved_id":         "…",
              "mutation_strategy":  "add_constraint",
              "original_problem":   "…",
              "problem":            "…",   ← the evolved version
              …
            },
            …
          ]
        }
    """
    if not cfg["pipeline"].get("save_step_checkpoints", True):
        logger.info("Skipping step checkpoint for step {} (save_step_checkpoints=false).", step)
        return ""

    meta    = _STEP_META[step]
    records = [_project(_to_record(obj), meta["key_fields"]) for obj in objects]

    payload = {
        "step":        step,
        "title":       meta["title"],
        "description": meta["description"],
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "total":       len(records),
        "records":     records,
    }

    path = str(Path(cfg["pipeline"]["checkpoint_dir"]) / f"{meta['name']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("{} → {} ({:,} records)", meta['title'], path, len(records))
    return path


# ── per-step runners ──────────────────────────────────────────────────────────


def run_step1(cfg: dict) -> list:
    logger.info("=== STEP 1: Seed data ingestion ===")
    seeds = load_seed_data(cfg)
    logger.info("Seeds loaded: {:,}", len(seeds))
    _save_step_json(cfg, 1, seeds)
    return seeds


def run_step2(cfg: dict, seeds: list) -> list:
    logger.info("=== STEP 2: Evol-Instruct difficulty scaling ===")
    scaler  = EvolInstructScaler(cfg)
    batch   = seeds[: cfg["pipeline"]["batch_size"]]   # cap for safety
    evolved = scaler.evolve_batch(batch)
    # Combine seeds + evolved variants for downstream steps
    combined = list(seeds) + evolved
    logger.info("Combined (seeds + evolved): {:,}", len(combined))
    _save_step_json(cfg, 2, combined)
    return combined


def run_step3(cfg: dict, samples: list, dry_run: bool = False) -> list:
    logger.info("=== STEP 3: MCQ → open-ended conversion ===")
    converter = MCQConverter(cfg, use_llm_rewrite=not dry_run)
    converted = converter.convert_batch(samples)
    _save_step_json(cfg, 3, converted)
    return converted


def run_step4(cfg: dict, converted: list, dry_run: bool = False) -> list:
    if dry_run:
        logger.info("=== STEP 4: Skipped (dry-run) ===")
        return []
    logger.info("=== STEP 4: DeepSeek-R1 teacher trace generation ===")
    generator = TeacherGenerator(cfg)
    traced    = generator.generate_batch(converted)
    _save_step_json(cfg, 4, traced)
    return traced


def run_step5(cfg: dict, traced: list) -> list:
    logger.info("=== STEP 5: Math-Verify + rejection sampling ===")
    verifier  = MathVerifier(cfg)
    verified  = verifier.verify_batch(traced)
    accepted  = MathVerifier.rejection_sample(verified)
    _save_step_json(cfg, 5, accepted)
    return accepted


def run_step6(cfg: dict, accepted: list) -> list:
    logger.info("=== STEP 6: Reasoning level tagging ===")
    tagger = ReasoningTagger()
    tagged = tagger.tag_batch(accepted)
    _save_step_json(cfg, 6, tagged)
    return tagged


def run_step7(cfg: dict, tagged: list) -> None:
    logger.info("=== STEP 7: Serialising final dataset ===")
    
    output_dir = Path(cfg["pipeline"]["output_dir"])
    records = [_build_final_output_record(_to_record(t), cfg) for t in tagged]

    total_before = len(records)
    filtered: list[dict] = []
    dropped_bad_question = 0
    dropped_empty_answer = 0

    for rec in records:
        question = rec.get("Question", "")
        answer = rec.get("Answer", "")

        if not _is_question_like(question):
            dropped_bad_question += 1
            continue
        if not _is_nonempty_answer(answer):
            dropped_empty_answer += 1
            continue
        filtered.append(rec)

    records = filtered
    logger.info(
        "Final quality filter: kept {:,}/{:,} | dropped bad_question={:,}, empty_answer={:,}",
        len(records),
        total_before,
        dropped_bad_question,
        dropped_empty_answer,
    )

    # Save as Parquet with custom naming: {domain}_{model}_{source}_{count}.parquet
    parquet_path, written = _save_parquet(records, cfg, str(output_dir))
    
    # Keep backward compatibility: also save full records as parquet latest
    latest_path = str(output_dir / "dataset_latest.parquet")
    pd.DataFrame(records).to_parquet(latest_path, engine="pyarrow", compression="snappy", index=False)

    if cfg["pipeline"].get("write_pipeline_summary", True):
        chk_dir = cfg["pipeline"]["checkpoint_dir"]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        summary = {
            "total_records": len(records),
            "records_written": written,
            "output_file": parquet_path,
            "latest_file": latest_path,
            "timestamp": timestamp,
            "file_format": "parquet",
            "step_json_files": {
                meta["title"]: str(Path(chk_dir) / f"{meta['name']}.json")
                for meta in _STEP_META.values()
            },
        }
        summary_path = str(Path(cfg["pipeline"]["output_dir"]) / "pipeline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Summary written → {}", summary_path)
    
    logger.info("Dataset saved → {} ({:,} records)", parquet_path, len(records))
    logger.info("Latest file → {}", latest_path)


# ── main entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Math Synthetic Data Pipeline")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--steps",
        default="1,2,3,4,5,6,7",
        help="Comma-separated step numbers to run (e.g. '1,2,3')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM API calls (useful for testing ingestion + evol logic)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Load exactly N seed problems from enabled datasets. "
            "Preferred alias for --max-samples."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override max_samples for ALL datasets (e.g. --max-samples 10). "
            "Useful for quick end-to-end smoke tests without touching config.yaml."
        ),
    )
    parser.add_argument(
        "--max-evol",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit the number of seeds sent to Evol-Instruct (Step 2). "
            "Overrides pipeline.batch_size at runtime. "
            "E.g. --max-evol 2 with --max-samples 10 → 10 seeds loaded but only 2 evolved."
        ),
    )
    parser.add_argument(
        "--generate-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Generate exactly N final output records. "
            "Preferred alias for --num-problems."
        ),
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit the final output size to N records (applied right before Step 7). "
            "Use this when you want an exact-sized output dataset."
        ),
    )
    parser.add_argument(
        "--only-output",
        action="store_true",
        help=(
            "Keep only the final output JSONL: skip per-step checkpoint JSON files and "
            "delete existing intermediate JSON artifacts."
        ),
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        metavar="ID",
        help="For job array parallelization: process only seeds for this chunk (0-indexed)",
    )
    parser.add_argument(
        "--total-chunks",
        type=int,
        default=None,
        metavar="N",
        help="Total number of chunks the job is split into (used with --chunk-id)",
    )
    args = parser.parse_args()

    steps = {int(s.strip()) for s in args.steps.split(",")}

    cfg = _load_config(args.config)
    _ensure_dirs(cfg)

    pipeline_cfg = cfg.setdefault("pipeline", {})

    # Dedicated seed/generate controls with clear precedence:
    # CLI aliases > legacy CLI flags > config fields.
    seed_count = (
        args.seed_count
        if args.seed_count is not None
        else args.max_samples
        if args.max_samples is not None
        else pipeline_cfg.get("seed_count")
    )
    generate_count = (
        args.generate_count
        if args.generate_count is not None
        else args.num_problems
        if args.num_problems is not None
        else pipeline_cfg.get("generate_count")
    )

    if args.only_output:
        cfg["pipeline"]["save_step_checkpoints"] = False
        cfg["pipeline"]["write_pipeline_summary"] = False
        _cleanup_intermediate_json(cfg)

    # Apply seed_count override to every dataset in the config
    if seed_count is not None:
        if seed_count < 0:
            logger.error("seed_count must be >= 0")
            sys.exit(1)
        logger.info(
            "SEED COUNT: capping every dataset at {} sample(s)",
            seed_count,
        )
        for ds_opts in cfg.get("datasets", {}).values():
            ds_opts["max_samples"] = seed_count

    # Apply --max-evol override (limits how many seeds are fed to Evol-Instruct)
    if args.max_evol is not None:
        logger.info(
            "EVOL CAP: only first {} seed(s) will be evolved (--max-evol {})",
            args.max_evol, args.max_evol,
        )
        cfg["pipeline"]["batch_size"] = args.max_evol

    # Verify required env vars
    if not os.environ.get("HF_TOKEN"):
        # NuminaMath-CoT is public — token not strictly required; warn only.
        logger.warning(
            "HF_TOKEN is not set. Public datasets (NuminaMath-CoT) work without it, "
            "but gated datasets (FineMath) will fail.  Set it in .env if needed."
        )
    needs_model_inference = (2 in steps) or (4 in steps) or (3 in steps and not args.dry_run)
    if needs_model_inference and not os.environ.get("MODEL_API_KEY"):
        logger.warning(
            "MODEL_API_KEY is not set. Continuing without Authorization header; "
            "this works for internal cluster endpoints that allow unauthenticated access."
        )

    # Handle job array chunking: slice seeds if --chunk-id and --total-chunks provided
    chunk_id = args.chunk_id
    total_chunks = args.total_chunks
    if (chunk_id is not None) != (total_chunks is not None):
        logger.error("Both --chunk-id and --total-chunks must be provided together")
        sys.exit(1)
    if chunk_id is not None and total_chunks is not None:
        if not (0 <= chunk_id < total_chunks):
            logger.error("--chunk-id must be in range [0, --total-chunks)")
            sys.exit(1)
        logger.info("Job array mode: chunk {}/{}", chunk_id, total_chunks)

    seeds     = run_step1(cfg)                            if 1 in steps else []
    
    # Apply chunking if in job array mode
    if chunk_id is not None and total_chunks is not None and seeds:
        chunk_size = (len(seeds) + total_chunks - 1) // total_chunks  # ceiling division
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, len(seeds))
        seeds = seeds[start_idx:end_idx]
        logger.info(
            "Chunked seeds: total={}, chunk_size={}, indices=[{}, {}) → {:,} seeds for this job",
            len(seeds) if chunk_id == 0 else "?",
            chunk_size,
            start_idx,
            end_idx,
            len(seeds),
        )
    
    combined  = run_step2(cfg, seeds)                     if 2 in steps else seeds
    if 3 in steps:
        converted = run_step3(cfg, combined, args.dry_run)
    elif any(s in steps for s in (4, 5, 6, 7)):
        # Resume from saved step3 checkpoint (e.g. --steps 4,5,6,7)
        converted = _load_step3_checkpoint(cfg)
    else:
        converted = []

    # Speed optimization: when a final target count is set, don't ask the teacher
    # model to process more samples than needed for final output.
    if generate_count is not None and 4 in steps and converted:
        if len(converted) > generate_count:
            logger.info(
                "Teacher input capped by generate_count: {} → {}",
                len(converted),
                generate_count,
            )
            converted = converted[:generate_count]

    if 4 in steps:
        traced = run_step4(cfg, converted, args.dry_run)
    elif any(s in steps for s in (5, 6, 7)):
        # Resume from saved step4 checkpoint (e.g. --steps 5,6,7)
        # But if steps 4,5,6 are all skipped and only 7 runs, use converted directly
        if not any(s in steps for s in (4, 5, 6)):
            # Fast path: only 1,3,7 → use converted data as-is
            logger.info("Fast path: skipping steps 4,5,6 (data already solved); passing converted → step 7")
            traced = converted
        else:
            traced = _load_step4_checkpoint(cfg)
    else:
        traced = []
    accepted  = run_step5(cfg, traced)                    if 5 in steps else traced
    tagged    = run_step6(cfg, accepted)                  if 6 in steps else accepted

    if generate_count is not None:
        if generate_count < 0:
            logger.error("generate_count must be >= 0")
            sys.exit(1)
        before = len(tagged)
        tagged = tagged[: generate_count]
        logger.info("Final output limited by generate_count: {} → {}", before, len(tagged))

    if 7 in steps:
        run_step7(cfg, tagged)

    logger.info("Pipeline complete. Final records: {}", len(tagged))
    _list_saved_datasets(cfg)


if __name__ == "__main__":
    main()

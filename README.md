# Math Synthetic Data Pipeline 🧮

A comprehensive data processing pipeline for generating and refining high-quality synthetic math training datasets. This pipeline integrates multiple open-source math problem datasets, performs difficulty scaling, format conversion, model-based reasoning annotation, and verification.

## Overview

The pipeline transforms diverse math problem datasets into a unified, high-quality training corpus through 7 sequential processing steps:

```
Step 1: Load seed data (NuminaMath-CoT, FineMath, S1K, etc.)
        ↓
Step 2: Evol-Instruct difficulty scaling (generate harder variants)
        ↓
Step 3: MCQ → open-ended conversion (unify problem formats)
        ↓
Step 4: DeepSeek-R1 teacher reasoning traces (generate <think> tokens)
        ↓
Step 5: Math-Verify validation + rejection sampling (verify correctness)
        ↓
Step 6: Reasoning-level tagging (assign 0-4 complexity scores)
        ↓
Step 7: Serialize to Parquet (final output dataset)
```

## Features

- **Multi-source dataset ingestion**: Supports 10+ math datasets from Hugging Face Hub
- **Difficulty scaling**: Generate harder/easier variants using Evol-Instruct technique
- **Format standardization**: Convert multiple-choice questions to open-ended format
- **Reasoning annotation**: Generate step-by-step reasoning traces via teacher models
- **Answer verification**: Validate math solutions using symbolic computation + rejection sampling
- **Complexity scoring**: Assign 0-4 reasoning level based on problem structure and solution depth
- **Deterministic output**: All outputs include timestamps to prevent data loss on reruns
- **Distributed processing**: SLURM job submission scripts for cluster execution

## Supported Datasets

- **NuminaMath-CoT**: AI-MO/NuminaMath-CoT
- **FineMath**: HuggingFaceTB/finemath
- **S1K**: simplescaling/s1K
- **LIMO**: GAIR/LIMO
- **OmniMath**: KbsdJames/omni-math
- **TwMath**: twinkle-ai/tw-math-reasoning-2k
- **TeleMath**: netop/TeleMath
- **TheoremQA**: TIGER-Lab/theorem_qa
- **AIME**: Contest math problems
- **DeepMath**: Large-scale mathematical reasoning dataset

## Installation

### Prerequisites
- Python 3.9+
- CUDA 12.0+ (for tensor computation)
- ~100GB disk space (for cached datasets)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd math-pipeline

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with:
# - HF_TOKEN: Your Hugging Face API token (for dataset access)
# - MODEL_API_KEY: API key for teacher model inference (DeepSeek, OpenAI, etc.)
```

## Configuration

Edit `config/config.yaml` to customize pipeline behavior:

```yaml
huggingface:
  token_env: HF_TOKEN
  model_api_key_env: MODEL_API_KEY
  cache_dir: data/hf_cache

datasets:
  numina_math:
    enabled: true
    max_samples: 200
  
  # Enable/disable datasets by setting enabled: true/false
  # Set max_samples: null to use entire dataset
```

### Step-Specific Configuration

Each step may have additional configuration options in `config.yaml`. Check the source code for step-specific parameters:

- **Step 2**: `evol_instruct.mutation_depth`, `difficulty_factors`
- **Step 4**: `teacher_model.model_name`, `inference_batch_size`
- **Step 5**: `verification.timeout`, `rejection_sampling_enabled`
- **Step 6**: `reasoning_tagger.complexity_thresholds`

## Usage

### Run Full Pipeline

```bash
python -m src.pipeline
```

Outputs timestamped parquet files to `data/` with naming format:
```
{domain}_{model}_{source}_{count}_{timestamp}.parquet
```

### Run Specific Steps

```bash
# Run only steps 1 and 2
python -m src.pipeline --steps 1,2

# Dry-run: load and evolve, skip API calls
python -m src.pipeline --dry-run
```

### Merge Output Parquets

After multiple runs, merge parquet files by dataset:

```bash
python merge_parquets_by_dataset.py
```

## Dataset Output Format

Each row in the output parquet contains:

```python
{
    "problem": str,           # Original or converted problem text
    "solution": str,          # Model-generated step-by-step solution
    "answer": str,            # Final numerical answer
    "reasoning_level": int,   # 0-4 complexity score
    "verified": bool,         # Whether answer passed verification
    "metadata": dict {        # Source and processing info
        "source_dataset": str,
        "mutation_depth": int,
        "original_format": str,  # "mcq" or "open_ended"
        "teacher_model": str,
        "think_trace": str     # Reasoning trace if available
    }
}
```

## Reasoning Level Scale

The pipeline assigns complexity scores (0-4):

- **0 – Minimal**: Simple factual recall, no deduction required
- **1 – Basic**: Straightforward connections, single-step logic
- **2 – Intermediate**: Multiple factors/concepts combined (e.g., Algebra + Geometry)
- **3 – Advanced**: Sophisticated multi-dimensional analysis, causal relationships
- **4 – Expert**: Theoretical frameworks, deep counterfactual reasoning, novel synthesis

## Cluster Submission

Run jobs on a SLURM cluster:

```bash
# Submit a single dataset processing job
sbatch run_aime_parallel_8.slurm

# Submit all dataset jobs
bash submit_jobs.sh
```

Each `.slurm` file configures:
- GPU allocation
- Job runtime and memory
- Batch size and parallelization level
- Output logging

## Project Structure

```
├── config/
│   └── config.yaml                 # Main configuration
├── data/
│   ├── checkpoints/                # Intermediate step outputs
│   ├── hf_cache/                   # Hugging Face dataset cache
│   └── [output parquets]           # Final timestamped outputs
├── src/
│   ├── pipeline.py                 # Main orchestrator
│   ├── data_ingestion/             # Dataset loaders
│   ├── evol_instruct/              # Difficulty scaling
│   ├── format_converter/           # MCQ conversion
│   ├── teacher_model/              # Reasoning generation
│   ├── verification/               # Answer validation
│   ├── reasoning_tagger/           # Complexity scoring
│   └── __init__.py
├── requirements.txt                # Python dependencies
├── merge_parquets_by_dataset.py   # Combine parallel outputs
├── [utilities scripts]             # Analysis and testing
└── [SLURM job files]               # Cluster submission scripts
```

## Dependencies

Key dependencies:
- **transformers**, **torch**: Model inference
- **datasets**, **huggingface_hub**: Dataset loading
- **pyarrow**: Parquet file I/O
- **sympy**: Mathematical computation
- **math-verify**: Answer verification engine
- **tenacity**: Retry logic for API calls

See `requirements.txt` for complete list.

## Advanced Usage

### Inspect Dataset Samples

```bash
# View sample problems and solutions
python view_output_samples.py

# Analyze calculator dataset
python inspect_calculator_dataset.py

# Check GLM tool outputs
python inspect_glm.py
```

### Generate Reports

```bash
# Generate pipeline quality report
python generate_final_report.py

# Generate LIMO benchmark analysis
python generate_limo_report.py

# Detailed team documentation
python generate_team_documentation.py
```

### Debugging

Enable verbose logging:

```bash
# Set in pipeline.py or config
LOG_LEVEL=DEBUG python -m src.pipeline
```

Check step outputs:

```bash
python test_math_calculator_integration.py
python test_mutation_quality.py
python test_numina_live.py
```

## Performance Considerations

- **Batch size**: Adjust in config for GPU memory constraints
- **Parallelization**: SLURM scripts use 8 parallel workers by default
- **API rate limits**: Tenacity handles retries; adjust backoff in config
- **Verification timeout**: Set per-step timeout to skip slow problems

## Troubleshooting

### Dataset Download Fails
- Verify `HF_TOKEN` is set and has proper permissions
- Check `data/hf_cache/` disk space
- Try `huggingface-cli login`

### Teacher Model Inference Errors
- Verify `MODEL_API_KEY` is valid
- Check API rate limits and quotas
- Review inference batch size settings

### Verification Fails
- Ensure `sympy` and `math-verify` dependencies installed
- Check answer format matches expected output
- Review verification timeout settings

### Memory Issues
- Reduce `batch_size` in config
- Process datasets one at a time (disable in config)
- Run fewer steps with `--steps` flag

## Contributing

To add a new dataset source:

1. Create a new loader in `src/data_ingestion/` (inherit from `BaseLoader`)
2. Register in `src/data_ingestion/loader.py`
3. Add config section to `config/config.yaml`
4. Test with `python -m src.pipeline --steps 1 --dry-run`

## Citation

If you use this pipeline in research, please cite the underlying datasets:

```bibtex
@misc{numina_math,
  title={NuminaMath: A Math Problem-Solving Dataset},
  publisher={Hugging Face Hub},
  year={2024}
}

@misc{limo,
  title={LIMO: Large-scale Math Problem Corpus},
  author={GAIR},
  year={2024}
}

# Add citations for other datasets as needed
```

## License

[Specify license here]

## Support

For issues, questions, or feedback:
- Open an issue on GitHub
- Check existing notebooks and analysis scripts for examples
- Review test files for usage patterns

---

**Last Updated**: April 2026  
**Python Version**: 3.9+  
**Status**: Active Development

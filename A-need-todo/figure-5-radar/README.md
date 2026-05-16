# Figure-5: Six-Mode Induction Radar Scoring

This folder contains everything needed to reproduce the six-mode induction scoring
that underlies the radar chart in Figure 5, which compares Pre-GIST and Post-GIST
strategy quality along six theoretically grounded induction dimensions.

## Folder Structure

```
figure-5-radar/
├── data/
│   ├── pre_gist_100.json       # 100 worst-performing pre-training strategies
│   └── post_gist_100.json      # 100 best-performing post-training strategies
├── prompts/
│   └── six_mode_scoring.txt    # Verbatim scoring prompt (Figure 9 in paper)
├── results/                    # Auto-created; holds scored outputs
├── select_samples.py           # Step 1: extract samples from raw eval shards
├── score.py                    # Step 2: run API scoring
├── requirements.txt
└── README.md
```

## Quick Start

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### 1. (Optional) Re-generate the data files

The `data/` folder already contains the pre-selected 100-sample files.
Skip this step unless you want to regenerate them from the raw shards.

```bash
python select_samples.py \
    --pre-dir  ../../checkpoints_eval_no_verl/habit/Qwen3-4B/habit/Qwen3-4B \
    --post-dir ../../checkpoints_eval_no_verl/MIST/Qwen3-semi/MIST/Qwen3-semi \
    --out-dir  data \
    --n 100
```

Selection logic:
- **Pre-GIST** (`pre_gist_100.json`): 100 rollouts with the **lowest** `reward.correctness`,
  stratified across `problem_type` so no single task type dominates.
- **Post-GIST** (`post_gist_100.json`): 100 rollouts with the **highest** `reward.correctness`,
  stratified in the same way.

### 2. Score with Gemini (recommended — matches the paper)

```bash
export GEMINI_API_KEY="your-key-here"

# Score pre-GIST
python score.py \
    --input   data/pre_gist_100.json \
    --api     gemini \
    --model   gemini-2.5-flash \
    --output  results/pre_gist_scores.json

# Score post-GIST
python score.py \
    --input   data/post_gist_100.json \
    --api     gemini \
    --model   gemini-2.5-flash \
    --output  results/post_gist_scores.json
```

### 2 (alt). Score with OpenAI

```bash
export OPENAI_API_KEY="your-key-here"

python score.py \
    --input   data/pre_gist_100.json \
    --api     openai \
    --model   gpt-4o \
    --output  results/pre_gist_scores.json

python score.py \
    --input   data/post_gist_100.json \
    --api     openai \
    --model   gpt-4o \
    --output  results/post_gist_scores.json
```

### 3. Aggregate results into radar summary

```bash
python score.py --summarize \
    --pre-scores  results/pre_gist_scores.json \
    --post-scores results/post_gist_scores.json \
    --summary-out results/radar_summary.json
```

This prints a table of per-dimension mean scores and writes `results/radar_summary.json`:

```json
{
  "pre_gist":  {
    "compositional": 2.1, "analogical": 1.8,
    "pattern_extrapolation": 2.0, "procedural": 2.3,
    "constraint_based": 1.9, "schema_induction": 2.1,
    "composite_mean": 2.03, "n_scored": 100, "n_valid_responses": 98
  },
  "post_gist": {
    "compositional": 3.8, "analogical": 3.5,
    ...
  }
}
```

Use these values directly as the six radar axes.

## Advanced Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | 8 | Number of concurrent API threads |
| `--max-retries` | 3 | Retries per sample; exponential back-off on rate limits |
| `--prompt-file` | `prompts/six_mode_scoring.txt` | Path to scoring prompt template |

The script **auto-checkpoints** every 10 completed samples. If interrupted,
re-running the same command resumes from where it left off (already-scored
`rollout_id`s are skipped).

## Six Induction Modes (Summary)

| Mode | What it measures |
|------|-----------------|
| **Compositional** | Combines multiple atomic sub-rules into a composite procedure |
| **Analogical** | Draws structural correspondences across examples |
| **Pattern Extrapolation** | Extracts a pattern extending beyond demonstrated instances |
| **Procedural** | Abstracts a step-wise algorithm from input-output pairs |
| **Constraint-based** | Identifies implicit constraints / boundary conditions |
| **Schema-induction** | Classifies the task type at a meta level |

Scores are integers 0-5; N/A is used when a mode is structurally inapplicable.
The composite is the mean over applicable (non-N/A) modes only.
See `prompts/six_mode_scoring.txt` for the full rubric.

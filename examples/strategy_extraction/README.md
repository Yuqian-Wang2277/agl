# Strategy Extraction Training - Stage 1

This example implements the first stage of a two-stage self-evolution model training pipeline. In this stage, the model learns to extract general, executable, and transferable problem-solving strategies from few-shot examples.

## Project Overview

**Goal**: Train a model to analyze a set of example problems and solutions, then summarize the underlying problem-solving strategy in a structured format.

**Key Features**:
- No specific problem solving - pure strategy extraction
- Model learns to wrap strategies in `<strategy>...</strategy>` tags
- Few-shot examples come from the same problem type
- Flexible data paths and configurable sampling

## Data Structure

### Directory Layout

```
/home/test/test16/chenlu/projects/LLMReflection/data/
├── train_20k/                    # Training data (configurable)
│   ├── simple_arithmetic_json_subtasks/
│   │   ├── one_digit.json
│   │   ├── two_digit.json
│   │   └── three_digit.json
│   ├── operators/
│   ├── color/
│   └── ... (110 problem types)
└── val_set/                      # Validation data (configurable)
    ├── problem_type_1/
    └── ...
```

### JSON Format

Each JSON file contains:
```json
{
    "task": "simple_arithmetic_json_subtasks",
    "subtask": "one_digit",
    "description": "A simple one-digit addition task",
    "examples": [
        {
            "input": "1 + 4 = ",
            "target": ["5"]
        },
        {
            "input": "4 + 9 = ",
            "target": ["13"]
        }
    ]
}
```

## Training Process

### 1. Data Loading

The system:
- Scans all problem type directories
- Loads examples from JSON files
- Organizes examples by problem type

### 2. Few-Shot Sampling

For each training sample:
- Randomly selects a problem type
- Randomly selects N examples (where N ∈ [fewshot_min, fewshot_max])
- Tracks used examples to maximize coverage
- Resets when a problem type is exhausted

**Sampling Strategy Benefits**:
- Wide coverage across problem types
- Minimal repetition of examples
- Balanced sampling across types

### 3. Prompt Construction

**System Prompt**:
```
You are an expert at analyzing problem-solving patterns. Your task is to 
examine a set of example problems and their solutions, then extract a 
general, executable, and transferable problem-solving strategy.

Requirements:
1. Analyze the given examples carefully
2. Identify the common problem-solving approach
3. Summarize the strategy in a clear, step-by-step manner
4. Your strategy should be general enough to apply to similar problems
5. Wrap your strategy within <strategy>...</strategy> tags
```

**User Prompt** (formatted few-shot examples):
```
Here are some example problems and their solutions:

Example 1:
Problem: 1 + 4 = 
Solution: 5

Example 2:
Problem: 4 + 9 = 
Solution: 13

Example 3:
Problem: 5 + 0 = 
Solution: 5

Based on these examples, extract the problem-solving strategy.
```

### 4. Reward Function

The model receives a **format reward**:
- **1.0**: Output contains properly formatted `<strategy>...</strategy>` tags with non-empty content
- **0.0**: Missing tags, malformed tags, or empty content

**Strict Checking**:
- Must have opening tag `<strategy>`
- Must have closing tag `</strategy>`
- Must have non-empty content between tags
- Must have exactly one strategy block

## Usage

### Basic Training

```bash
python train_strategy.py \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --train-subdir train_20k \
    --val-subdir val_set \
    --model-path /home/test/test16/chenlu/model/Qwen3-8B \
    --fewshot-min 3 \
    --fewshot-max 8 \
    --num-train-samples 1000 \
    --num-val-samples 200 \
    --n-runners 10
```

### Training with LoRA

```bash
python train_strategy.py \
    --model-path /home/test/test16/chenlu/model/Qwen3-8B \
    --lora \
    --lora-rank 32
```

### Using External Store

```bash
# Terminal 1: Start store server
agl store --port 9999

# Terminal 2: Run training
AGL_MANAGED_STORE=0 python train_strategy.py \
    --external-store-address http://localhost:9999
```

### Debug Mode

```bash
python train_strategy.py --debug
```

## Configuration Parameters

### Data Configuration

- `--data-base-path`: Base path for data directory (default: `/home/test/test16/chenlu/projects/LLMReflection/data/`)
- `--train-subdir`: Training data subdirectory (default: `train_20k`)
- `--val-subdir`: Validation data subdirectory (default: `val_set`)

### Model Configuration

- `--model-path`: Path to the model (default: `/home/test/test16/chenlu/model/Qwen3-8B`)
- `--lora`: Enable LoRA training
- `--lora-rank`: LoRA rank when enabled (default: 32)

### Few-Shot Configuration

- `--fewshot-min`: Minimum number of examples (default: 3)
- `--fewshot-max`: Maximum number of examples (default: 8)

### Dataset Size

- `--num-train-samples`: Number of training samples (default: 1000)
- `--num-val-samples`: Number of validation samples (default: 200)

### Training Configuration

- `--n-runners`: Number of parallel runners (default: 10)
- `--debug`: Enable debug logging

### Infrastructure

- `--external-store-address`: External store URL (e.g., `http://localhost:9999`)

## Implementation Details

### File Structure

```
examples/strategy_extraction/
├── __init__.py              # Package initialization
├── README.md                # This file
├── config.py                # Configuration classes and VERL config
├── data_loader.py           # Data loading and sampling logic
├── prompts.py              # System and user prompt templates
├── reward.py                # Format reward computation
├── strategy_agent.py        # Agent implementation
└── train_strategy.py        # Training entry point
```

### Key Components

**`StrategyConfig`** (`config.py`):
- Dataclass holding all configuration parameters
- Provides sensible defaults
- Used to generate VERL configuration

**`load_problem_types()`** (`data_loader.py`):
- Scans data directory for problem types
- Loads all JSON files
- Returns `{problem_type: [examples]}` dictionary

**`sample_fewshot_examples()`** (`data_loader.py`):
- Samples N examples from a problem type
- Maintains exclude_indices to avoid repetition
- Resets when problem type is exhausted

**`create_strategy_dataset()`** (`data_loader.py`):
- Generates training/validation datasets
- Balances sampling across problem types
- Ensures wide coverage

**`compute_format_reward()`** (`reward.py`):
- Strictly validates strategy tag format
- Returns binary reward (0.0 or 1.0)
- Logs detailed failure reasons in debug mode

**`StrategyExtractionAgent`** (`strategy_agent.py`):
- Implements `LitAgent[StrategyTask]`
- Constructs prompts from examples
- Calls LLM via OpenAI API
- Returns format reward

## Expected Output

After training, the model should consistently produce outputs like:

```
<strategy>
To solve single-digit addition problems:
1. Identify the two numbers to be added
2. Count up from the first number by the value of the second number
3. The result is the sum of the two numbers
4. Express the answer as a single integer

This strategy applies to any addition of two single-digit numbers,
where the result may be single or double digits.
</strategy>
```

## Monitoring Training

During training, you can monitor:
- Format reward rate (should increase over time)
- Problem type distribution (should be balanced)
- Sample output logs (in debug mode)
- WandB metrics (if configured)

## Next Steps (Stage 2)

After stage 1 training completes:
1. The model will have learned to extract strategies from examples
2. Stage 2 will train the model to apply strategies to solve actual problems
3. The stage 2 prompt will include both strategy and a specific problem to solve

## Troubleshooting

### Issue: "No problem types found"
- Check that `--data-base-path` and `--train-subdir` are correct
- Verify that subdirectories contain JSON files with "examples" field

### Issue: "fewshot_max exceeds available examples"
- Some problem types may have fewer examples than requested
- Reduce `--fewshot-max` or ensure all problem types have sufficient examples

### Issue: "Reward stays at 0.0"
- Model may need more training steps
- Check debug logs to see what format errors occur
- Verify that system prompt is correctly instructing tag usage

### Issue: "Memory error during training"
- Reduce `--num-train-samples`
- Reduce `--n-runners`
- Enable LoRA with `--lora`

## References

- [Agent Lightning Documentation](https://microsoft.github.io/agent-lightning/)
- [VERL Algorithm Documentation](https://verl.readthedocs.io/)




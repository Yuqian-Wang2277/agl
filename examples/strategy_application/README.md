# Strategy Application Training - Stage 2

This example implements the second stage of a two-stage self-evolution model training pipeline. In this stage, the model learns to apply problem-solving strategies (extracted in Stage 1) to solve actual problems.

## Project Overview

**Goal**: Train a model to apply a given problem-solving strategy to solve a specific problem, producing answers wrapped in `<answer>...</answer>` tags.

**Key Features**:
- Applies strategies extracted from Stage 1
- Supports two training modes: same-domain and cross-domain
- Strict answer format validation (`<answer>...</answer>` tags)
- Multi-stage reward computation: format → consistency → correctness
- Dynamic batch sampling to ensure learnable rewards

## Architecture

The training process follows this flow:

1. **Online Strategy Extraction**: During each rollout, uses Stage 1 model to extract strategies from few-shot examples online
2. **Problem Solving**: Applies the extracted strategy to solve a problem
3. **Answer Generation**: Produces answer wrapped in `<answer>...</answer>` tags
4. **Reward Computation**: Multi-stage reward:
   - Format check (must be 1.0 to continue)
   - Strategy consistency check (must meet threshold)
   - Answer correctness check (final reward)

## Data Structure

### Directory Layout

```
/home/test/test16/chenlu/projects/LLMReflection/data/
├── train_20k/                    # Training data
│   ├── simple_arithmetic_json_subtasks/
│   │   ├── one_digit.json
│   │   └── ...
│   └── ... (110 problem types)
└── test-bbh/                     # Validation data
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
        }
    ]
}
```

## Training Modes

### Same Domain Mode

- Extracts strategy from few-shot examples of a problem type
- Solves a problem from the same problem type
- Strategy and problem come from the same domain

### Cross Domain Mode

- Extracts strategy from few-shot examples of a source problem type
- Solves a problem from a target problem type
- Uses similarity matching to find the best strategy for the problem
- Tests transferability of strategies across domains

## Training Process

### 1. Data Loading

The system:
- Loads problem types from data directory
- For same-domain: samples few-shot examples and problem from same type
- For cross-domain: samples few-shot examples from source type, problem from target type
- Uses similarity matching for cross-domain mode to select relevant few-shot examples
- **Note**: Strategies are NOT pre-generated during data loading; they are extracted online during training

### 2. Online Strategy Extraction (During Rollout)

- Each rollout dynamically calls Stage 1 model to extract strategies from few-shot examples
- Uses the same prompt format as Stage 1 training
- Strategies are generated fresh for each training sample, ensuring alignment with current model state

### 3. Problem Solving

- Agent receives: few-shot examples (for strategy extraction) + problem
- First extracts strategy using Stage 1 model
- Then applies the extracted strategy to solve the problem using Stage 2 model
- Output must be wrapped in `<answer>...</answer>` tags

### 4. Reward Computation

**Stage 1: Format Check**
- Checks for `<answer>...</answer>` tags
- Must have exactly one answer block with non-empty content
- Returns 1.0 if valid, 0.0 otherwise
- **If format check fails, reward is 0.0 and no further checks are performed**

**Stage 2: Strategy Consistency**
- Extracts steps from strategy
- Checks if answer follows the strategy steps
- Returns: 1.0 (all steps), 0.7 (>70%), 0.4 (>40%), 0.0 (otherwise)
- **If consistency < threshold, reward is 0.0 and correctness check is skipped**

**Stage 3: Answer Correctness**
- Tries exact match (normalized) = 1.0
- Tries numeric tolerance (if numeric, error < 2%) = 0.8
- Tries F1 score (if F1 > 0.5) = 0.5
- Otherwise = 0.0

## Usage

### Using the Training Script

```bash
cd /home/test/test16/chenlu/projects/agent-lightning

# Same domain mode
python examples/strategy_application/train_strategy_application.py \
    --stage2-mode same_domain \
    --stage1-model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --train-subdir train_20k \
    --model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --wandb-project StrategyApplication \
    --wandb-experiment stage2_same_domain

# Cross domain mode
python examples/strategy_application/train_strategy_application.py \
    --stage2-mode cross_domain \
    --stage1-model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --train-subdir train_20k \
    --model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --embedding-model-path BAAI/bge-large-en-v1.5 \
    --wandb-project StrategyApplication \
    --wandb-experiment stage2_cross_domain
```

### Using the Shell Script

```bash
# Same domain mode
./examples/strategy_application/scripts/train.sh \
    --stage1-model-path /home/test/test16/chenlu/model/Qwen3-4B

# Cross domain mode
./examples/strategy_application/scripts/train.sh \
    --stage2-mode cross_domain \
    --stage1-model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --embedding-model-path BAAI/bge-large-en-v1.5
```

## Configuration

Key configuration parameters in `config.py`:

- `stage2_mode`: "same_domain" or "cross_domain"
- `stage1_model_path`: Path to Stage 1 model for strategy extraction
- `strategy_consistency_threshold`: Threshold for consistency check (default: 0.5)
- `answer_correctness_numeric_tolerance`: Numeric tolerance (default: 0.02)
- `answer_correctness_f1_threshold`: F1 threshold (default: 0.5)
- `embedding_model_path`: Embedding model for cross-domain matching
- `batch_min_learnable_reward`: Minimum reward for learnable samples (default: 0.0)
- `batch_max_retry`: Maximum retry attempts for batch sampling (default: 10)

## Output Files

Training generates the following structure:

```
checkpoints_stage2/
├── {experiment_id}/
│   ├── logs/
│   │   └── train_{experiment_id}.log
│   ├── training_configs/
│   │   └── config_{experiment_id}.json
│   ├── rollout_traces/
│   │   └── rollout_traces.jsonl
│   ├── validation_outputs/
│   │   ├── validation_global_step0.json
│   │   ├── validation_global_step50.json
│   │   └── ...
│   └── global_step_50/  # VERL checkpoints
```

### Rollout Traces

Each trace contains:
- Input: fewshot_examples, problem, ground_truth, extracted_strategy
- Output: raw_response, extracted_answer, answer_length
- Reward: final_reward, format_reward, consistency_reward, correctness_reward

### Validation Outputs

Similar structure to rollout traces, saved at validation steps.

## Key Components

**`StrategyApplicationConfig`** (`config.py`):
- Dataclass holding all configuration parameters
- Provides sensible defaults

**`StrategyExtractor`** (`strategy_extractor.py`):
- Uses Stage 1 model to extract strategies from few-shot examples
- Extracts strategies online during each rollout
- Supports batch extraction for efficiency

**`SimilarityMatcher`** (`similarity_matcher.py`):
- Uses sentence-transformers for embeddings
- Finds best matching strategies for problems
- Used in cross-domain mode

**`create_strategy_application_dataset()`** (`data_loader.py`):
- Generates training/validation datasets with few-shot examples and problems
- **Does NOT pre-generate strategies** - strategies are extracted online during training
- Supports same-domain and cross-domain modes
- Uses similarity matching for cross-domain few-shot selection
- Saves dataset info to checkpoint directory

**`compute_stage2_reward()`** (`reward.py`):
- Multi-stage reward computation
- Format → consistency → correctness
- Returns detailed reward breakdown

**`StrategyApplicationAgent`** (`strategy_application_agent.py`):
- Implements `LitAgent[StrategyApplicationTask]`
- **Online strategy extraction**: Calls Stage 1 model to extract strategy from few-shot examples during each rollout
- Applies extracted strategy to solve problem using Stage 2 model
- Saves answers to traces and validation outputs

## Expected Output

After training, the model should consistently produce outputs like:

```
<answer>
5
</answer>
```

Or for more complex problems:

```
<answer>
The result is 42, which is obtained by following the strategy step-by-step.
</answer>
```

## Monitoring Training

During training, you can monitor:
- Format reward rate (should increase over time)
- Strategy consistency rate
- Answer correctness rate
- Final reward distribution
- WandB metrics (if configured)

## Troubleshooting

### Issue: "Stage 1 model path is invalid"
- Ensure the model path is correct
- Check that the model directory exists and contains model files
- Verify that the model can be loaded by the LLM server

### Issue: "No strategies available for source domain"
- In cross-domain mode, ensure source domain has enough examples
- Check that strategy extraction is working correctly

### Issue: "All rewards are 0"
- Check that answer format is correct (`<answer>...</answer>`)
- Verify strategy consistency threshold is appropriate
- Ensure ground truth answers are correct

### Issue: "sentence-transformers not installed"
- Install with: `pip install sentence-transformers`
- Required for cross-domain mode

## Next Steps

After Stage 2 training completes:
1. The model will have learned to apply strategies to solve problems
2. Evaluate on test sets to measure performance
3. Compare same-domain vs cross-domain performance
4. Analyze strategy transferability

## Related Documentation

- [Stage 1: Strategy Extraction](../strategy_extraction/README.md)
- [Agent Lightning Documentation](../../docs/)


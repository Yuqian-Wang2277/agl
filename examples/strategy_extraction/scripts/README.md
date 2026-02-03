# Training Scripts

This directory contains bash scripts for running training with different configurations.

## Available Scripts

### `train.sh`

Default training script that uses configuration from `config.py`.

**Usage:**
```bash
./scripts/train.sh
```

**Configuration:**
- All parameters use defaults from `StrategyConfig` in `config.py`
- You can modify `config.py` to change default settings
- Or override specific parameters by editing `train.sh`

**Output:**
- Logs: `checkpoints/logs/train_YYYYMMDD_HHMMSS.log`
- Checkpoints: `checkpoints/`
- Model outputs: Saved to log file if `save_full_output=True` in config

## Adding New Scripts

You can create additional training scripts for different configurations:

```bash
# Example: train_debug.sh
#!/bin/bash
python examples/strategy_extraction/train_strategy.py \
    --debug \
    --save-full-output \
    ...
```

## Configuration Options

Key configuration options in `config.py`:

- `save_full_output`: Whether to save complete model output for each rollout (default: `True`)
- `checkpoint_dir`: Directory to save checkpoints (default: `"./checkpoints"`)
- `wandb_project`: WandB project name
- `wandb_experiment`: WandB experiment/run name

See `config.py` for all available options.



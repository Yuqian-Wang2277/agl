#!/usr/bin/env bash
# =============================================================================
# train.sh — Actor-Judge Phase II full training launch
#
# Usage:
#   bash scripts/train.sh [--sft_checkpoint PATH] [--epochs N] [--extra ARGS...]
#
# Examples:
#   # Start from SFT checkpoint (recommended):
#   bash scripts/train.sh --sft_checkpoint /path/to/actor_hf
#
#   # Ablation B — pure sparse reward (also freezes Judge):
#   bash scripts/train.sh --sft_checkpoint /path/to/actor_hf \
#       --dense_reward_alpha 0 --freeze_judge
#
#   # Ablation E — disable UCB replay:
#   bash scripts/train.sh --sft_checkpoint /path/to/actor_hf \
#       --disable_ucb_replay
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SFT_CHECKPOINT=""
TOTAL_EPOCHS=5
K=8
ALPHA=0.3
DENSE_REWARD_ALPHA=0.3
FREEZE_JUDGE=""
DISABLE_UCB_REPLAY=""
WANDB_RUN_NAME="phase2_co_evolution_$(date +%Y%m%d_%H%M%S)"
RESUME_FROM=""
NUM_TRAIN_SAMPLES=20000
EXTRA_ARGS=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sft_checkpoint)   SFT_CHECKPOINT="$2";      shift 2 ;;
        --epochs)           TOTAL_EPOCHS="$2";         shift 2 ;;
        --K)                K="$2";                    shift 2 ;;
        --alpha)            ALPHA="$2";                shift 2 ;;
        --dense_reward_alpha) DENSE_REWARD_ALPHA="$2"; shift 2 ;;
        --freeze_judge)     FREEZE_JUDGE="--freeze_judge"; shift ;;
        --disable_ucb_replay) DISABLE_UCB_REPLAY="--disable_ucb_replay"; shift ;;
        --wandb_run_name)   WANDB_RUN_NAME="$2";            shift 2 ;;
        --resume_from)      RESUME_FROM="$2";               shift 2 ;;
        --num_train_samples) NUM_TRAIN_SAMPLES="$2";        shift 2 ;;
        --extra)            EXTRA_ARGS="$2";                shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agl

cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# ── Accelerate FSDP config ────────────────────────────────────────────────────
# Expects accelerate_fsdp.yaml in the project root; generate a minimal one
# if it doesn't exist yet.
ACCEL_CFG="$PROJECT_DIR/accelerate_fsdp.yaml"
if [[ ! -f "$ACCEL_CFG" ]]; then
    echo "Generating default accelerate_fsdp.yaml ..."
    cat > "$ACCEL_CFG" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
YAML
fi

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    accelerate launch
    --config_file "$ACCEL_CFG"
    train.py
    --actor_sft_checkpoint "$SFT_CHECKPOINT"
    --total_epochs "$TOTAL_EPOCHS"
    --K "$K"
    --alpha "$ALPHA"
    --dense_reward_alpha "$DENSE_REWARD_ALPHA"
    --wandb_run_name "$WANDB_RUN_NAME"
    --num_train_samples "$NUM_TRAIN_SAMPLES"
)

[[ -n "$RESUME_FROM" ]] && CMD+=("--resume_from_checkpoint" "$RESUME_FROM")

[[ -n "$FREEZE_JUDGE"      ]] && CMD+=("$FREEZE_JUDGE")
[[ -n "$DISABLE_UCB_REPLAY" ]] && CMD+=("$DISABLE_UCB_REPLAY")
[[ -n "$EXTRA_ARGS"        ]] && CMD+=($EXTRA_ARGS)

echo "========================================================"
echo "Launch command:"
echo "  ${CMD[*]}"
echo "========================================================"

exec "${CMD[@]}"

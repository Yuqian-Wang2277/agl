#!/usr/bin/env bash
# =============================================================================
# train.sh — Actor-Judge Phase II full training launch
#
# Usage:
#   bash scripts/train.sh [--sft_checkpoint PATH] [--epochs N]
#       [--num_train_samples N] [--rollout_steps_per_epoch N] [--val_steps N] [--save_steps N]
#       [--wandb_project NAME] [--wandb_run_name NAME]
#       [--min_buffer_size N]
#       [--dry_run_val_size N] [--extra ARGS...]
#   Default: --rollout_steps_per_epoch 250, --rollout_partition_mode stratified,
#   --val_steps 50, --save_steps 50 (aligns with config.py; train.py default is 100).
#   (10 epochs × 250 batches = full data coverage once; shuffle = random cap each epoch).
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
#
# Val item dumps (train.py; via --extra or extend this script):
#   --val_item_storage jsonl|both   # sidecar eval_*_items.jsonl
#   --val_log_items_wandb_table
#
# Judge warmup modes (train.py flags via --extra):
#   --judge_warmup_mode always   # default: run warmup + save to checkpoints/.../judge_warmup_latest
#   --judge_warmup_mode cold     # skip warmup
#   --judge_warmup_mode reuse --judge_init_checkpoint /path/to/judge_model.pt
#
# Default behavior in this launcher:
#   if ./judge_warmup_ckpt/judge_model.pt exists, it auto-enables
#   --judge_warmup_mode reuse --judge_init_checkpoint <that path>
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SFT_CHECKPOINT=""
TOTAL_EPOCHS=250
K=8
ALPHA=0.3
DENSE_REWARD_ALPHA=0.3
FREEZE_JUDGE=""
DISABLE_UCB_REPLAY=""
WANDB_PROJECT=""
WANDB_RUN_NAME="phase2_co_evolution_$(date +%Y%m%d_%H%M%S)"
RUN_NAME=""
CHECKPOINT_ROOT=""
RESUME_FROM=""
NUM_TRAIN_SAMPLES=20000
ROLLOUT_STEPS_PER_EPOCH=50
ROLLOUT_PARTITION_MODE=stratified
ROLLOUT_PARTITION_SEED=42
VAL_STEPS=50
SAVE_STEPS=50
MIN_BUFFER_SIZE=""
DRY_RUN_VAL_SIZE=""
EXTRA_ARGS=""
JUDGE_WARMUP_CKPT=""
# Passed to train.py --vllm_max_num_batched_tokens (empty = train.py default, usually 4096).
VLLM_MAX_NUM_BATCHED_TOKENS=""

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
        --wandb_project)    WANDB_PROJECT="$2";             shift 2 ;;
        --wandb_run_name)   WANDB_RUN_NAME="$2";            shift 2 ;;
        --run_name)         RUN_NAME="$2";                  shift 2 ;;
        --checkpoint_root)  CHECKPOINT_ROOT="$2";           shift 2 ;;
        --resume_from)      RESUME_FROM="$2";               shift 2 ;;
        --num_train_samples) NUM_TRAIN_SAMPLES="$2";        shift 2 ;;
        --rollout_steps_per_epoch) ROLLOUT_STEPS_PER_EPOCH="$2"; shift 2 ;;
        --rollout_partition_mode) ROLLOUT_PARTITION_MODE="$2"; shift 2 ;;
        --rollout_partition_seed) ROLLOUT_PARTITION_SEED="$2"; shift 2 ;;
        --val_steps)          VAL_STEPS="$2";                shift 2 ;;
        --save_steps)         SAVE_STEPS="$2";               shift 2 ;;
        --min_buffer_size)   MIN_BUFFER_SIZE="$2";          shift 2 ;;
        --dry_run_val_size)  DRY_RUN_VAL_SIZE="$2";         shift 2 ;;
        --judge_warmup_ckpt) JUDGE_WARMUP_CKPT="$2";        shift 2 ;;
        --vllm_max_num_batched_tokens) VLLM_MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --extra)            EXTRA_ARGS="$2";                shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agl

# Single-machine Phase II: vLLM uses local Ray. A stale RAY_ADDRESS (e.g. from a
# Slurm/other job pointing at 11.11.x.x:6379) makes ray.init() hang ~10+ min then
# ConnectionError. Multi-node Ray users: export RAY_ADDRESS before this script.
unset RAY_ADDRESS RAY_HEAD_IP 2>/dev/null || true

cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Write per-rank Python tracebacks to /tmp/torch_elastic_error.json on crash.
# Without this the distributed launcher only shows "exitcode 1, traceback: N/A".
export TORCHELASTIC_ERROR_FILE="${TORCHELASTIC_ERROR_FILE:-/tmp/torch_elastic_error_$(date +%Y%m%d_%H%M%S).json}"
echo "Rank error file: $TORCHELASTIC_ERROR_FILE"
# Reduces allocator fragmentation during long FSDP + Judge backward (PyTorch 2.x).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -z "$CHECKPOINT_ROOT" ]] && CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints_actor_judge"

# ── Cache / artifact cleanup ──────────────────────────────────────────
# Checkpoints live under checkpoints_actor_judge/<run_name>/ (per-run isolation).
# Do not rm the whole tree — that would delete unrelated experiments.
# Always clear the vLLM ↔ FSDP shared-memory weight hand-off directory.
rm -rf "/dev/shm/actor_weight_tmp" 2>/dev/null || true

[[ -z "$RUN_NAME" ]] && RUN_NAME="$WANDB_RUN_NAME"

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
    --checkpoint_root "$CHECKPOINT_ROOT"
    --run_name "$RUN_NAME"
    --num_train_samples "$NUM_TRAIN_SAMPLES"
    --rollout_steps_per_epoch "$ROLLOUT_STEPS_PER_EPOCH"
    --rollout_partition_mode "$ROLLOUT_PARTITION_MODE"
    --rollout_partition_seed "$ROLLOUT_PARTITION_SEED"
    --val_steps "$VAL_STEPS"
    --save_steps "$SAVE_STEPS"
)

[[ -n "$WANDB_PROJECT" ]] && CMD+=("--wandb_project" "$WANDB_PROJECT")

[[ -n "$MIN_BUFFER_SIZE" ]] && CMD+=("--min_buffer_size" "$MIN_BUFFER_SIZE")
[[ -n "$DRY_RUN_VAL_SIZE" ]] && CMD+=("--dry_run_val_size" "$DRY_RUN_VAL_SIZE")

[[ -n "$RESUME_FROM" ]] && CMD+=("--resume_from_checkpoint" "$RESUME_FROM")

[[ -n "$VLLM_MAX_NUM_BATCHED_TOKENS" ]] && CMD+=("--vllm_max_num_batched_tokens" "$VLLM_MAX_NUM_BATCHED_TOKENS")

[[ -n "$FREEZE_JUDGE"      ]] && CMD+=("$FREEZE_JUDGE")
[[ -n "$DISABLE_UCB_REPLAY" ]] && CMD+=("$DISABLE_UCB_REPLAY")

# Prefer a pre-warmed Judge checkpoint by default (if present), unless user
# explicitly controls warmup mode/init checkpoint through --extra.
if [[ -z "$JUDGE_WARMUP_CKPT" ]]; then
    JUDGE_WARMUP_CKPT="$PROJECT_DIR/judge_warmup_ckpt/judge_model.pt"
fi
if [[ -f "$JUDGE_WARMUP_CKPT" ]]; then
    if [[ "$EXTRA_ARGS" != *"--judge_warmup_mode"* && "$EXTRA_ARGS" != *"--judge_init_checkpoint"* ]]; then
        CMD+=("--judge_warmup_mode" "reuse" "--judge_init_checkpoint" "$JUDGE_WARMUP_CKPT")
        echo "Using pre-warmed Judge: $JUDGE_WARMUP_CKPT"
    fi
fi

[[ -n "$EXTRA_ARGS"        ]] && CMD+=($EXTRA_ARGS)

echo "========================================================"
echo "Launch command:"
echo "  ${CMD[*]}"
echo "========================================================"

exec "${CMD[@]}"

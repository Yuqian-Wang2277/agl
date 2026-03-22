#!/usr/bin/env bash
# =============================================================================
# dry_run_phase2.sh — 10 题 × 2 epoch 的 Phase II 冒烟（管线、Ray/vLLM、步级验证/存盘）
#
# --- 1) Dry run 会输出 WandB 数据吗？ ---
# 会。默认 WANDB_MODE=offline：指标与 Table 写入本机 run 目录（通常在仓库下的 wandb/），
# 不上传到 wandb.ai。需要云端面板时在同一 shell 里先执行 export WANDB_MODE=online
#（仍可用 WANDB_SILENT=true 减少控制台噪音）。
#
# --- 2) 本脚本在做什么（简要）---
# - Actor：用你给的 HF 目录（SFT/VERL 转出来的 actor_hf）起训。
# - Judge：--judge_warmup_mode reuse，直接加载预热好的 judge_model.pt，跳过长时间 warmup。
# - 数据：--num_train_samples 10、--min_buffer_size 8，让小池子也能触发 Judge ODVA。
# - 验证：--dry_run_val_size 10；--val_steps/--save_steps 2，保证短跑也会触发步级 vLLM 验证与 step_* 存盘。
# - 产物：checkpoint 在 checkpoint_root/run_name/；eval_baseline.json、eval_step_*.json 等同目录。
#
# --- 3) 完整启动示例（按你当前机器上的路径）---
# 在 examples/actor-judge 下执行：
#
#   cd /home/test/test16/chenlu/projects/agent-lightning/examples/actor-judge
#   bash scripts/dry_run_phase2.sh \
#     /home/test/test16/chenlu/projects/agent-lightning/checkpoints_strategy_gen/20260219_111326/global_step_400/actor_hf
#
# Judge 默认目录已是：
#   .../examples/actor-judge/judge_warmup_ckpt/2026-03-20
# 若 Judge 在别处，再设：
#   export JUDGE_INIT_DIR=/path/含/judge_model.pt的目录
#
# 等价写法（用环境变量代替第一个参数）：
#   export SFT_CHECKPOINT=/home/test/test16/chenlu/projects/agent-lightning/checkpoints_strategy_gen/20260219_111326/global_step_400/actor_hf
#   bash scripts/dry_run_phase2.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONUNBUFFERED=1

SFT="${1:-${SFT_CHECKPOINT:-}}"
JUDGE_INIT_DIR="${JUDGE_INIT_DIR:-$PROJECT_DIR/judge_warmup_ckpt/2026-03-20}"

if [[ -z "$SFT" ]]; then
    echo "Set SFT_CHECKPOINT or pass actor SFT dir as first argument."
    exit 1
fi
if [[ ! -f "$JUDGE_INIT_DIR/judge_model.pt" ]]; then
    echo "Missing warmed Judge: $JUDGE_INIT_DIR/judge_model.pt"
    echo "Train one first (warmup_judge.sh) or set JUDGE_INIT_DIR."
    exit 1
fi

RUN_TAG="dry_run_smoke_$(date +%Y%m%d_%H%M%S)"

exec bash "$SCRIPT_DIR/train.sh" \
    --sft_checkpoint "$SFT" \
    --epochs 2 \
    --num_train_samples 10 \
    --min_buffer_size 8 \
    --dry_run_val_size 10 \
    --run_name "$RUN_TAG" \
    --extra "--judge_warmup_mode reuse --judge_init_checkpoint $JUDGE_INIT_DIR --val_item_storage jsonl --val_log_items_wandb_table --val_steps 2 --save_steps 2 --keep_last_k_checkpoints 0 --keep_best_k_checkpoints 0"

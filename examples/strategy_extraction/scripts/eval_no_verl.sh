#!/bin/bash
# Lightweight evaluation entrypoint for StrategyGenerationAgent WITHOUT VERL/Ray.
# This reuses the same data path and model defaults as the training scripts,
# but runs rollouts concurrently via eval_no_verl.py.
#
# ============================================================
#  模式选择（MODE 环境变量）
# ============================================================
#
#  MODE=few-shot（仅答案模型，ICL in-context learning）
#    使用 prompt: answer_generation/ICL(few-shot).toml
#    只需启动答案模型（Qwen3-8B，端口 8200），无需策略模型。
#    --skip-strategy-generation 跳过策略生成调用。
#    fewshot-min/max=3：每道题看 3 个 few-shot 示例后直接回答。
#    类比：婴幼儿靠具体示范解题（无元学习能力，每次需要新示范）。
#
#  MODE=MIST（默认）（策略模型 + 答案模型）
#    使用 prompt: strategy_generation/repetition_controls_2026-04-01.toml
#                 answer_generation/v1.toml
#    需要同时启动策略模型（Qwen3-4B，端口 8100）和答案模型（Qwen3-8B，端口 8200）。
#
#  MODE=MIST+few-shot（策略模型 + 含 few-shot 示例的答案模型）
#    策略生成阶段：与 MIST 模式完全相同。
#      prompt: strategy_generation/repetition_controls_2026-04-01.toml
#      策略模型: Qwen3-4B  端口 8100  no-think  repetition_penalty=1.1
#    答案生成阶段：在 MIST 的基础上，答案模型同时看到 few-shot 示例 + 策略，再回答新题。
#      prompt: answer_generation/mist_fewshot_answer.toml（{examples_text}+{strategy}+{problem}）
#      答案模型: Qwen3-8B  端口 8200  no-think
#    与 habit 模式的核心区别：策略来自独立的外部策略模型（Qwen3-4B），而非答案模型内联生成。
#    fewshot-min/max=3：提供 3 个 few-shot 示例给策略模型和答案模型。
#    需要同时启动策略模型（端口 8100）和答案模型（端口 8200）。
#
#  MODE=mist-inline（单模型 train-free MIST）
#    同一模型先提取两层策略（FIRST_ORDER + SECOND_ORDER），再用策略回答新题。
#    无需单独策略模型，适合闭源 API（GPT-4o、Claude 等）。
#    只需启动答案模型（端口 8200）或配置闭源 API。
#    使用 prompt: answer_generation/mist_inline_strategy.toml（策略提取）
#                 answer_generation/mist_inline_answer.toml（策略作答）
#
#  MODE=0-shot（婴幼儿基线：无元学习能力，直接回答）
#    不提供任何 few-shot 示例，直接要求模型回答新问题。
#    fewshot-min/max=0，跳过策略生成。
#    使用 prompt: answer_generation/zero-shot.toml
#    类比：婴幼儿面对 unseen task，既无示范也无内化策略，只能依赖原始能力。
#    只需启动答案模型（端口 8200）。
#
#  MODE=habit-0-shot（成年人基线：已掌握元学习能力，无需示范）
#    不提供任何 few-shot 示例，但模型先从题目本身自主生成两层策略，再应用策略作答。
#    fewshot-min/max=0，两次 API 调用（策略提取 + 策略作答）。
#    使用 prompt: answer_generation/habit_0shot_strategy.toml（策略提取，仅输入题目）
#                 answer_generation/mist_inline_answer.toml（策略作答）
#    类比：成年人已通过训练内化了元归纳能力，看到新题后能自主识别题型、规划解法，
#          无需再看具体示范。
#    只需启动答案模型（端口 8200）。
#
#  MODE=meta-test（元归纳内化证据测试）
#    核心问题：在完全没有"请生成策略"类指令的情况下，经过训练的模型是否仍然
#    自发先生成策略再解题？若是，则为策略内化的实证证据。
#    指令从"请生成策略"改为完全中性的"请思考这个问题"。
#    fewshot-min/max=0（无任何 few-shot 示例）。
#    仅一次 API 调用（答案模型），底层映射为 Python 层的 0-shot 模式。
#    使用 prompt: answer_generation/meta_test_neutral.toml
#    测量目标：①OOD 准确率（与 habit-0-shot 对比，看 prompt 中性化后性能下降幅度）
#              ②模型输出中是否自发出现 <strategy> 块（质性内化证据）
#    适用场景：已在 GPU 6,7 启动经过训练的 4B 模型（META_TEST_MODEL_BASE_URL/NAME）。
#    默认端口：META_TEST_MODEL_BASE_URL=http://localhost:8200/v1
#              META_TEST_MODEL_NAME=Qwen3-4B（可通过环境变量覆盖）
#
#  MODE=CoT（标准链式推理，1×，task-aware）
#    单次调用，模型在看到新题的同时自由生成推理，然后输出答案。
#    这是 Wei et al. (2022) 意义上的经典 CoT——推理内容围绕具体新题展开，是问题特异性的。
#    模型开启 think 模式（不传 --answer-no-think），内部 <think> 块作为额外推理暂存区。
#    可见输出包含 "推理: ..." 段落和 <answer> 块。
#    Prompt 结构：[示例1: Q→A]  [示例2: Q→A]  [示例3: Q→A]
#                 Q: [新题]  →  推理: ...（自由文本，见到新题后展开）  →  <answer>
#    prompt: answer_generation/std_cot.toml（{examples_text}+{problem}）
#    fewshot-min/max=3，底层映射为 Python 层的 few-shot 模式（单次调用+示例）。
#    只需启动答案模型（端口 8200）。
#
#  MODE=Blind-CoT（盲式链式推理，2×，free format）
#    给模型 K 个示例，先自由生成一段推理过程（chain-of-thought），再产出答案。
#    推理在看到新题之前产生，是问题无关（problem-agnostic）的。两次 API 调用（同一答案模型）。
#    Call 1: 基于 [示例1~K] 自由推理，生成推理文本 r（尚未见到新题）
#            prompt: answer_generation/cot_reasoning.toml（{examples_text}）
#    Call 2: 基于推理 r 回答新问题 Q → A（不保留原始示例）
#            prompt: answer_generation/cot_answer.toml（{strategy}+{problem}）
#    与 CoT 的核心区别：推理在看到新题之前产生（blind to new question）。
#    fewshot-min/max=3，底层映射为 Python 层的 mist-inline 模式。
#    只需启动答案模型（端口 8200）。
#
#  MODE=FS+CoT（示例+自由推理，2×，free format）
#    在 Blind-CoT 的基础上，Call 2 阶段不丢弃原始示例，而是把自由推理文本叠加在示例上
#    一起送给答案模型。两次 API 调用（同一答案模型）。
#    Call 1: 基于 [示例1~K] 生成自由推理 r（与 Blind-CoT Call 1 完全相同）
#            prompt: answer_generation/cot_reasoning.toml（{examples_text}）
#    Call 2: [示例1~K] + 推理 r + 新问题 Q → A
#            prompt: answer_generation/fs_cot_answer.toml（{examples_text}+{strategy}+{problem}）
#    与 Blind-CoT 的区别：答案模型同时可见原始示例和自由推理，而非仅推理。
#    与 habit 的区别：habit 使用 FIRST_ORDER+SECOND_ORDER 结构化格式；FS+CoT 使用自由推理。
#    fewshot-min/max=3，底层映射为 Python 层的 habit 模式。
#    只需启动答案模型（端口 8200）。
#
#  MODE=MIST-single（结构化策略单 pass，1×，struct.）
#    使用与 MIST 完全相同的结构化策略格式（含 FIRST_ORDER_PATTERN、FEWSHOT_LEARNING、
#    SECOND_ORDER_STEPS 等章节），但策略生成和答案产出在同一次 forward pass 完成：
#    先输出完整 <strategy> 块，紧接着输出 <answer>。
#    仅一次 API 调用（答案模型），底层映射为 Python 层的 few-shot 模式。
#    与 MIST 的区别：不需要独立策略模型（Qwen3-4B），仅使用答案模型（Qwen3-8B）。
#    与 mist-inline 的区别：mist-inline 分两次调用；MIST-single 在单次前向传播中
#                          同时生成策略和答案，无策略-答案之间的推理中断。
#    使用 prompt: answer_generation/mist_single_pass.toml（{examples_text}+{problem}）
#    fewshot-min/max=3，只需启动答案模型（端口 8200）。
#
# ============================================================
#  重要 setting 速查（实际生效值，含显式参数 + 隐式 default）
# ============================================================
#
#  [temperature & 采样多样性]
#    默认: --temperature 0.0
#    实际: NUM_SAMPLES_PER_PROBLEM=3 > 1 且 temperature==0 → 自动切换为 temperature=0.7
#          (eval_no_verl.py 内部逻辑，确保多次采样输出互不相同)
#    显式覆盖: --temperature 0.7（或其他值）可绕过自动切换
#    单次确定性推理: --num-samples-per-problem 1 --temperature 0
#
#  [pass@k & seeds]
#    NUM_SAMPLES_PER_PROBLEM=3：每道题独立推理 3 次
#    种子: base_seed=42 → 3 次 rollout 依次使用 seed=42, 43, 44
#    输出指标: pass@1 / pass@2 / pass@3（hard 0/1 和 soft F1 各一份）
#
#  [模型 — MIST 模式]
#    策略模型: Qwen3-4B  端口 8100  think 由 STRATEGY_THINK 控制（默认关闭）
#              repetition_penalty=1.1  prompt_version=repetition_controls_2026-04-01
#    答案模型: Qwen3-8B  端口 8200  think 由 ANSWER_THINK 控制（默认关闭）
#              answer_prompt_version=v1  grounded-proxy-k=1
#
#  [think 模式独立开关]
#    ANSWER_THINK=0（默认）   关闭 answer model think，直接输出 <answer>
#    ANSWER_THINK=1           开启 answer model think，先生成 <think> 再输出答案
#                             建议在 MIST/MIST+few-shot 下设为 1，使 answer model 能按策略步骤推理
#    STRATEGY_THINK=0（默认） 关闭 strategy model think
#    STRATEGY_THINK=1         开启 strategy model think，有助于提升策略归纳质量（截断风险极低）
#    注：CoT 模式强制开启 answer think，忽略 ANSWER_THINK 设置
#    两个开关同时作用于额外测评集（HARDMath2 / Linguini）
#
#  [模型 — MIST+few-shot 模式]
#    策略模型: Qwen3-4B  端口 8100  think 由 STRATEGY_THINK 控制（默认关闭）
#              repetition_penalty=1.1  prompt_version=repetition_controls_2026-04-01
#    答案模型: Qwen3-8B  端口 8200  think 由 ANSWER_THINK 控制（默认关闭；建议设为 1）
#              answer_prompt_version=mist_fewshot_answer
#              （答案 prompt 包含 {examples_text}+{strategy}+{problem}）
#
#  [模型 — few-shot 模式]
#    答案模型: Qwen3-8B  端口 8200  no-think  answer_prompt_version=ICL(few-shot)
#
#  [模型 — 0-shot 模式]
#    答案模型: Qwen3-8B  端口 8200  no-think  answer_prompt_version=zero-shot
#    fewshot-min=0 fewshot-max=0（无示例）
#
#  [模型 — habit-0-shot 模式]
#    答案模型: Qwen3-8B  端口 8200  no-think
#    策略 prompt: habit_0shot_strategy（仅输入题目，无示例）
#    答案 prompt: mist_inline_answer（策略+题目）
#    fewshot-min=0 fewshot-max=0（无示例）
#
#  [模型 — meta-test 模式]
#    答案模型: 已训练 4B 模型  默认端口 8200  no-think
#    prompt:   meta_test_neutral（中性指令，不含策略引导）
#    fewshot-min=0 fewshot-max=0（无示例）
#    Python 层: --mode 0-shot（复用无策略单次调用逻辑）
#
#  [模型 — CoT 模式]
#    答案模型: Qwen3-8B  端口 8200  think 模式开启（不传 --answer-no-think）
#    prompt:   std_cot（{examples_text}+{problem}，单次调用，visible 推理+<answer>）
#    max_tokens: 32768（think block 可消耗 5k–15k tokens，默认 16384 对难题可能截断）
#    fewshot-min=3 fewshot-max=3
#    Python 层: --mode few-shot（单次调用+示例，跳过策略模型）
#
#  [模型 — Blind-CoT 模式]
#    答案模型: Qwen3-8B  端口 8200  no-think
#    Call 1 prompt: cot_reasoning（自由推理提取，{examples_text}）
#    Call 2 prompt: cot_answer（基于推理回答，{strategy}+{problem}）
#    fewshot-min=3 fewshot-max=3
#    Python 层: --mode mist-inline（复用 inline strategy 双调用逻辑）
#
#  [模型 — FS+CoT 模式]
#    答案模型: Qwen3-8B  端口 8200  no-think
#    Call 1 prompt: cot_reasoning（自由推理提取，与 Blind-CoT 相同）
#    Call 2 prompt: fs_cot_answer（示例+推理联合作答，{examples_text}+{strategy}+{problem}）
#    fewshot-min=3 fewshot-max=3
#    Python 层: --mode habit（复用 habit 双调用+示例逻辑）
#
#  [模型 — MIST-single 模式]
#    答案模型: Qwen3-8B  端口 8200  no-think
#    prompt:   mist_single_pass（完整 MIST 结构化策略 + 答案，{examples_text}+{problem}）
#    fewshot-min=3 fewshot-max=3
#    Python 层: --mode few-shot（复用单次调用+示例逻辑，跳过策略模型）
#
#  [数据集]
#    val-subdirs: test-id-subtask + test-ood-task + test-bbh
#    采样模式 (FULL_DATASET=0): per_subtask_fixed，每 subtask 20 条，val-sampling-seed=42
#    采样模式 (FULL_DATASET=1): 全量枚举，~123k 题
#
#  [reward / 评测逻辑]
#    reward-mode=scorer_only，correctness-weight=1.0，format-weight=0.0，scorer-weight=0.0
#    → 纯答案正确率评测，不含 format tag / strategy scorer 评分
#    reward-version=v3，answer-request-retries=3
#
#  [并发]
#    EVAL_CONCURRENCY=64（64 个异步 worker 并发推理）
#
# ============================================================
#  前置步骤
# ============================================================
#
#  [MIST 模式] 分别在两个终端启动两个 vLLM 服务：
#
#   终端 1 — 策略生成模型（Qwen3-4B，端口 8100）：
#     CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-4B \
#         --served-model-name Qwen3-4B \
#         --tensor-parallel-size 2 \
#         --port 8100 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
#   策略 vLLM：建议将默认采样设为 repetition_penalty=1.1（具体参数名见 vllm serve --help，随版本而异）。
#   eval_no_verl.py 默认在请求的 extra_body 中发送 repetition_penalty=1.1；传 --strategy-repetition-penalty 0
#   可省略该字段、完全依赖服务端默认。
#
#   终端 2 — 答案生成模型（Qwen3-8B，端口 8200）：
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-8B \
#         --served-model-name Qwen3-8B \
#         --tensor-parallel-size 4 \
#         --port 8200 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
#  [few-shot / mist-inline 模式] 只需启动答案模型（终端 2，同上）。
#  [mist-inline 闭源 API] 无需本地 vLLM，配置 ANSWER_MODEL_BASE_URL 和 ANSWER_MODEL_NAME 即可。
#
# 运行示例：
#   MODE=MIST           bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=MIST+few-shot  bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=few-shot       bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=mist-inline    bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=habit          bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=0-shot         bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=habit-0-shot   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=meta-test      bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=CoT            bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=Blind-CoT      bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=FS+CoT         bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=MIST-single    bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=few-shot       bash examples/strategy_extraction/scripts/eval_no_verl.sh --max-samples 64
#
# meta-test 自定义模型端口示例（已训练 4B 模型运行在 GPU 6,7，端口 8200）：
#   MODE=meta-test \
#   META_TEST_MODEL_BASE_URL=http://localhost:8200/v1 \
#   META_TEST_MODEL_NAME=Qwen3-4B \
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#
# 可通过环境变量覆盖默认值，例如（MIST 模式）：
#   MODE=MIST \
#   STRATEGY_MODEL_BASE_URL=http://localhost:8100/v1 \
#   STRATEGY_MODEL_NAME=Qwen3-4B \
#   ANSWER_MODEL_BASE_URL=http://localhost:8200/v1 \
#   ANSWER_MODEL_NAME=Qwen3-8B \
#   EVAL_CONCURRENCY=64 \
#   NUM_SAMPLES_PER_PROBLEM=3 \
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#
# think 模式开关示例：
#   # MIST+few-shot：开启 answer think（推荐），使 answer model 按策略步骤推理
#   MODE=MIST+few-shot ANSWER_THINK=1 \
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#
#   # MIST+few-shot：同时开启 strategy think 和 answer think
#   MODE=MIST+few-shot ANSWER_THINK=1 STRATEGY_THINK=1 \
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#
#   # MIST：仅开启 strategy think（answer 仍关闭）
#   MODE=MIST STRATEGY_THINK=1 \
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#
# 采样模式开关：
#   FULL_DATASET=0（默认）  — 分层采样，每个 subtask JSON 取 VAL_SAMPLES_PER_SUBTASK 条（默认 20）
#   FULL_DATASET=1          — 全量枚举，每道题恰好跑一次（~123k 题，约为默认的 17 倍）
#
# ICR 开关：
#   COMPUTE_ICR=0（默认）   — 不计算 ICR，速度更快
#   COMPUTE_ICR=1           — 计算 ICR（需要 answer model vLLM 支持 logprobs）
#                             所有模式均可使用（few-shot / MIST / mist-inline / habit 等）
#                             ICR 结果随准确率一并打印在评测汇总中
#
# 示例：
#   VAL_SAMPLES_PER_SUBTASK=50 bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   FULL_DATASET=1 bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   COMPUTE_ICR=1 MODE=MIST bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   COMPUTE_ICR=1 MODE=few-shot bash examples/strategy_extraction/scripts/eval_no_verl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- 模式选择 ---
MODE="${MODE:-MIST}"
if [[ "$MODE" != "few-shot" && "$MODE" != "MIST" && "$MODE" != "MIST+few-shot" && "$MODE" != "mist-inline" && "$MODE" != "habit" && "$MODE" != "0-shot" && "$MODE" != "habit-0-shot" && "$MODE" != "meta-test" && "$MODE" != "CoT" && "$MODE" != "Blind-CoT" && "$MODE" != "FS+CoT" && "$MODE" != "MIST-single" ]]; then
    echo "[ERROR] MODE 必须为 'few-shot'、'MIST'、'MIST+few-shot'、'mist-inline'、'habit'、'0-shot'、'habit-0-shot'、'meta-test'、'CoT'、'Blind-CoT'、'FS+CoT' 或 'MIST-single'（当前值: $MODE）"
    exit 1
fi
echo "[INFO] 运行模式: $MODE"

# --- 采样模式 ---
FULL_DATASET="${FULL_DATASET:-0}"
VAL_SAMPLES_PER_SUBTASK="${VAL_SAMPLES_PER_SUBTASK:-20}"

SAMPLING_MODE_ARGS=()
if [[ "${FULL_DATASET}" == "1" ]]; then
    SAMPLING_MODE_ARGS+=(--val-sampling-mode all)
    echo "[INFO] 采样模式：全量枚举（FULL_DATASET=1）"
else
    SAMPLING_MODE_ARGS+=(--val-sampling-mode per_subtask_fixed --val-samples-per-subtask "${VAL_SAMPLES_PER_SUBTASK}")
    echo "[INFO] 采样模式：分层采样，每 subtask ${VAL_SAMPLES_PER_SUBTASK} 条（FULL_DATASET=0）"
fi

# --- fewshot 数量（0-shot / habit-0-shot 模式在各自 block 内覆盖为 0） ---
FEWSHOT_MIN="${FEWSHOT_MIN:-3}"
FEWSHOT_MAX="${FEWSHOT_MAX:-3}"

# --- think 模式控制 ---
# ANSWER_THINK=0（默认）  关闭 answer model think，直接输出 <answer>...</answer>
# ANSWER_THINK=1          开启 answer model think，模型先生成 <think>...</think> 再输出答案
#                         建议在 MIST/MIST+few-shot 模式下设为 1，使 answer model 能按策略步骤推理
# STRATEGY_THINK=0（默认）关闭 strategy model think
# STRATEGY_THINK=1        开启 strategy model think，有助于提升策略归纳质量（截断风险极低）
# 注：CoT 模式语义上强制开启 answer think，忽略 ANSWER_THINK 设置
ANSWER_THINK="${ANSWER_THINK:-0}"
STRATEGY_THINK="${STRATEGY_THINK:-0}"

if [[ "${ANSWER_THINK}" == "1" ]]; then
    ANSWER_THINK_ARGS=()
    echo "[INFO] Answer model think 模式：开启（ANSWER_THINK=1）"
else
    ANSWER_THINK_ARGS=(--answer-no-think)
    echo "[INFO] Answer model think 模式：关闭（ANSWER_THINK=0）"
fi

if [[ "${STRATEGY_THINK}" == "1" ]]; then
    STRATEGY_THINK_ARGS=()
    echo "[INFO] Strategy model think 模式：开启（STRATEGY_THINK=1）"
else
    STRATEGY_THINK_ARGS=(--strategy-no-think)
    echo "[INFO] Strategy model think 模式：关闭（STRATEGY_THINK=0）"
fi

# --- ICR 评测开关 ---
# COMPUTE_ICR=0（默认）  不计算 ICR，评测速度更快
# COMPUTE_ICR=1          计算 ICR（Internal Confidence Reward），
#                        需要 answer model vLLM 服务支持 logprobs（--enable-log-probs）
#                        每次 answer 调用附带 logprobs=True，结果与准确率一并输出
COMPUTE_ICR="${COMPUTE_ICR:-0}"
COMPUTE_ICR_ARGS=()
if [[ "${COMPUTE_ICR}" == "1" ]]; then
    COMPUTE_ICR_ARGS=(--compute-icr)
    echo "[INFO] ICR 计算：开启（COMPUTE_ICR=1，需要 answer server 支持 logprobs）"
else
    echo "[INFO] ICR 计算：关闭（COMPUTE_ICR=0）"
fi

# --- 模式专属参数 ---
MODE_ARGS=()
if [[ "$MODE" == "few-shot" ]]; then
    # few-shot：只有一个答案模型，跳过策略生成，使用 ICL prompt
    MODE_ARGS+=(
        --mode few-shot
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --answer-prompt-version "ICL(few-shot)"
    )
elif [[ "$MODE" == "mist-inline" ]]; then
    # mist-inline：单模型，两次调用（策略提取 + 策略作答），无需策略服务
    MODE_ARGS+=(
        --mode mist-inline
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --inline-strategy-prompt-version "${INLINE_STRATEGY_PROMPT_VERSION:-mist_inline_strategy}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-mist_inline_answer}"
    )
elif [[ "$MODE" == "habit" ]]; then
    # habit：单模型，两次调用（策略提取 + 策略+示例联合作答），无需策略服务
    # 与 mist-inline 的区别：answer call 同时传入 few-shot 示例 + 策略，而非仅策略
    MODE_ARGS+=(
        --mode habit
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --inline-strategy-prompt-version "${INLINE_STRATEGY_PROMPT_VERSION:-mist}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-habit_answer}"
    )
elif [[ "$MODE" == "0-shot" ]]; then
    # 0-shot（婴幼儿基线）：无示例、无策略，直接回答新问题
    # 类比婴幼儿：既无具体示范也无内化策略，仅凭原始能力解题
    FEWSHOT_MIN=0
    FEWSHOT_MAX=0
    MODE_ARGS+=(
        --mode 0-shot
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-zero-shot}"
    )
elif [[ "$MODE" == "habit-0-shot" ]]; then
    # habit-0-shot（成年人基线）：无示例，但模型先从题目自主生成两层策略，再应用策略作答
    # 类比成年人：已通过训练内化元归纳能力，看到新题可自主识别题型并生成解法策略，
    # 无需具体示范——体现"学会学习"的迁移能力
    FEWSHOT_MIN=0
    FEWSHOT_MAX=0
    MODE_ARGS+=(
        --mode habit-0-shot
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --inline-strategy-prompt-version "${INLINE_STRATEGY_PROMPT_VERSION:-habit_0shot_strategy}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-mist_inline_answer}"
    )
elif [[ "$MODE" == "meta-test" ]]; then
    # meta-test（元归纳内化证据测试）：
    #   去除所有策略生成指令，改用完全中性的"请思考这个问题"，
    #   观察经过训练的模型是否仍自发在回答前生成策略（内化证据）。
    #   底层映射为 Python 的 0-shot 模式（无示例、单次 API 调用、直接作答）。
    #   与 habit-0-shot 的核心区别：prompt 不含任何"生成策略"指引，
    #   由此测量：① OOD 准确率相对 habit-0-shot 下降多少；
    #             ② 模型输出中是否自发出现策略块（质性内化证据）。
    #   前置：在 GPU 6,7 启动已训练的 4B 模型（默认端口 8200）：
    #     CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
    #         --model <训练后 4B 模型路径> \
    #         --served-model-name Qwen3-4B \
    #         --tensor-parallel-size 2 \
    #         --port 8200 \
    #         --gpu-memory-utilization 0.90 \
    #         --max-model-len 32768
    FEWSHOT_MIN=0
    FEWSHOT_MAX=0
    MODE_ARGS+=(
        --mode 0-shot
        --answer-model-base-url "${META_TEST_MODEL_BASE_URL:-${ANSWER_MODEL_BASE_URL:-http://localhost:8100/v1}}"
        --answer-model-name "${META_TEST_MODEL_NAME:-${ANSWER_MODEL_NAME:-4b-trained}}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-meta_test_neutral}"
    )
elif [[ "$MODE" == "MIST+few-shot" ]]; then
    # MIST+few-shot：策略生成与 MIST 完全相同（外部 Qwen3-4B 策略模型）；
    # 答案生成时，答案模型同时接收 few-shot 示例 + 策略，综合两者回答新题。
    # 与 habit 模式的区别：策略由外部专用策略模型生成，而非答案模型内联生成。
    # think 控制：通过 ANSWER_THINK / STRATEGY_THINK 环境变量独立设置。
    #   建议 ANSWER_THINK=1，使 answer model 能按策略步骤显式推理后再输出答案。
    MODE_ARGS+=(
        --mode MIST
        --model-path /home/test/test16/chenlu/model/Qwen3-4B
        --strategy-model-base-url "${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}"
        --strategy-model-name "${STRATEGY_MODEL_NAME:-Qwen3-4B}"
        --answer-model-path /home/test/test16/chenlu/model/Qwen3-8B
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        "${STRATEGY_THINK_ARGS[@]}"
        --strategy-prompt-version repetition_controls_2026-04-01
        --strategy-repetition-penalty 1.1
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-mist_fewshot_answer}"
    )
elif [[ "$MODE" == "CoT" ]]; then
    # CoT（标准链式推理，1×，task-aware）：单次调用，think 模式强制开启。
    # 给定 K 个 few-shot 示例后，模型同时看到新题，自由生成 task-aware 推理再输出答案。
    # 这是 Wei et al. (2022) 意义上的经典 CoT——推理内容围绕具体新题展开，是问题特异性的。
    # 可见输出包含 "推理: ..." 段落和 <answer> 块；内部同时开启 <think> 暂存区。
    # 底层映射为 Python few-shot 模式（单次调用+示例，跳过策略模型）。
    # CoT 语义上要求 think，忽略 ANSWER_THINK 设置，强制开启。
    ANSWER_THINK_ARGS=()  # 强制开启 think（CoT 模式专属）
    MODE_ARGS+=(
        --mode few-shot
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-std_cot}"
        --answer-max-tokens "${ANSWER_MAX_TOKENS:-32768}"
    )
elif [[ "$MODE" == "Blind-CoT" ]]; then
    # Blind-CoT（盲式链式推理，2×，free format）：同一答案模型，两次调用，无结构约束。
    # 推理在看到新题之前产生（problem-agnostic），是与 CoT 的核心区别。
    # Call 1（inline strategy）：给定 K 个示例，自由生成推理文本 r（free format，尚未见新题）。
    # Call 2（answer）：基于推理 r 回答新问题（不保留原始示例）。
    # 底层映射为 Python mist-inline 模式（共用 inline strategy 双调用路径）。
    MODE_ARGS+=(
        --mode mist-inline
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --inline-strategy-prompt-version "${INLINE_STRATEGY_PROMPT_VERSION:-cot_reasoning}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-cot_answer}"
    )
elif [[ "$MODE" == "FS+CoT" ]]; then
    # FS+CoT（示例+自由推理）：同一答案模型，两次调用，无结构约束。
    # Call 1（inline strategy）：给定 K 个示例，自由生成推理文本 r（与 Blind-CoT 完全相同）。
    # Call 2（answer）：原始示例 + 推理 r + 新问题 Q → A（示例不被丢弃）。
    # 与 Blind-CoT 的区别：Call 2 同时看到原始示例和推理，而非仅推理。
    # 与 habit 的区别：habit 使用结构化 FIRST_ORDER+SECOND_ORDER 格式；FS+CoT 使用自由文本。
    # 底层映射为 Python habit 模式（共用 habit 双调用+示例路径）。
    MODE_ARGS+=(
        --mode habit
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --inline-strategy-prompt-version "${INLINE_STRATEGY_PROMPT_VERSION:-cot_reasoning}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-fs_cot_answer}"
    )
elif [[ "$MODE" == "MIST-single" ]]; then
    # MIST-single（结构化策略单 pass）：同一答案模型，单次调用。
    # 使用与 MIST 完全相同的结构化策略格式（FIRST_ORDER_PATTERN、FEWSHOT_LEARNING、
    # SECOND_ORDER_STEPS 等章节），但策略生成和答案产出在同一次 forward pass 完成：
    # 先输出 <strategy> 块（完整 MIST 格式），紧接着输出 <answer>。
    # 与 mist-inline 的区别：mist-inline 分两次 API 调用；MIST-single 仅一次。
    # 与 MIST 的区别：无需独立策略模型（Qwen3-4B），仅用答案模型（Qwen3-8B）。
    # 底层映射为 Python few-shot 模式（单次调用+示例，跳过策略模型）。
    MODE_ARGS+=(
        --mode few-shot
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-mist_single_pass}"
    )
else
    # MIST：策略模型（Qwen3-4B）+ 答案模型（Qwen3-8B）
    # think 控制：通过 ANSWER_THINK / STRATEGY_THINK 环境变量独立设置。
    MODE_ARGS+=(
        --mode MIST
        --model-path /home/test/test16/chenlu/model/Qwen3-4B
        --strategy-model-base-url "${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}"
        --strategy-model-name "${STRATEGY_MODEL_NAME:-Qwen3-4B}"
        --answer-model-path /home/test/test16/chenlu/model/Qwen3-8B
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        "${STRATEGY_THINK_ARGS[@]}"
        --strategy-prompt-version repetition_controls_2026-04-01
        --strategy-repetition-penalty 1.1
        --answer-prompt-version v1
    )
fi

python -m examples.strategy_extraction.eval_no_verl \
  --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
  --val-subdirs test-id-subtask test-ood-task test-bbh \
  --fewshot-min "${FEWSHOT_MIN}" \
  --fewshot-max "${FEWSHOT_MAX}" \
  --concurrency "${EVAL_CONCURRENCY:-64}" \
  --llm-seed 42 \
  --num-samples-per-problem "${NUM_SAMPLES_PER_PROBLEM:-3}" \
  "${ANSWER_THINK_ARGS[@]}" \
  --reward-version v3 \
  --answer-request-retries 3 \
  --answer-retry-delay-sec 1.0 \
  "${MODE_ARGS[@]}" \
  "${SAMPLING_MODE_ARGS[@]}" \
  "${COMPUTE_ICR_ARGS[@]}" \
  "$@"

# ============================================================
#  额外测评集：HARDMath2 + Linguini
# ============================================================
#  默认随主评测自动运行，可通过环境变量关闭：
#    EXTRA_BENCHMARKS=0 bash eval_no_verl.sh
#
#  模式映射规则：
#    MIST          → HARDMath2=MIST        Linguini=MIST
#    MIST+few-shot → HARDMath2=MIST        Linguini=MIST  (策略生成与 MIST 相同)
#    few-shot      → HARDMath2=few-shot    Linguini=few-shot
#    mist-inline   → HARDMath2=mist-inline Linguini=mist-inline
#    habit         → HARDMath2=mist-inline Linguini=mist-inline  (habit 本质同 mist-inline)
#    0-shot        → HARDMath2=（跳过）    Linguini=zero-shot
#    CoT           → HARDMath2=few-shot    Linguini=few-shot  (单次调用，额外测评集用标准 ICL prompt)
#    Blind-CoT     → HARDMath2=mist-inline Linguini=mist-inline  (双调用，额外测评集用标准 mist prompt)
#    habit-0-shot / meta-test → 两者均跳过（无对应模式）
#
#  fewshot 示例数：额外测评集使用 EXTRA_FEWSHOT_K（默认 3），
#    与主评测的 FEWSHOT_MIN/MAX 独立，避免 0-shot 等模式将其清零。
#
#  结果目录：
#    HARDMATH2_OUTPUT_DIR  默认 banchmark/HARDMath2/results
#    LINGUINI_OUTPUT_DIR   默认 banchmark/linguini/results/passk
# ============================================================

# Result-file paths — populated inside each extra-benchmark block, consumed in the
# final summary at the bottom of this script.
_hm_result_file=""
_ling_result_file=""

EXTRA_BENCHMARKS="${EXTRA_BENCHMARKS:-1}"

if [[ "${EXTRA_BENCHMARKS}" == "1" ]]; then

    HARDMATH2_DIR="/home/test/test16/chenlu/projects/agent-lightning/banchmark/HARDMath2"
    LINGUINI_DIR="/home/test/test16/chenlu/projects/agent-lightning/banchmark/linguini"
    HARDMATH2_OUTPUT_DIR="${HARDMATH2_OUTPUT_DIR:-${HARDMATH2_DIR}/results}"
    LINGUINI_OUTPUT_DIR="${LINGUINI_OUTPUT_DIR:-${LINGUINI_DIR}/results/passk}"
    EXTRA_FEWSHOT_K="${EXTRA_FEWSHOT_K:-3}"
    EXTRA_NUM_SAMPLES="${EXTRA_NUM_SAMPLES:-${NUM_SAMPLES_PER_PROBLEM:-3}}"
    EXTRA_CONCURRENCY="${EXTRA_CONCURRENCY:-${EVAL_CONCURRENCY:-32}}"

    # ── 模式映射 ──────────────────────────────────────────────────────────────
    _hardmath_mode=""
    _linguini_mode=""
    case "$MODE" in
        MIST)
            _hardmath_mode="MIST"
            _linguini_mode="MIST"
            ;;
        MIST+few-shot)
            _hardmath_mode="MIST"
            _linguini_mode="MIST"
            echo "[INFO] 额外测评集：MIST+few-shot 模式映射为 MIST（额外测评集不支持 few-shot 注入）"
            ;;
        few-shot)
            _hardmath_mode="few-shot"
            _linguini_mode="few-shot"
            ;;
        mist-inline)
            _hardmath_mode="mist-inline"
            _linguini_mode="mist-inline"
            ;;
        habit)
            _hardmath_mode="mist-inline"
            _linguini_mode="mist-inline"
            echo "[INFO] 额外测评集：habit 模式映射为 mist-inline"
            ;;
        0-shot)
            _hardmath_mode=""
            _linguini_mode="zero-shot"
            echo "[INFO] 额外测评集：0-shot → HARDMath2 跳过，Linguini 使用 zero-shot"
            ;;
        habit-0-shot|meta-test)
            echo "[INFO] 额外测评集：MODE=${MODE} 无对应模式，跳过 HARDMath2 和 Linguini"
            ;;
        CoT)
            _hardmath_mode="few-shot"
            _linguini_mode="few-shot"
            echo "[INFO] 额外测评集：CoT 模式映射为 few-shot（单次调用+示例，额外测评集不支持 std_cot prompt，使用标准 ICL prompt）"
            ;;
        Blind-CoT)
            _hardmath_mode="mist-inline"
            _linguini_mode="mist-inline"
            echo "[INFO] 额外测评集：Blind-CoT 模式映射为 mist-inline（额外测评集使用标准 mist 推理 prompt，非 cot_reasoning）"
            ;;
        FS+CoT)
            _hardmath_mode="mist-inline"
            _linguini_mode="mist-inline"
            echo "[INFO] 额外测评集：FS+CoT 模式映射为 mist-inline（额外测评集使用标准 mist 推理 prompt，非 cot_reasoning）"
            ;;
        MIST-single)
            _hardmath_mode="few-shot"
            _linguini_mode="few-shot"
            echo "[INFO] 额外测评集：MIST-single 模式映射为 few-shot（额外测评集使用标准 ICL prompt，非 mist_single_pass）"
            ;;
    esac

    # 提取当前模式下已解析的答案 / 策略模型参数（复用 MODE_ARGS 中的值）
    _answer_url="${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
    _answer_name="${ANSWER_MODEL_NAME:-Qwen3-8B}"
    _strategy_url="${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}"
    _strategy_name="${STRATEGY_MODEL_NAME:-Qwen3-4B}"

    # meta-test 模式下答案模型指向训练后 4B
    if [[ "$MODE" == "meta-test" ]]; then
        _answer_url="${META_TEST_MODEL_BASE_URL:-${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}}"
        _answer_name="${META_TEST_MODEL_NAME:-${ANSWER_MODEL_NAME:-Qwen3-4B}}"
    fi

    # ── HARDMath2 ─────────────────────────────────────────────────────────────
    if [[ -n "${_hardmath_mode}" ]]; then
        echo ""
        echo "[INFO] ════════════════════════════════════════════════════"
        echo "[INFO] 开始额外测评：HARDMath2  (mode=${_hardmath_mode})"
        echo "[INFO] ════════════════════════════════════════════════════"

        _hm_extra=()
        if [[ "${_hardmath_mode}" == "MIST" ]]; then
            _hm_extra+=(
                --strategy-model-name  "${_strategy_name}"
                --strategy-model-base-url "${_strategy_url}"
                "${STRATEGY_THINK_ARGS[@]}"
                --strategy-repetition-penalty 1.1
            )
        fi
        if [[ "${_hardmath_mode}" == "mist-inline" ]]; then
            # 与主评测（test-id/ood/bbh）及 Linguini 保持一致，均使用 mist.toml（重型7区块）
            # eval_hardmath.py 默认值为 mist_inline_strategy，需显式覆盖
            _hm_extra+=(--inline-strategy-prompt-version mist)
        fi

        # answer think：跟随全局 ANSWER_THINK 开关
        _hm_answer_think_args=()
        [[ "${ANSWER_THINK}" != "1" ]] && _hm_answer_think_args=(--answer-no-think)

        python "${HARDMATH2_DIR}/eval_hardmath.py" \
            --mode               "${_hardmath_mode}" \
            --answer-model-name  "${_answer_name}" \
            --answer-model-base-url "${_answer_url}" \
            "${_hm_answer_think_args[@]}" \
            --fewshot-k          "${EXTRA_FEWSHOT_K}" \
            --num-samples        "${EXTRA_NUM_SAMPLES}" \
            --concurrency        "${EXTRA_CONCURRENCY}" \
            --output-dir         "${HARDMATH2_OUTPUT_DIR}" \
            "${_hm_extra[@]}"

        # Capture the most recently written HARDMath2 result file
        _hm_result_file=$(ls -t "${HARDMATH2_OUTPUT_DIR}"/hardmath_*.json 2>/dev/null | head -1 || true)
    fi

    # ── Linguini ──────────────────────────────────────────────────────────────
    if [[ -n "${_linguini_mode}" ]]; then
        echo ""
        echo "[INFO] ════════════════════════════════════════════════════"
        echo "[INFO] 开始额外测评：Linguini  (mode=${_linguini_mode})"
        echo "[INFO] ════════════════════════════════════════════════════"

        # Linguini 使用下划线风格参数（run_linguini_passk.py 接口）
        _ling_extra=()
        if [[ "${_linguini_mode}" == "MIST" ]]; then
            _ling_extra+=(
                --strategy_model    "${_strategy_name}"
                --strategy_api_base "${_strategy_url}"
            )
            # strategy think：跟随全局 STRATEGY_THINK 开关
            [[ "${STRATEGY_THINK}" != "1" ]] && _ling_extra+=(--strategy_no_think)
        fi

        # answer think：跟随全局 ANSWER_THINK 开关（Linguini 用 --no_think 禁用）
        _ling_no_think_arg=()
        [[ "${ANSWER_THINK}" != "1" ]] && _ling_no_think_arg=(--no_think)

        python "${LINGUINI_DIR}/run_linguini_passk.py" \
            --model       "${_answer_name}" \
            --mode        "${_linguini_mode}" \
            --backend     openai_compatible \
            --api_base    "${_answer_url}" \
            "${_ling_no_think_arg[@]}" \
            --shot_num    "${EXTRA_FEWSHOT_K}" \
            --num_samples "${EXTRA_NUM_SAMPLES}" \
            --concurrency "${EXTRA_CONCURRENCY}" \
            --output_dir  "${LINGUINI_OUTPUT_DIR}" \
            --prompt_dir  "/home/test/test16/chenlu/projects/agent-lightning/examples/strategy_extraction/prompt" \
            "${_ling_extra[@]}"

        # Compute the deterministic Linguini result-file path (mirrors run_linguini_passk.py logic)
        _safe_ling_model=$(python3 -c "import re, sys; print(re.sub(r'[^a-zA-Z0-9._-]+', '_', sys.argv[1])[:80])" "${_answer_name}")
        _ling_result_file="${LINGUINI_OUTPUT_DIR}/linguini_passk_${_safe_ling_model}_${_linguini_mode}.json"
    fi

fi

# ============================================================
#  最终汇总：将所有测评结果集中输出一次，便于日志检索
# ============================================================

export _HM_RESULT_FILE="${_hm_result_file}"
export _LING_RESULT_FILE="${_ling_result_file}"
export _EVAL_MODE="${MODE}"

python3 << 'PYEOF'
import os, json

SEP  = "═" * 66
DASH = "─" * 66

def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

print()
print(SEP)
print("  ALL BENCHMARKS — FINAL RESULTS SUMMARY")
print(f"  MODE: {os.environ.get('_EVAL_MODE', '?')}")
print(SEP)

# ── 主评测 (eval_no_verl) ──────────────────────────────────────────────────
print()
print("  ┌─ 主评测 (eval_no_verl) ─────────────────────────────────────┐")
print("  │  ↑ 详细结果已打印在上方 (StrategyGenerationAgent summary)  │")
print("  └────────────────────────────────────────────────────────────┘")

# ── HARDMath2 ──────────────────────────────────────────────────────────────
hm_file = os.environ.get('_HM_RESULT_FILE', '')
print()
print(DASH)
print("  HARDMath2")
print(DASH)
if hm_file and os.path.isfile(hm_file):
    with open(hm_file, encoding='utf-8') as f:
        d = json.load(f)
    print(f"  mode      : {d.get('mode', '?')}")
    print(f"  model     : {d.get('answer_model', '?')}")
    print(f"  n_problems: {d.get('num_problems', '?')}  samples/prob: {d.get('num_samples', '?')}")
    print(f"  acc_hard  : {_fmt(d.get('acc_hard', '?'))}   acc_soft: {_fmt(d.get('acc_soft', '?'))}")
    ph = d.get('pass_at_k_hard', {})
    ps = d.get('pass_at_k_soft', {})
    for k in sorted(ph):
        print(f"  pass@{k}    : hard={_fmt(ph[k])}   soft={_fmt(ps.get(k, '?'))}")
    by_type = d.get('by_type', {})
    if by_type:
        print("  by type:")
        for pt, tv in sorted(by_type.items()):
            pth = tv.get('pass_at_k_hard', {})
            pts = tv.get('pass_at_k_soft', {})
            parts = " ".join(
                f"p@{k}(h={_fmt(pth.get(k,'?'))},s={_fmt(pts.get(k,'?'))})"
                for k in sorted(pth)
            )
            print(f"    {pt:28s}  n={tv.get('n_problems','?'):3}  "
                  f"acc_h={_fmt(tv.get('acc_hard','?'))}  {parts}")
    print(f"  file: {hm_file}")
else:
    print("  (未运行或结果文件未找到)")

# ── Linguini ───────────────────────────────────────────────────────────────
ling_file = os.environ.get('_LING_RESULT_FILE', '')
print()
print(DASH)
print("  Linguini")
print(DASH)
if ling_file and os.path.isfile(ling_file):
    with open(ling_file, encoding='utf-8') as f:
        d = json.load(f)
    meta  = d.get('_meta', {})
    summ  = d.get('_summary', {}).get('overall', {})
    print(f"  mode      : {meta.get('mode', '?')}")
    print(f"  model     : {meta.get('model', '?')}")
    print(f"  n_problems: {summ.get('n_problems', '?')}  samples/prob: {meta.get('num_samples', '?')}")
    for k in (1, 2, 3):
        v = summ.get(f'pass@{k}')
        if v is not None:
            print(f"  pass@{k}(hard): {_fmt(v)}")
    buckets = d.get('_summary', {}).get('buckets', {})
    if buckets:
        print("  by task type:")
        for bt, bv in sorted(buckets.items()):
            pk = bv.get('pass@k', {})
            p1 = pk.get('pass@1', '?')
            p1s = f"{p1:.4f}" if isinstance(p1, float) else str(p1)
            print(f"    {bt:16s}  n={bv.get('n','?'):3}  pass@1(hard)={p1s}")
    print(f"  file: {ling_file}")
else:
    print("  (未运行或结果文件未找到)")

print()
print(SEP)
print("  评测完成。")
print(SEP)
PYEOF

# ============================================================
#  统计显著性检验（可选后处理步骤）
# ============================================================
#  运行完评测后，可用 stat_test.py 对两个条件做 bootstrap 置信区间
#  和配对显著性检验（McNemar / Wilcoxon / 配对 t 检验）。
#  脚本自动识别 Linguini、HARDMath2 和 eval_no_verl 三种结果格式。
#
#  示例——比较两个 Linguini 结果文件：
#    python examples/strategy_extraction/scripts/stat_test.py \
#        --a banchmark/linguini/my_linguini_results/few-shot/Qwen3-8B-trained/linguini_passk_Qwen3-8B-trained_few-shot.json \
#        --b banchmark/linguini/my_linguini_results/MIST+few-shot/Qwen3-8B-trained/linguini_passk_Qwen3-8B-trained_MIST.json \
#        --label-a "few-shot" --label-b "MIST+few-shot" \
#        --subgroup --out results/stat_report.json
#
#  示例——比较两个 HARDMath2 结果文件：
#    python examples/strategy_extraction/scripts/stat_test.py \
#        --a banchmark/HARDMath2/results/hardmath_few-shot.json \
#        --b banchmark/HARDMath2/results/hardmath_MIST.json \
#        --label-a "few-shot" --label-b "MIST" --subgroup
#
#  示例——比较训练前后的 eval_no_verl rollout 目录：
#    python examples/strategy_extraction/scripts/stat_test.py \
#        --a checkpoints_eval_no_verl/pre/ \
#        --b checkpoints_eval_no_verl/post_step250/ \
#        --label-a "4B base" --label-b "4B MIST step-250"
#
#  参数速查：
#    --n-bootstrap N   bootstrap 次数（默认 10000）
#    --ci FLOAT        置信水平，单位 %（默认 95）
#    --seed INT        随机种子（默认 42）
#    --subgroup        输出按任务类型分组的子集检验
#    --out PATH        将完整报告写入 JSON 文件
# ============================================================

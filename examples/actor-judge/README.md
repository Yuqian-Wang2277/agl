# Actor-Judge Co-Evolution — Phase II RL Training

> **Self-Evolving Agent via Strategy-Outcome Co-Evolution**  
> Tech stack: PyTorch · accelerate FSDP · vLLM 0.10.2 · Ray 2.53.0 (all in `agl` conda env)

---

## 目录

1. [实验动机与核心思想](#1-实验动机与核心思想)
2. [整体架构](#2-整体架构)
3. [两阶段 Rollout 详解](#3-两阶段-rollout-详解)
4. [协同进化飞轮](#4-协同进化飞轮)
5. [文件结构与职责](#5-文件结构与职责)
6. [Prompt 迭代指南](#6-prompt-迭代指南)
7. [快速启动](#7-快速启动)
8. [消融实验配置](#8-消融实验配置)
9. [监控指标](#9-监控指标)
10. [关键工程细节](#10-关键工程细节)

---

## 1. 实验动机与核心思想

### 核心假设（解耦假设）

$$P(A_{\text{correct}} \mid S_{\text{valid}}) \approx 1$$

如果策略 $S$ 是有效的，Actor 执行后得到正确答案的概率接近 1。  
因此，答案 $A$ 的成败主要归因于策略 $S$ 的质量，将高维生成评估压缩为**策略评分问题**。

### 两阶段设计

```
Phase I  (已跳过): SFT 冷启动
         → 使用 strategy_extraction 的 global_step_400 checkpoint 替代

Phase II (本模块):  协同进化 RL
         → Actor 生成策略 + 答案
         → Judge 学习评估策略质量
         → Judge 的稠密奖励驱动 Actor 进化
```

### 为什么需要 Judge？

纯 RL（只用答案正误作为奖励）面临两个问题：
1. **奖励稀疏**：只有二元信号（0/1），无法区分"差一点"和"完全错误"的策略。
2. **信用分配困难**：两阶段生成中，答案错误可能源于策略差，也可能源于策略好但答案执行失败。

Judge 通过 Bradley-Terry 学习从 *成功/失败经验* 中自动校准策略质量分，为 Actor 提供**稠密、连续的奖励信号**。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Phase II 协同进化主循环                           │
│                                                                     │
│   ┌──────────┐  K条策略+答案   ┌──────────┐  y∈{-1,0,1}           │
│   │  Actor   │─────────────▶│   Env    │──────────────┐           │
│   │ (Qwen3)  │◀─────────────│ (F1+Ext) │              ▼           │
│   └──────────┘  GRPO更新     └──────────┘        ┌──────────┐      │
│         ▲                                        │   UCB    │      │
│         │ 稠密奖励 σ(V)                           │  Buffer  │      │
│   ┌──────────┐  BT Loss更新   ┌──────────┐        └────┬─────┘      │
│   │  Judge   │◀──────────────│ Pairwise │◀────────────┘            │
│   │ (Scalar  │               │ Sampling │  (win, lose) pairs       │
│   │  Head)   │               └──────────┘                          │
│   └──────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**三个模型**（均来自同一 Actor SFT checkpoint 初始化）：

| 模型 | 结构 | 训练状态 | 用途 |
|------|------|----------|------|
| Actor | Qwen3-4B 全参数 | 训练（GRPO） | 生成策略 + 答案 |
| Reference | Qwen3-4B 全参数 | 完全冻结 | KL 惩罚基准 |
| Judge | Qwen3-4B backbone + Scalar Head | 训练（BT Loss） | 策略质量评分 |

---

## 3. 两阶段 Rollout 详解

```
输入: batch of B 道题目 (domain, fewshot_examples, question, answer_gold)
                              ↓
         ┌─────────────────────────────────────────┐
         │ Stage 1: 策略生成（K 次并行采样）          │
         │                                         │
         │  prompt: few-shot context               │
         │  model:  vLLM(Actor), temperature=0.7   │
         │  stop:   "</strategy>"                  │
         │  output: K 条策略 S_1…S_K               │
         └─────────────────────────────────────────┘
                              ↓
                    Format_Check: S 必须含 <strategy>…</strategy>
                    ✗ → outcome=-1 (format error，不进入 Judge 训练)
                              ↓
         ┌─────────────────────────────────────────┐
         │ Stage 2: 答案生成（每条有效策略各一次）    │
         │                                         │
         │  prompt: strategy + question            │
         │  model:  vLLM(Actor), temperature=0     │
         │  stop:   "</answer>"                    │
         │  output: 答案 A_k                        │
         └─────────────────────────────────────────┘
                              ↓
                    Env.evaluate(S_k, A_k, answer_gold)
                    → y_k ∈ {0, 1}  (exact/numeric/F1 cascade)
                              ↓
              Experience(traj_id=UUID, context, Q, S_k, y_k) → Buffer
```

**同域 vs 跨域**：由 `cross_domain_ratio` 控制。跨域时 few-shot context 来自 domain D₁，新问题 Q' 来自 domain D₂，验证策略的跨域泛化能力。

---

## 4. 协同进化飞轮

```
Step 1: Rollout（Rank 0 独占 vLLM）
        Actor 生成 B×K 条 (S, y) → 写入 UCB Buffer

Step 2: Judge 更新（off-policy，使用历史 buffer）
        UCB 加权采样 (win, lose) 对
        Loss = BT_Loss(logit_win - logit_lose) + L2_penalty
        写回 v_pred → O(1) traj_id 索引

Step 3+4: Actor 更新（on-policy，仅使用当前 experiences）
        log_prob_old  = Actor.no_grad(current batch)    [P1: 不从 vLLM 取]
        log_prob_ref  = RefModel.no_grad(current batch)
        log_prob_actor = Actor.forward(current batch)   [有梯度]
        reward = (1-α)·y + α·σ(Judge) + len_penalty - β·KL
        Z-Score 组内归一化 → GRPO Clip → backward
```

**GRPO 奖励公式**：

$$r_k = \underbrace{(1-\alpha) \cdot y_k}_{\text{稀疏奖励}} + \underbrace{\alpha \cdot \sigma(V_J(S_k))}_{\text{稠密奖励}} + \underbrace{\text{len\_penalty}}_{\text{长度惩罚}} - \underbrace{\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})}_{\text{KL 惩罚}}$$

**UCB 优先级分数**：

$$\text{Score}(i) = \lambda_{\text{err}} \cdot |y_i - \hat{v}_i| + \lambda_{\text{exp}} \cdot \sqrt{\frac{\ln T}{n_i + 1}}$$

---

## 5. 文件结构与职责

```
actor-judge/
├── README.md                    ← 本文档
├── IMPLEMENTATION.md            ← 详细实现方案（含全部 Review 修正记录）
├── config.py                    ← ActorJudgeConfig dataclass（所有超参+消融开关）
├── data_loader.py               ← 数据加载：RolloutSample, ActorJudgeDataset, S_gold 匹配
├── prompts.py                   ← Prompt 构建 API（从 prompt/ TOML 加载）
├── env.py                       ← 答案验证：Format_Check + exact/numeric/F1 + 长度惩罚
├── buffer.py                    ← UCBBuffer：per-Q 容量、O(1) v_pred 回写、纯 Python 标量
├── judge_model.py               ← JudgeModel: Qwen3 backbone + Scalar Head + 热启动工具
├── rollout_engine.py            ← 两阶段 vLLM Rollout（@ray.remote(num_gpus=0)）
├── actor_trainer.py             ← GRPO：三段式 action_mask、P1 log_prob_old、零方差短路
├── judge_trainer.py             ← Bradley-Terry Loss + L2 logit 正则 + UCB 采样
├── train.py                     ← Phase II 主循环（Rank 0 vLLM → broadcast → FSDP）
├── convert_checkpoint.py        ← VERL FSDP 分片 → HuggingFace safetensors
│
├── prompt/                      ← TOML-based prompt 包（可独立迭代）
│   ├── __init__.py              ← load_prompt(), list_versions(), format_examples()
│   ├── strategy_generation/
│   │   └── fewshot_extract_v1.toml    ← Actor Stage-1: few-shot → strategy
│   ├── answer_generation/
│   │   └── strategy_guided_v1.toml   ← Actor Stage-2: strategy + Q → answer
│   └── judge_evaluation/
│       └── quality_scalar_v1.toml    ← Judge: strategy quality scoring
│
└── scripts/
    ├── train.sh                 ← 多卡 FSDP 训练启动
    ├── debug.sh                 ← 单卡 debug（无 vLLM/Ray，快速验证流程）
    ├── validate.sh              ← 单独跑 Pass@1 验证
    └── convert_checkpoint.sh   ← VERL checkpoint 转换
```

### 模块依赖关系

```
config.py
    └── 被所有模块引用

data_loader.py
    └── 依赖: config

prompt/
    └── 被 prompts.py 引用

prompts.py
    └── 依赖: prompt/

env.py
    └── 依赖: prompts (sentinel tokens)

buffer.py
    └── 纯 Python，无外部依赖

judge_model.py
    └── 依赖: transformers

rollout_engine.py
    └── 依赖: ray, vllm, prompts, env, buffer, data_loader

actor_trainer.py
    └── 依赖: torch, buffer, prompts

judge_trainer.py
    └── 依赖: torch, buffer, prompts

train.py
    └── 依赖: 所有上述模块
```

---

## 6. Prompt 迭代指南

Prompt 文本存储在 `prompt/<category>/<purpose>_v<N>.toml` 中，**修改 prompt 不需要改任何 Python 代码**。

文件命名规范：`<用途描述>_v<版本号>.toml`，例如 `fewshot_extract_v1.toml`，便于在同一 category 下区分不同实验 prompt。

### 添加新版本 prompt

```bash
# 以策略生成 prompt 为例
cp prompt/strategy_generation/fewshot_extract_v1.toml \
   prompt/strategy_generation/fewshot_extract_v2.toml
# 编辑 v2.toml 中的 system / user 字段
```

### 在训练中使用新版本

方法 1：修改 `config.py` 增加版本字段并在 `build_*_prompt()` 调用时传入：
```python
# 在 rollout_engine.py 中
msgs = build_strategy_prompt(examples, version=cfg.strategy_prompt_version)
```

方法 2：直接在代码中临时指定：
```python
msgs = build_strategy_prompt(examples, version="fewshot_extract_v2")
```

### TOML 文件与占位符

| 类别 | 当前文件 | 占位符 |
|------|---------|--------|
| `strategy_generation` | `fewshot_extract_v1.toml`  | `{examples_text}` |
| `answer_generation`   | `strategy_guided_v1.toml`  | `{strategy}`, `{problem}` |
| `judge_evaluation`    | `quality_scalar_v1.toml`   | `{examples_text}`, `{question}`, `{strategy}` |

---

## 7. 快速启动

### 前置条件

```bash
conda activate agl
# agl 环境已包含: torch 2.8, vllm 0.10.2, accelerate 1.12, ray 2.53, transformers 4.57
# 如需 DeepSpeed ZeRO-3（可选）:
pip install deepspeed
```

### Step 0：转换 SFT checkpoint（如果从 VERL 分片开始）

```bash
bash scripts/convert_checkpoint.sh \
    --checkpoint_dir /path/to/global_step_400/actor \
    --output_dir     /path/to/global_step_400/actor_hf
```

### Step 1：Debug 验证（推荐先跑，无需 GPU）

```bash
bash scripts/debug.sh
```

### Step 2：启动训练

```bash
# 从 SFT checkpoint 开始（推荐）
bash scripts/train.sh \
    --sft_checkpoint /path/to/actor_hf \
    --epochs 5 \
    --K 8

# 从 base model 开始（回退方案）
bash scripts/train.sh \
    --epochs 5 --K 8
```

### Dry run（10 题 × 2 epoch 冒烟）

缩小数据量时容易踩到边界：Judge 经验池达不到 `min_buffer_size` 导致 ODVA 永远不跑、验证集仍按 500/拆跑满 vLLM、WandB 污染、完整 Judge warmup 无意义、崩溃后 Ray/vLLM 占显存。

- **一键脚本**（默认 `WANDB_MODE=offline`、10 训练题、2 epoch、`min_buffer_size=8`、每 split 验证 10 条、复用 `judge_warmup_ckpt/2026-03-20/judge_model.pt`）：

```bash
export SFT_CHECKPOINT=/path/to/actor_hf
bash scripts/dry_run_phase2.sh
# 或: bash scripts/dry_run_phase2.sh /path/to/actor_hf
```

- **清道夫**（主进程崩后释放显存）：`bash scripts/cleanup_ray_vllm.sh`

手动等价参数见 `train.py`：`--min_buffer_size`、`--dry_run_val_size`（会覆盖 `--val_num_samples`）。

**验证阶段 Judge 与训练对齐**：val 明细里会写入与 rollout 相同的 `context_text`（`Q: …  A: …` 拼接 few-shot），Judge 打分使用 `build_judge_prompt(..., context_text_raw=context_text)`，与 ODVA / dense reward 一致（避免与 `format_examples` 版式混用带来的分布偏移）。

**大批量 val 明细**：`--val_item_storage jsonl` 将逐行写入 `eval_*_items.jsonl`（主 JSON 里可不含 `items`，见 `val_items_jsonl` 字段）；`--val_item_storage both` 双写。`--val_log_items_wandb_table` 将明细记为可排序的 `wandb.Table`（便于按 `judge_prob` 筛查）。

### Step 3：验证

```bash
# 验证单个 split
bash scripts/validate.sh \
    --checkpoint ./checkpoints_actor_judge/epoch_004 \
    --split test-bbh

# 验证全部三个 split
for split in test-id-subtask test-ood-task test-bbh; do
    bash scripts/validate.sh \
        --checkpoint ./checkpoints_actor_judge/epoch_004 \
        --split "$split"
done
```

---

## 8. 消融实验配置

通过 `train.sh` 参数控制，每组实验改动最小。

| 实验 | 目标 | 启动命令 |
|------|------|----------|
| **完整系统**（基线） | Actor+Judge 协同进化 | `bash scripts/train.sh --sft_checkpoint PATH` |
| **实验 A**: 冻结 Judge | 验证协同进化是否必要 | `bash scripts/train.sh ... --freeze_judge` |
| **实验 B**: 纯稀疏奖励 | 验证稠密奖励的价值 | `bash scripts/train.sh ... --dense_reward_alpha 0 --freeze_judge` |
| **实验 C**: 无 SFT 冷启动 | 验证 Phase I 的必要性 | `bash scripts/train.sh` (不传 --sft_checkpoint) |
| **实验 D**: 无 KL 惩罚 | 验证灾难性遗忘防护 | `bash scripts/train.sh ... --extra "--kl_penalty_beta 0"` |
| **实验 E**: FIFO Buffer | 验证 UCB 反思机制 | `bash scripts/train.sh ... --disable_ucb_replay` |

> ⚠️ **实验 B 注意**：`dense_reward_alpha=0` 时必须同时传 `--freeze_judge`，否则 Judge 白白训练而其输出从未被使用，对比不公平。

---

## 9. 监控指标

所有指标通过 `wandb` 实时记录。

### 结果指标

| 指标 | 含义 | 理想趋势 |
|------|------|----------|
| `val/test-id-subtask/pass@1` | 同域验证集准确率 | 上升 |
| `val/test-ood-task/pass@1` | 跨域验证集准确率 | 上升（验证策略泛化） |
| `val/test-bbh/pass@1` | BBH benchmark 准确率 | 上升 |

### 过程指标

| 指标 | 含义 | 说明 |
|------|------|------|
| `train/pass_rate` | 当前 batch 答对比例 | 训练中的即时指标 |
| `train/actor_loss` | GRPO loss | 应稳定下降 |
| `train/judge_loss` | BT loss | 应快速下降 |
| `train/buffer_size` | Buffer 总经验数 | 应持续增长 |
| `train/avg_strategy_len` | 平均策略长度（token 数） | **关键防作弊指标** |

### Length Hacking 预警

若在 Wandb 中观察到：
- `pass@1` 连续多个 epoch **不涨**
- `avg_strategy_len` 连续 3 个 epoch **涨幅 >10%**

说明 Judge 已被长度欺骗，应在 `config.py` 中启用：
```python
length_penalty_coeff = 0.01   # 超过 500 tokens 线性惩罚
```

---

## 10. 关键工程细节

以下是跨越四轮 Review 积累的最重要工程经验，写代码时务必注意。

### 分布式并发架构

```
accelerate launch (8 进程)
       ↓
Rank 0 独占: ray.remote(num_gpus=0) → VLLMActor → 生成 experiences
       ↓
Rank 0: ray.kill(vllm_actor) + torch.cuda.empty_cache()
       ↓
全部 Rank: wait_for_everyone() → broadcast_object_list_from_rank0()
       ↓
全部 Rank: FSDP 训练（actor + judge）
```

**为什么 `num_gpus=0`**：PyTorch FSDP 已物理占用所有 GPU，Ray 调度器若看到 `num_gpus=8` 会永远等待空闲卡（Pending 死锁）。设为 0 让 vLLM 的 `tensor_parallel_size` 自行接管硬件。

### log_prob_old 的计算方式

**不从 vLLM 获取 logprobs**。vLLM 返回的 token 序列与 PyTorch tokenizer 的 `input_ids` 因 Prefix Space 机制可能错位（差 1 token 就全错）。正确做法：vLLM 只返回文本，在 `actor_trainer.train_step()` 开头用当前 Actor（权重尚未更新）做一次 `torch.no_grad()` forward。

### Experience 的纯 Python 约束

`buffer.py` 中 `Experience` 的所有字段必须是 `str/int/float/List[int]` 等纯 Python 类型，**绝对禁止 GPU Tensor**。`broadcast_object_list` 底层用 pickle，混入 GPU Tensor 会触发 NCCL Hang（永久卡死，无任何报错）。

### ref_model 的分发

```python
# 正确（P5）
actor, ref_model, judge = (
    accelerator.prepare(actor),
    accelerator.prepare(ref_model),   # 必须！
    accelerator.prepare(judge),
)
```

若忘记 prepare ref_model，每个 Rank 各加载一个完整 4B 模型 → 8×8GB = 64GB 冗余显存。

### `<|judge|>` token 热启动

```python
tokenizer.add_tokens(["<|judge|>"])
judge.transformer.resize_token_embeddings(len(tokenizer))
# 用 <|im_end|> 的 embedding 初始化，避免前几 epoch 的噪声
warmstart_judge_token_embedding(judge, tokenizer, source_token="<|im_end|>")
```

`resize_token_embeddings` 后新 token 是随机向量，经过 30+ 层后是纯噪声，BT Loss 前几 epoch 会飙升。

### action_mask 三段式

```
[Prompt tokens = 0 | Strategy tokens = 1 | Padding tokens = 0]
```

构建方式：先用 `(input_ids != pad_token_id).int()` 排除 PAD，再将 Prompt 段强行置 0。**不能漏掉 Padding 部分**，否则 PAD token 的 log_prob 会污染 KL 和 Ratio。

### Z-Score 零方差保护

```python
mask_valid_std = (std_k > 1e-4).float()
advantage = ((reward_bk - mean_k) / (std_k + 1e-8)) * mask_valid_std
```

当 K 个样本奖励完全相同时（低温 / Greedy 采样易发生），跳过梯度更新比产生 ±150 的极端噪声要好得多。

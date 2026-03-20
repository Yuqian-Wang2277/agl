# Actor-Judge 协同进化实现方案

> 实验全称：Self-Evolving Agent via Strategy-Outcome Co-Evolution  
> 技术栈：PyTorch + accelerate FSDP + vLLM 0.10.2 + Ray 2.53.0（均在 `agl` 环境中）

---

## 目录

1. [实验方案总览](#1-实验方案总览)
2. [训练环境](#2-训练环境)
3. [起始模型处理](#3-起始模型处理)
4. [目录结构与文件职责](#4-目录结构与文件职责)
5. [核心模块实现细节](#5-核心模块实现细节)
   - 5.1 data_loader.py
   - 5.2 prompts.py
   - 5.3 env.py
   - 5.4 buffer.py
   - 5.5 judge_model.py
   - 5.6 rollout_engine.py
   - 5.7 actor_trainer.py
   - 5.8 judge_trainer.py
   - 5.9 train.py
   - 5.10 config.py
6. [关键技术问题与修正](#6-关键技术问题与修正)
7. [消融实验设计](#7-消融实验设计)
8. [评估指标体系](#8-评估指标体系)
9. [快速启动命令](#9-快速启动命令)

---

## 1. 实验方案总览

### 核心假设（解耦假设）

$$P(A_{correct} | S_{valid}) \approx 1$$

即：如果策略 S 是有效的，Actor 执行后得到正确答案的概率接近 1。在此假设下，答案 A 的成败主要归因于策略 S 的质量，从而将高维的生成评估压缩为策略评分问题。

### 两阶段设计

```
Phase I (已跳过): SFT 冷启动 — 使用 strategy_extraction checkpoint（global_step_400）代替
Phase II:         协同进化 RL — 本文档实现目标
```

### Actor 的两阶段 Rollout

```
输入                          Actor 处理                     输出
--------------------         ------------------             ----------
C = {few-shot examples}  →  策略生成 P(S|C)           →   S（策略）
S + Q'（新问题）         →  答案生成 P(A|S,C,Q')      →   A（答案）
                                                            ↓
                                                       Env(A, A_gold) = y ∈ {0,1}
```

- 同域（In-Domain）：few-shot C 与新问题 Q' 来自同一 domain
- 跨域（Cross-Domain）：few-shot C 来自 domain D₁，Q' 来自 domain D₂，验证策略泛化性

### 协同进化飞轮

```
Actor 探索 → 产生 (S, y) 数据 → 填充 UCB Buffer
                                         ↓
                              Judge 从 Buffer 学习评价策略质量
                                         ↓
                              Judge 给 Actor 生成稠密奖励信号
                                         ↓
                              Actor 向 Judge 高分区域偏移 → 更好的策略
```

---

## 2. 训练环境

### 结论：**使用 `agl` 虚拟环境，不需要重新搭建**

#### agl 环境已有的关键依赖

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | 2.8.0+cu128 | 训练基础（CUDA 12.8） |
| `vllm` | 0.10.2 | 两阶段 rollout 推理 |
| `accelerate` | 1.12.0 | FSDP 分布式训练 + checkpoint 转换 |
| `transformers` | 4.57.5 | 模型加载 / tokenizer |
| `ray` | 2.53.0 | 多进程编排（Actor ↔ vLLM Worker） |
| `peft` | 0.18.1 | LoRA（可选，节省显存） |
| `flash_attn` | 2.8.3 | 高效 attention |
| `openai` | 2.15.0 | 通过 vLLM 的 OpenAI 接口生成 |

#### 唯一需要额外安装的包

```bash
conda activate agl

# 安装 deepspeed（用于 ZeRO-3 Actor/Judge 分布式训练）
pip install deepspeed

# 验证全链路
python -c "import torch, vllm, accelerate, ray, deepspeed; print('All OK')"
```

**注意**：deepspeed 安装时会编译 CUDA 算子（需要 NVCC）。如果编译失败，可回退到 `accelerate` 内置 FSDP（不依赖 deepspeed），功能基本等价，仅 ZeRO-3 offload 能力略弱。

---

## 3. 起始模型处理

### 现有 Checkpoint 状态

```
checkpoints/global_step_400/actor/
├── huggingface/               ← 只有 tokenizer 文件（无模型权重）
├── model_world_size_8_rank_{0..7}.pt   ← VERL FSDP 8 rank 分片权重
├── optim_world_size_8_rank_{0..7}.pt   ← 优化器状态（训练用，推理不需要）
├── extra_state_world_size_8_rank_{0..7}.pt
└── fsdp_config.json
```

### 转换方案：`convert_checkpoint.py`

使用 `accelerate` 的 FSDP Consolidation API，将分片合并为 HuggingFace safetensors：

```python
# convert_checkpoint.py 核心逻辑
from accelerate.utils import merge_fsdp_weights

merge_fsdp_weights(
    checkpoint_dir="checkpoints/global_step_400/actor",
    output_path="checkpoints/global_step_400/actor_hf",
    safe_serialization=True,
)
# 同时复制 tokenizer 文件到 actor_hf/
```

转换后的 `actor_hf/` 目录包含标准 HuggingFace safetensors，可以直接 `from_pretrained` 加载。

### 三个模型的初始化

```python
# Actor（全参数训练）
actor = AutoModelForCausalLM.from_pretrained("actor_hf")

# Reference Model（完全冻结，仅 inference，用于 KL 惩罚）
ref_model = AutoModelForCausalLM.from_pretrained("actor_hf")
ref_model.requires_grad_(False)

# Judge（同 Actor 权重，替换最后一层 lm_head）
judge = JudgeModel.from_actor_checkpoint("actor_hf")
# JudgeModel 内部：删除 lm_head，增加 nn.Linear(hidden_size, 1)
```

**回退方案**：若 `accelerate.utils.merge_fsdp_weights` 接口不可用（版本差异），从 `/home/test/test16/chenlu/model/Qwen3-4B` base model 开始（缺少 strategy following SFT 能力，效果预期略降，但框架逻辑不变）。

---

## 4. 目录结构与文件职责

```
examples/actor-judge/
├── __init__.py
├── IMPLEMENTATION.md          ← 本文档
├── config.py                  ← ActorJudgeConfig dataclass（超参 + 路径 + 消融开关）
├── data_loader.py             ← 从 LLMReflection 数据加载 (domain, fewshot, Q, A_gold)，匹配 S_gold
├── prompts.py                 ← Actor 两阶段 prompt + Judge 输入 prompt 模板
├── env.py                     ← 答案验证（Format_Check + exact/numeric/F1）
├── buffer.py                  ← UCB Reflective Experience Replay Buffer
├── judge_model.py             ← JudgeModel：Qwen3 backbone + Scalar Head
├── rollout_engine.py          ← vLLM 两阶段 rollout（同域/跨域，格式兜底）
├── actor_trainer.py           ← GRPO Actor 更新（三路 log-prob，[B,K] Z-Score）
├── judge_trainer.py           ← ODVA/Bradley-Terry Judge 更新（UCB 采样，traj_id 回写）
├── train.py                   ← Phase II 主训练入口（Step 0-4）
└── convert_checkpoint.py      ← VERL FSDP 分片 → HuggingFace safetensors（via accelerate）
```

---

## 5. 核心模块实现细节

### 5.1 `data_loader.py`

**数据来源**：
- 问题数据：`/home/test/test16/chenlu/projects/LLMReflection/data/train_20k/`
- 策略数据（仅用于 Judge Warmup）：`/home/test/test16/chenlu/projects/fs/strategy/`

**数据格式**（LLMReflection JSON）：
```json
{
    "task": "simple_arithmetic_json",
    "subtask": "one_digit",
    "examples": [
        {"input": "1 + 4 = ", "target": ["5"]},
        {"input": "4 + 9 = ", "target": ["13"]}
    ]
}
```

**数据集样本结构**：
```python
@dataclass
class RolloutSample:
    domain: str                        # 任务类型名
    fewshot_examples: List[dict]       # 用于策略生成的 k 个示例（k=3-5）
    question: str                      # 需要回答的新问题 Q'
    answer_gold: str                   # 标准答案
    s_gold: Optional[str] = None       # 来自 fs/strategy 的黄金策略（仅 warmup 用）
```

**S_gold 匹配逻辑**：遍历 `fs/strategy/train_all/gain_pos/` 下的目录，按 domain 名（标准化后）进行字符串匹配，找到对应的 `strategies_out/size_*/001.cand1.strategy.txt`。

**同域 vs 跨域采样**：由 `cross_domain_ratio` 控制比例，跨域时随机选取源 domain（提供 few-shot）和目标 domain（提供 Q'）。

---

### 5.2 `prompts.py`

**Actor 阶段 1 —— 策略生成**：
```
[系统]: 你是一个善于从例子中归纳解题策略的助手。
[用户]:
以下是一些例题和解答：

例题 1：{input_1}
答案：{answer_1}

例题 2：{input_2}
答案：{answer_2}

...（k 个示例）

请分析上述例题，归纳出通用的解题策略。策略需具有可迁移性，能指导解决同类型的新题目。
请将策略包裹在 <strategy>...</strategy> 标签内输出。
```

**Actor 阶段 2 —— 答案生成**：
```
[系统]: 你是一个严格遵循解题策略的助手。
[用户]:
解题策略：
{strategy}

题目：{question}

请严格按照上述策略解题，将最终答案包裹在 <answer>...</answer> 标签内输出。
```

**Judge 输入格式**（含 `<|judge|>` 锚点 token）：
```
[INST]
任务背景示例：
{few-shot examples}

待解题目：{question}

候选策略：{strategy}

请评估：上述策略是否能有效指导解出该题目？
[/INST]<|judge|>
```

> `<|judge|>` 需在 tokenizer 中注册为特殊 token，Scalar Head 基于该 token 的 hidden state 输出打分 logit。

---

### 5.3 `env.py`

答案验证环境，在 strategy_extraction 的 reward/v1.py 基础上新增 Format_Check：

```python
def evaluate(strategy_text: str, answer_text: str, answer_gold: str) -> int:
    """
    返回值：
      -1: 格式完全破坏（策略无 </strategy> 结束符，或答案无 <answer> 标签）
       0: 格式正确但答案错误
       1: 格式正确且答案正确
    """
    # Format_Check（防止"毒数据"污染 Judge）
    if not _format_valid(strategy_text, answer_text):
        return -1

    answer = _extract_answer(answer_text)
    if answer is None:
        return 0

    return _check_correctness(answer, answer_gold)  # exact/numeric/F1
```

y = -1 的样本存入 buffer 时标记，**不参与 Judge pairwise 采样**（防止格式错误的"毒数据"进入 Judge 训练）。

---

### 5.4 `buffer.py` — UCB Reflective Experience Replay Buffer

**数据结构**：

```python
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math, random

@dataclass
class Experience:
    traj_id: str           # UUID，用于 O(1) 写回 v_pred
    context_text: str      # few-shot examples 原始文本（纯文本，不存 tensor）
    question: str          # 新问题 Q'
    strategy: str          # 生成的策略 S
    outcome: int           # y ∈ {-1, 0, 1}，-1 为格式错误硬惩罚
    n_sampled: int = 0     # 被 Judge 采样次数
    v_pred: float = 0.5    # Judge 最新预测分（初始 0.5）
    timestamp: int = 0     # 入库时的 global step

class UCBBuffer:
    def __init__(self, max_size: int = 50000,
                 per_q_max: int = 50,        # 单 Q 容量上限（防止极难题无限堆积）
                 lambda_err: float = 1.0,
                 lambda_exp: float = 1.0):
        # buffer[q_hash][traj_id] = Experience
        self._data: Dict[str, Dict[str, Experience]] = {}
        self.global_sample_steps: int = 0   # T：全局采样次数
        self.max_size = max_size
        self.per_q_max = per_q_max          # 每个 q_hash 最多保留的轨迹数
        self.lambda_err = lambda_err
        self.lambda_exp = lambda_exp

    def add(self, exp: Experience):
        q_hash = _hash_question(exp.question)
        if q_hash not in self._data:
            self._data[q_hash] = {}

        self._data[q_hash][exp.traj_id] = exp

        # per-Q 容量控制：超出 per_q_max 时，淘汰该 q 下 UCB 分数最低的轨迹
        # 防止"永远全错"的极难题无限堆积占满 Buffer
        if len(self._data[q_hash]) > self.per_q_max:
            T = max(self.global_sample_steps, 1)
            def ucb_score(e: Experience) -> float:
                return (self.lambda_err * abs(e.outcome - e.v_pred)
                        + self.lambda_exp * math.sqrt(math.log(T) / (e.n_sampled + 1)))
            worst_id = min(self._data[q_hash], key=lambda tid: ucb_score(self._data[q_hash][tid]))
            del self._data[q_hash][worst_id]
```

**UCB 优先级分数**：

$$\text{Score}(i) = \lambda_{err} \cdot |y_i - \hat{v}_i| + \lambda_{exp} \cdot \sqrt{\frac{\ln T}{n_i + 1}}$$

**Pairwise 采样**：
1. 遍历 q_hash，过滤掉 y=-1 的样本，收集可构成 (win, lose) 对的 q_hash
2. 对每个有效 q_hash，计算所有轨迹的 UCB Score
3. 从 win 组（y=1）和 lose 组（y=0）中各按 UCB Score 加权随机抽取一条
4. `global_sample_steps += 1`（T 递增）
5. 返回 `(q_hash, traj_id_win, traj_id_lose, experience_win, experience_lose)`

**惰性 v_pred 更新**（O(1)）：

```python
def update_v_pred(self, q_hash: str, traj_id: str, new_v: float):
    exp = self._data[q_hash][traj_id]
    exp.v_pred = new_v
    exp.n_sampled += 1
```

**单极性兜底**：若某 q_hash 下全 y=1 或全 y=0，记录日志并跳过该 q_hash 的 pairwise 采样。若此类 q_hash 占比 > 30%，发出 WARNING 提示调高采样温度。

> **单极性与 per-Q 上限的协同**：极难题（永远 y=0）的每次新轨迹都会触发 per-Q 淘汰逻辑，自动丢弃最无信息量的旧轨迹，保证 Buffer 整体信息密度不被单一问题劣化。

---

### 5.5 `judge_model.py`

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class JudgeModel(nn.Module):
    """Qwen3 backbone + Scalar Head，用于评估策略质量。"""

    JUDGE_TOKEN = "<|judge|>"

    def __init__(self, config):
        super().__init__()
        self.transformer = AutoModel.from_config(config)
        self.scalar_head = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_actor_checkpoint(cls, checkpoint_path: str) -> "JudgeModel":
        """从 Actor 的 HuggingFace checkpoint 初始化 Judge。"""
        config = AutoConfig.from_pretrained(checkpoint_path)
        model = cls(config)
        # 加载 transformer 权重（跳过 lm_head）
        state_dict = ...  # AutoModelForCausalLM.from_pretrained 加载后提取 model.model 部分
        model.transformer.load_state_dict(state_dict, strict=False)
        # scalar_head 随机初始化（小方差）
        nn.init.normal_(model.scalar_head.weight, std=0.02)
        nn.init.zeros_(model.scalar_head.bias)
        return model

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # 定位每个样本最后一个有效 token（即 <|judge|> token 的位置）
        # attention_mask.sum(dim=1) - 1 即为最后非 PAD token 的索引
        seq_lens = attention_mask.sum(dim=1) - 1       # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, seq_lens, :]   # [B, hidden_size]

        logit = self.scalar_head(last_hidden).squeeze(-1)  # [B]
        return logit  # raw logit，sigmoid 在 loss 计算时处理
```

**关键**：不用 `hidden[:, -1, :]`（会取到 PAD token），改用 `attention_mask` 精确定位最后有效 token。由于 Judge 输入末尾显式追加了 `<|judge|>`，这个 token 一定是最后一个有效 token，使得 Scalar Head 有稳定的语义锚点。

**Tokenizer 词表扩充 + Embedding 热启动（必须在 train.py 初始化时完成）**：

```python
# train.py 初始化阶段——引入 <|judge|> 特殊 token
tokenizer.add_tokens(["<|judge|>"])
# 扩充 Judge 的 embedding 层（Actor 和 ref_model 无需扩充，它们不处理 <|judge|>）
judge_model.transformer.resize_token_embeddings(len(tokenizer))

# ⚠️ 关键：Embedding 热启动！
# 直接 resize 后 <|judge|> 会被赋予完全随机的 Embedding 向量，
# 这个随机向量经过 Qwen 30+ 层 Transformer 后输出的 last_hidden 是纯噪声，
# Judge 前几个 Epoch 会陷入极度混乱，BT Loss 飙升。
# 解决方案：用语义相近的已有 token（如 Qwen3 的 <|im_end|>）的 Embedding 热启动。
with torch.no_grad():
    # 获取 <|im_end|> 的 token id（Qwen3 chat template 的结束标记）
    # 其语义为"对话结束/交付结果"，与 <|judge|> 的锚点角色最为接近
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    embedding_table = judge_model.transformer.get_input_embeddings()
    # 用 <|im_end|> 的 embedding 初始化 <|judge|>（最后一个新增 token）
    embedding_table.weight[-1] = embedding_table.weight[im_end_token_id].clone()
```

> 此操作让 `<|judge|>` 的初始 hidden state 处于合理范围，而非随机噪声，BT Loss 可在第一个 Epoch 内快速收敛。

---

### 5.6 `rollout_engine.py` — 两阶段 vLLM Rollout

**整体流程**：

```
输入 batch: List[RolloutSample]（每个 sample 对应 B 道题）

阶段 1：策略生成（K 个并行采样）
  - 为 B 道题各构建 prompt1（few-shot context → 策略生成）
  - vLLM 批量生成，每题 K 条策略 S_k，temperature=0.7
    SamplingParams(temperature=0.7, max_tokens=1024, stop=["</strategy>"])
  - vLLM 返回文本可能不含 stop word 本身，手动拼接：s_text = output.text + "</strategy>"
  - 同时从 vLLM 获取 S_k 的 log_prob（= log_prob_old，存入 Experience）

格式兜底（Format_Check）：
  - 若 S_k 不含 <strategy>...</strategy> 完整标签 → outcome=-1，跳过阶段 2

阶段 2：答案生成（greedy）
  - 对格式有效的 S_k，构建 prompt2（strategy + question → 答案生成）
  - vLLM 批量生成答案 A_k
    SamplingParams(temperature=0.0, max_tokens=512, stop=["</answer>"])
  - 手动拼接：a_text = output.text + "</answer>"

阶段 3：Env 评估
  - Env.evaluate(S_k, A_k, answer_gold) → y_k ∈ {-1, 0, 1}
  - 构建 Experience 对象（含 UUID traj_id）

输出：List[Experience]（写入 UCB Buffer）
```

> **为什么必须设 stop 字符串**：vLLM 批量生成时，如果某个样本触发幻觉 case，会不停输出直到 `max_model_len`。这不仅拖慢整个 batch（因为 vLLM 对齐最长序列），还会耗尽 KV Cache，导致后续 batch OOM。设置 `stop=["</strategy>"]` 后遇到结束标签立即截断。

**vLLM 生命周期管理（Ray 硬隔离 + 分布式主从编排，防止 OOM 和多进程并发灾难）**：

存在两个叠加的问题：
1. vLLM + FSDP 显存冲突（已知）
2. **`accelerate launch` 会启动 8 个平行 Python 进程**（每张 GPU 一个）。如果所有 8 个进程都执行 `VLLMActor.remote(...)`，Ray 会瞬间收到 8 个"申请 8 张卡创建 vLLM"的请求 → **8×8=64 张卡 → 死锁崩溃**。

**正确方案：主从节点编排**——vLLM Rollout 必须且只能由 **Rank 0（主进程）** 发起，生成数据后通过 PyTorch 分布式通信原语广播（Broadcast）给所有其他进程，然后再一起进入 FSDP 训练：

```python
# train.py 修正伪代码：分布式隔离 + Rank 0 广播
from accelerate import Accelerator
import torch.distributed as dist

accelerator = Accelerator()

@ray.remote(num_gpus=N_ROLLOUT_GPUS)
class VLLMActor:
    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,  # Few-shot context 相同，大幅提速
        )
    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params)

for epoch in range(total_epochs):
    experiences = None

    # ① 只有主进程（Rank 0）负责召唤 vLLM 并生成数据
    if accelerator.is_main_process:
        vllm_actor = VLLMActor.remote(current_model_path)
        experiences = ray.get(vllm_actor.generate.remote(batch_prompts, params))
        # 主动 kill vLLM，强制释放 GPU 显存
        ray.kill(vllm_actor)
        del vllm_actor

    # ② 强制所有 Rank 同步，等待 Rank 0 的 vLLM 显存释放完毕
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

    # ③ 将 Rank 0 生成的数据广播给所有 Rank！
    #    否则只有 Rank 0 有数据，其他 Rank 是 None，FSDP 无法拼 Batch
    experiences = broadcast_object_list_from_rank0(experiences)

    # ④ 所有 Rank 一起进入 FSDP 训练
    #    注意：judge_trainer 使用 buffer（off-policy），actor_trainer 使用 experiences（on-policy）
    #    Actor 绝对不能接触 buffer 中的历史数据！
    if not config.freeze_judge:
        judge_trainer.train_step(buffer)          # off-policy，历史数据
    actor_trainer.train_step(experiences, ...)    # on-policy，当前生成数据

    # ⑤ 权重保存到内存盘，供下轮 vLLM 加载
    if accelerator.is_main_process:
        accelerator.unwrap_model(actor).save_pretrained("/dev/shm/actor_weight_tmp")
    current_model_path = "/dev/shm/actor_weight_tmp"
```

> 重新启动 vLLM 约需 10-15 秒，但完全消除了 OOM 和多进程并发冲突两大风险。

---

### 5.7 `actor_trainer.py` — GRPO（三路 log-prob + action_mask）

**三路 log-prob 来源的区别**（极易混淆）：

| 变量 | 来源 | 是否有梯度 | 用途 |
|---|---|---|---|
| `log_prob_old` | vLLM rollout 时输出，存入 buffer | 无（buffer 中的 float） | ratio 分母 π_old |
| `log_prob_actor` | 当前 π_θ 实时 forward | 有梯度（求导节点） | ratio 分子 |
| `log_prob_ref` | 冻结 ref model 的 `torch.no_grad()` forward | 无梯度 | KL 惩罚参考基准 |

**action_mask 的必要性（致命漏洞）**：

`strategy_input_ids` 同时包含 Prompt（Few-shot + 问题，所有 K 个样本完全相同）和 Response（生成的策略 S_k，不同 K 之间有差异）。若对整条序列求 log-prob 之和，Prompt 的常数概率会污染 KL 散度和 Ratio——使得"用更长 Prompt 的样本"天然拥有更高绝对 log-prob，GRPO 从数学上就是错误的。

必须使用 `action_mask` 屏蔽 Prompt 部分和末尾 Padding，只对策略生成的 token 计算概率。

**action_mask 必须是三段式**：`[Prompt(0) | Strategy(1) | Padding(0)]`

构建方式：
```python
# 构建 action_mask（三段式）
# 1. 首先排除 PAD token（attention_mask 已经做了这一步，但要转为 int 类型作为乘法掩码）
action_mask = (input_ids != tokenizer.pad_token_id).int()   # [B, seq_len]，PAD 处为 0
# 2. 然后将 Prompt 部分强行置 0（prompt_lengths 是每条样本 Prompt 的 token 数）
for i, prompt_len in enumerate(prompt_lengths):
    action_mask[i, :prompt_len] = 0
# 结果：[Prompt=0 | Strategy=1 | Padding=0]，三段式
```

```python
def compute_log_probs(model, input_ids, attention_mask, action_mask):
    """
    Parameters
    ----------
    action_mask : Tensor, shape [B, seq_len]
        三段式：Prompt=0，Strategy=1，Padding=0。
    """
    logits = model(input_ids, attention_mask=attention_mask).logits

    # 错位对齐：预测下一个 token（LM 的标准做法）
    logits      = logits[:, :-1, :]       # [B, seq-1, vocab]
    labels      = input_ids[:, 1:]        # [B, seq-1]
    action_mask = action_mask[:, 1:]      # [B, seq-1]，随同错位

    per_token_log_probs = torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=2,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)                         # [B, seq-1]

    # 极其关键：只对 Strategy 生成阶段的 token 求和！
    return (per_token_log_probs * action_mask).sum(dim=-1)  # [B]
```

此函数对 Actor（有梯度）和 Reference Model（`torch.no_grad()`）均适用，传入对应的 `action_mask` 即可。

```python
# GRPO 更新循环开始前，冻结 ref_model 做一次 forward（本 epoch 固定不变）
with torch.no_grad():
    log_prob_ref = compute_log_probs(ref_model, input_ids, attn_mask, action_mask)

# per mini-batch：Actor 实时 forward（有梯度）
log_prob_actor = compute_log_probs(actor_model, input_ids, attn_mask, action_mask)
ratio = torch.exp(log_prob_actor - log_prob_old)   # log_prob_old 来自 buffer，无梯度
kl    = log_prob_actor - log_prob_ref              # KL ≈ log π_θ - log π_ref，标量 per sample

reward_raw = (1 - alpha) * outcome + alpha * sigma(v_judge) - beta * kl
```

**Z-Score 维度操作（含零方差短路保护）**：

```python
# batch 维度：[B*K] → reshape 到 [B, K] 再做 Z-Score
reward_flat = reward_raw             # shape: [B*K]
reward_bk   = reward_flat.view(B, K) # shape: [B, K]

mean_k = reward_bk.mean(dim=1, keepdim=True)  # [B, 1]
std_k  = reward_bk.std(dim=1, keepdim=True)   # [B, 1]

# 零方差短路保护：当 K 个样本奖励完全相同时（如 Greedy 采样或低温度），
# std_k → 0，直接用 1e-8 会产生极端噪声（如 ±150），一波带走 Actor 参数。
# 正确做法：std < 1e-4 时说明没有区分度，Advantage 全置 0（跳过此 batch 的梯度更新）。
mask_valid_std = (std_k > 1e-4).float()  # [B, 1]，方差过小的组全部清零
advantage = ((reward_bk - mean_k) / (std_k + 1e-8)) * mask_valid_std  # [B, K]

advantage_flat = advantage.view(B * K)   # reshape 回 [B*K] 用于 Loss

# GRPO Clip
ratio_clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
loss = -torch.min(ratio * advantage_flat, ratio_clipped * advantage_flat).mean()
```

**On-Policy 边界的代码架构约束**：GRPO 是严格的 On-Policy 算法，`log_prob_old` 必须是刚刚生成的当前 batch 数据，跨 Epoch 采样会导致 ρ (Ratio) 方差爆表，模型立刻崩溃。**在代码中必须保证：`buffer.sample()` 方法只能由 `judge_trainer` 调用，`actor_trainer` 只接受 `experiences`（当前 rollout 的新鲜数据），绝不传入 buffer。**

---

### 5.8 `judge_trainer.py` — ODVA（UCB 采样 + Bradley-Terry Loss）

```python
def train_step(self, buffer: UCBBuffer, batch_size: int):
    # Step 1: UCB 加权采样 Pairwise 数据
    pairs = buffer.sample_pairwise(batch_size)
    # pairs: List[(q_hash, traj_id_win, traj_id_lose, exp_win, exp_lose)]

    # Step 2: 构建 Judge 输入（win 和 lose 分别 tokenize）
    inputs_win  = tokenize([build_judge_prompt(p.exp_win)  for p in pairs])
    inputs_lose = tokenize([build_judge_prompt(p.exp_lose) for p in pairs])

    # Step 3: Bradley-Terry Loss + L2 Logit 锚定正则化
    logit_win  = judge_model(**inputs_win)   # [batch_size]
    logit_lose = judge_model(**inputs_lose)  # [batch_size]

    bt_loss = -F.logsigmoid(logit_win - logit_lose).mean()

    # Logit Drift 防护：BT Loss 只关注相对差值，不约束绝对值大小。
    # 若无正则化，logit_win 可能飙升至 +100，logit_lose 降至 -100，
    # 导致 sigmoid 梯度消失（饱和区）。L2 正则把绝对值拉回 0 附近。
    l2_penalty = 0.001 * (logit_win ** 2 + logit_lose ** 2).mean()

    loss = bt_loss + l2_penalty

    # Step 4: Backward
    loss.backward()
    optimizer.step()

    # Step 5: 写回 v_pred（惰性更新，O(1)）
    with torch.no_grad():
        new_v_win  = torch.sigmoid(logit_win).cpu().tolist()
        new_v_lose = torch.sigmoid(logit_lose).cpu().tolist()
    for i, p in enumerate(pairs):
        buffer.update_v_pred(p.q_hash, p.traj_id_win,  new_v_win[i])
        buffer.update_v_pred(p.q_hash, p.traj_id_lose, new_v_lose[i])
```

---

### 5.9 `train.py` — Phase II 主循环

```python
# 训练流程伪代码

# ─── 初始化 ───────────────────────────────────────────────────────
actor     = load_actor(config.actor_sft_checkpoint or config.actor_model_path)
ref_model = load_actor(same).requires_grad_(False)
judge     = JudgeModel.from_actor_checkpoint(same)
buffer    = UCBBuffer(max_size=config.buffer_max_size, ...)
vllm_llm  = LLM(model=config.actor_model_path, ...)

# accelerate FSDP 包装
actor = accelerator.prepare(actor)
judge = accelerator.prepare(judge)

# ─── Step 0: Judge Warmup（可选）────────────────────────────────
if config.judge_warmup:
    warmup_data = load_warmup_data(config)   # S_gold 正样本 + 扰动负样本
    for _ in range(config.warmup_epochs):
        judge_trainer.train_step_warmup(warmup_data)

# ─── Phase II 主循环 ─────────────────────────────────────────────
for epoch in range(config.total_epochs):
    for batch in dataloader:
        # Step 1: Rollout
        experiences = rollout_engine.run(batch, vllm_llm, K=config.K)
        for exp in experiences:
            buffer.add(exp)

        # Step 2: Judge 更新（ODVA）
        if not config.freeze_judge and len(buffer) >= config.min_buffer_size:
            judge_trainer.train_step(buffer, batch_size=config.judge_batch_size)

        # Step 3+4: Actor 更新（GRPO）
        # 先用冻结 ref_model 计算本 batch 所有样本的 log_prob_ref（固定）
        log_prob_ref = compute_log_probs_no_grad(ref_model, experiences)
        actor_trainer.train_step(experiences, log_prob_ref, judge, config)

    # epoch 结束：同步 Actor 权重到 vLLM
    sync_weights_to_vllm(actor, vllm_llm)

    # 定期验证
    if epoch % config.val_freq == 0:
        validate(actor, judge, val_dataset, config)
```

---

### 5.10 `config.py`

```python
@dataclass
class ActorJudgeConfig:
    # ── 模型路径 ──────────────────────────────
    actor_model_path: str = "/home/test/test16/chenlu/model/Qwen3-4B"
    actor_sft_checkpoint: str = ""   # FSDP 转换后的 HF 路径（留空则用 base）

    # ── 数据路径 ──────────────────────────────
    data_base_path: str = "/home/test/test16/chenlu/projects/LLMReflection/data/"
    train_subdir: str = "train_20k"
    val_subdirs: List[str] = field(default_factory=lambda: ["test-id-subtask", "test-ood-task", "test-bbh"])
    strategy_dir: str = "/home/test/test16/chenlu/projects/fs/strategy"

    # ── Rollout 超参 ──────────────────────────
    K: int = 8                        # 每题采样条数
    fewshot_min: int = 3
    fewshot_max: int = 5
    rollout_temperature: float = 0.7  # 策略生成温度
    answer_temperature: float = 0.0   # 答案生成温度（greedy）
    cross_domain_ratio: float = 0.5   # 跨域 batch 比例

    # ── Phase II RL 超参 ──────────────────────
    alpha: float = 0.3                # Dense reward 权重
    beta: float = 0.04                # KL penalty 系数
    lambda_err: float = 1.0           # UCB 利用项系数
    lambda_exp: float = 1.0           # UCB 探索项系数
    grpo_epsilon: float = 0.2         # PPO clip epsilon
    buffer_max_size: int = 50000
    min_buffer_size: int = 100        # 低于此值不做 Judge 更新

    # ── 训练超参 ──────────────────────────────
    total_epochs: int = 5
    train_batch_size: int = 8         # B
    actor_lr: float = 1e-6
    judge_lr: float = 1e-5
    judge_batch_size: int = 16
    judge_warmup: bool = True
    warmup_epochs: int = 2
    val_freq: int = 1                 # 每 N epoch 验证一次

    # ── 消融实验开关 ──────────────────────────
    freeze_judge: bool = False        # 实验A：冻结 Judge（测试共同进化的必要性）
    dense_reward_alpha: float = 0.3   # 实验B：设 0 = 纯稀疏奖励（测试稠密信号价值）
    kl_penalty_beta: float = 0.04     # 实验D：设 0 = 无 KL 惩罚（测试灾难性遗忘）
    disable_ucb_replay: bool = False  # 实验E：退化为 FIFO 均匀采样（测试反思机制）

    # ── 基础设施 ──────────────────────────────
    checkpoint_dir: str = "./checkpoints_actor_judge"
    save_freq: int = 1
    n_gpus: int = 8
    weight_sync_tmp_dir: str = "/dev/shm/actor_weight_tmp"
```

---

## 6. 关键技术问题与修正

### 6.1 Reference Model 的 log-prob 计算时机（算法层面）

**问题**：vLLM Rollout 返回的 `log_prob_S_k` 是生成时的策略概率（π_old），**不是** Reference Model（π_SFT）的概率。KL 惩罚需要的是 `log_prob_ref = log π_SFT(S|Q)`。

**修正**：在 actor_trainer.py 每次 GRPO 更新循环**开始前**，将 rollout 产生的 (Q, S_k) 喂给挂载在显存中的冻结 Reference Model，进行一次 `torch.no_grad()` 的 Forward，获取 `log_prob_ref`。这个值在本 Epoch 的多次 GRPO Mini-batch 更新中保持固定。

### 6.2 π_old 与 π_θ 的概念分离

- **π_old** (`log_prob_old`)：生成数据时的策略概率，直接由 vLLM rollout 时返回，存入 buffer，是历史快照（无梯度）
- **π_θ** (`log_prob_actor`)：当前 Actor 正在求导的网络输出，每次 mini-batch 实时 forward 计算（有梯度）
- **Ratio ρ** = exp(log_prob_actor − log_prob_old)

代码中千万不能把 vLLM 传回的 `log_prob_old` 当成实时梯度节点使用。

### 6.3 Judge hidden state 取 PAD token 的陷阱

**问题**：训练 Judge 时 batch 内样本长度不等，通常进行 right padding。此时 `hidden[:, -1, :]` 取到的是 `<PAD>` token 的 hidden state，语义毫无意义。

**修正**：
```python
seq_lens = attention_mask.sum(dim=1) - 1
batch_idx = torch.arange(hidden.size(0), device=hidden.device)
last_hidden = hidden[batch_idx, seq_lens, :]
```

### 6.4 Judge 输入的 EOS 锚点稳定性

**问题**：Judge 的 Scalar Head 需要在一个语义稳定的位置打分，但不同输入长度不同，"最后有效 token" 的含义随输入变化而变化。

**修正**：在 Judge prompt 末尾 `[/INST]` 之后追加自定义特殊 token `<|judge|>`（在 tokenizer 中注册），Scalar Head 专门基于该 token 的 hidden state 输出打分。

### 6.5 Buffer v_pred 写回的效率问题

**问题**：Buffer 的 key 是 q_hash，对应一个 list。如果用字符串匹配找到哪条需要更新 v_pred，O(N) 且容易出错。

**修正**：Experience 类增加 `traj_id: str`（UUID），写回时直接 `buffer[q_hash][traj_id].v_pred = new_v`，O(1) 完成。

### 6.6 UCB 公式中 T 的来源

Buffer 维护全局状态 `self.global_sample_steps`，每次 Judge 触发 `sample_pairwise()` 时递增。不能用 `len(buffer)` 代替 T（buffer 大小和采样次数不是一个概念）。

### 6.7 Actor 权重同步到 vLLM 的方案选择

**放弃 IPC**：DeepSpeed ZeRO-3 参数是 sharded 的，vLLM 需要完整连续权重。跨进程 IPC 极易 Core Dump。

**推荐方案**：
1. epoch 结束时 Actor 用 `save_pretrained` 写到 `/dev/shm`（内存盘，~10s IO）
2. vLLM 调用 `llm_engine.model_executor.load_weights()` 重载

**进阶方案（可选）**：引入 Ray Object Store，Actor gather 好全量权重后，通过 Ray 的零拷贝对象传递给 vLLM Ray Actor，消除文件 IO。

### 6.8 Z-Score 维度的张量操作陷阱

**问题**：GRPO 要求 K 个样本的 advantage 在同一道题 Q 内部归一化。如果同一 Q 的 K 个样本被 DataLoader 打散到不同 GPU，分布式 Z-Score 计算会产生错误结果。

**修正**：
- DataLoader 的 Sampler 必须保证同一 Q 的 K 个样本作为整体不被拆分
- 批次维度布局：`[B, K, seq_len]` → flatten 为 `[B*K, seq_len]` 做 forward → 计算 reward 后 reshape 回 `[B, K]` → 在 dim=1 上做 Z-Score → flatten 回 `[B*K]`

### 6.9 格式破坏数据防污染

**问题**：vLLM 生成不可控，若策略 S_k 格式完全破坏（如输出乱码、缺少 `</strategy>` 结束符），后续答案生成也会出错，这类数据若进入 Judge 训练会"毒化"评价标准。

**修正**：env.py 的 `Format_Check` 对格式完全破坏的样本返回 y=-1，buffer 记录但不参与 pairwise 采样。

### 6.10 action_mask 缺失（致命漏洞）

**问题**：`compute_log_probs` 若对整条 `input_ids`（含 Prompt + Response）求 log-prob 之和，Prompt 部分的 token 概率作为常数项混入，彻底污染 KL 散度和 Ratio。所有 K 个样本的 Prompt 完全相同，其 log-prob 差异为 0，不影响相对大小；但当 Prompt 很长时，其绝对值会主导总 log-prob，使得 KL 惩罚失真，进而导致 Actor 更新朝错误方向走。

**修正**：引入 `action_mask`（Prompt=0, Strategy=1），仅对 Strategy token 求和（见 5.7 节完整代码）。

### 6.11 vLLM 与 FSDP 显存冲突（必然 OOM）

**问题**：8 张 GPU 上同时运行 FSDP 训练状态（激活值 + 梯度 + AdamW 状态）和 vLLM KV Cache，即使是 Qwen3-4B 这样的小模型也绝对 OOM。vLLM 默认 `gpu_memory_utilization=0.9`，PyTorch 根本无法执行 backward。

**修正**：Ray 硬隔离——rollout 后主动 `ray.kill(vllm_actor)` 释放显存，`torch.cuda.empty_cache()` 确保归还，训练完毕再重新拉起 vLLM（见 5.6 节完整代码）。

### 6.12 `<|judge|>` Token 未注册导致失效

**问题**：若不在 tokenizer 中 `add_tokens` 并对 Judge 模型 `resize_token_embeddings`，遇到 `<|judge|>` 会报越界错误或 fallback 为 `[UNK]`，Scalar Head 彻底失效。

**修正**：
```python
tokenizer.add_tokens(["<|judge|>"])
judge_model.transformer.resize_token_embeddings(len(tokenizer))
```
注意：Actor 和 ref_model 的 embedding 无需 resize（它们不处理该 token）。

### 6.13 BT Loss Logit 漂移（梯度饱和）

**问题**：Bradley-Terry Loss 只优化 `logit_win - logit_lose` 的相对差值，不约束绝对大小。训练后期 `logit_win → +∞, logit_lose → -∞`，sigmoid 进入饱和区，梯度消失，Judge 停止学习。

**修正**：增加 L2 正则 `l2_penalty = 0.001 * (logit_win**2 + logit_lose**2).mean()`（见 5.8 节）。

### 6.14 vLLM 生成无 Stop Words（KV Cache 耗尽）

**问题**：批量生成若无 stop 字符串约束，遭遇幻觉 case 时模型无休止输出直到 `max_model_len`，拖慢整个 batch 并耗尽 KV Cache。

**修正**：`SamplingParams(stop=["</strategy>"])`，返回文本手动拼接结束标签（见 5.6 节）。

### 6.15 极难题无限堆积 Buffer

**问题**：若某问题永远答错（y=0），每次 rollout 都追加 (Q, S, y=0) 条目，最终该 q_hash 挤占 Buffer 大量槽位，恶化其他问题的采样概率。

**修正**：per-Q 容量上限（`per_q_max=50`），超出时淘汰 UCB 分数最低的轨迹（见 5.4 节）。

### 6.16 消融实验 B 的隐性作弊

**问题**：仅设 `alpha=0` 时 Judge 仍在后台训练（Step 2 照常运行），与完整系统算力消耗不对等，实验结论不可信。

**修正**：实验 B 正确配置为 `alpha=0 + freeze_judge=True`（同时禁用 Step 2），保证对比公平（见第 7 节）。

### 6.17 accelerate launch 多进程 Ray 并发灾难（致命）

**问题**：`accelerate launch` 启动 8 个平行 Python 进程（每张 GPU 一个）。若每个进程都执行 `VLLMActor.remote(...)`，Ray 会收到 8 个"申请 8 张卡"的请求 → 8×8=64 张卡 → 死锁崩溃，与 vLLM+FSDP 显存冲突叠加，双重致命。

**修正**：vLLM Rollout 只由 **Rank 0（主进程）** 启动，`accelerator.is_main_process` 保护；生成数据后通过 `broadcast_object_list_from_rank0()` 广播给所有 Rank；其他 Rank 在 `accelerator.wait_for_everyone()` 处阻塞等待（见 5.6 节完整代码）。

### 6.18 action_mask 未掩盖 Right Padding（严重）

**问题**：构建 Batch 时各样本的 Strategy 长度不同，必然进行 Right Padding。若 `action_mask` 只掩盖 Prompt 而不掩盖末尾 PAD token，模型会对 PAD token 计算 log_prob 和 KL 散度，严重扰乱梯度。

**修正**：`action_mask` 必须是三段式 `[Prompt=0 | Strategy=1 | Padding=0]`。构造方式：先用 `(input_ids != pad_token_id).int()` 排除所有 PAD，再将 Prompt 段强行置 0（见 5.7 节完整代码）。

### 6.19 Z-Score 零方差爆炸（严重）

**问题**：Greedy 采样或低温度时，K 个策略样本完全相同，Judge 对它们打出相同的分数，导致 `std_k → 0`。`std_k + 1e-8` 无法防止极端噪声（如 ±150），一波带走 Actor 参数。

**修正**：加入 `mask_valid_std = (std_k > 1e-4).float()`，std 过小时 Advantage 全置 0，跳过此 batch 的梯度更新——有限区分度时不如不更新（见 5.7 节）。

### 6.20 On-Policy 边界代码架构约束（严重）

**问题**：GRPO 是严格的 On-Policy 算法，`log_prob_old` 必须来自当前 batch 的即时生成，若意外让 Actor 触碰 UCB Buffer 中的历史 `log_prob_old`，Ratio 方差立即爆炸，模型崩溃。

**修正**：`buffer.sample()` 方法只在 `judge_trainer.py` 内调用；`actor_trainer.train_step(experiences, ...)` 只接受当前 rollout 的 `experiences` 参数，代码层面彻底隔离。

### 6.21 `<|judge|>` 随机 Embedding 导致初期 Loss 飙升（优化）

**问题**：`resize_token_embeddings` 后新 token 被赋予完全随机的 Embedding，经过 30+ 层 Transformer 输出的 `last_hidden` 是纯噪声，Judge 前几个 Epoch 会陷入极度混乱。

**修正**：在 resize 后用语义相近的已有 token（Qwen3 的 `<|im_end|>`，语义为"对话结束/交付结果"）的 embedding 热启动，见 5.5 节 `torch.no_grad()` 代码块。

### 6.22 Length Hacking 预警与防御

**问题**：RLHF 中模型容易学到"策略写得越长，Judge 打分越高"的捷径，形成 Reward Hacking，导致 Actor 生成冗长无意义的策略。

**修正**：必须在 wandb 监控 `Average_Strategy_Length`。若 `Pass@1` 连续 3 个 epoch 不涨而该指标陡升 >10%，启用轻度 Length Penalty（见第 8 节评估体系及 `config.py` 的 `length_penalty_coeff` 开关）。

### 6.23 vLLM Logprob 对齐陷阱（强烈建议简化）

**问题**：vLLM 返回的 logprobs 结构复杂，且由于 Tokenizer Prefix Space 机制，vLLM 返回的 token 序列与 PyTorch 里 `tokenizer(prompt + response)` 算出的 `input_ids` 极易发生 **Misalignment（错位）**。只要错 1 个 token，GRPO 的 Ratio 就全错，且排查极其痛苦。

**修正**：**彻底放弃从 vLLM 获取 `log_prob_old`**。让 vLLM 只返回纯文本（还能提升生成速度并降低显存占用）。在 `actor_trainer.py` 中，拿到文本并构建好 `input_ids` 后，在做 GRPO 第一个 mini-batch 更新前，用当前 Actor 做一次 `torch.no_grad()` forward：

```python
# actor_trainer.py 中，train_step 最开头
with torch.no_grad():
    log_prob_old = compute_log_probs(actor, input_ids, attention_mask, action_mask)
    # 此时 Actor 权重与 rollout 时完全相同（epoch 内尚未更新），对齐完美
log_prob_old = log_prob_old.detach()
```

> 这是最稳妥的方案：tokenization 由同一套 PyTorch tokenizer 完成，不存在任何 offset 问题。

### 6.24 Ray `num_gpus` 死锁伪命题（必须 Hack）

**问题**：`@ray.remote(num_gpus=N_ROLLOUT_GPUS)` 在与 PyTorch FSDP 混跑时必然触发死锁。此时 8 张卡已被 PyTorch 进程"物理占有"，Ray 调度器认为"没有空闲卡"，Task 永远处于 Pending 状态，程序死锁。

**修正**：欺骗 Ray，由 vLLM 内部接管硬件分配：

```python
# 将 num_gpus 设为 0，告诉 Ray 调度器不要管 GPU 分配
@ray.remote(num_gpus=0)
class VLLMActor:
    def __init__(self, model_path):
        # vLLM 内部的 tensor_parallel_size 会自动吃满可见的 GPU
        self.llm = LLM(model=model_path, tensor_parallel_size=8,
                       gpu_memory_utilization=0.75,
                       enable_prefix_caching=True)
```

> Ray 不负责 GPU 资源计量，vLLM 的 `tensor_parallel_size` 自行接管所有可见卡。

### 6.25 `gpu_memory_utilization` 不可超过 0.75

**问题**：即便 Rank 0 调用了 `torch.cuda.empty_cache()`，PyTorch 的 CUDA Context（上下文句柄）在每张 GPU 上仍会物理"钉住"约 **1GB~1.5GB** 显存（这是 Nvidia Driver 层面的机制，无法释放）。若设 0.9 或 0.85，vLLM 启动时会直接 CUDA Out of Memory。

**修正**：对于 Qwen3-4B 在 8 卡（8×80G 或 8×40G）环境，`gpu_memory_utilization` 稳妥设为 **0.7 或 0.75**，留出足够 buffer 给 PyTorch 常驻 Context。

### 6.26 Experience 不能含 GPU Tensor（NCCL Hang）

**问题**：`broadcast_object_list_from_rank0(experiences)` 底层使用 Python 的 pickle 序列化。如果 `Experience` dataclass 里混入哪怕一个 GPU 上的 `torch.Tensor`，广播时会导致 NCCL 直接 **Hang 死（永久卡住无报错）**，极难排查。

**修正**：`buffer.py` 中 `Experience` 的所有字段必须严格为纯 Python 标量：

```python
@dataclass
class Experience:
    traj_id: str          # UUID str
    context_text: str     # 纯文本
    question: str
    strategy: str
    outcome: int          # int，不是 Tensor
    n_sampled: int
    v_pred: float         # float，不是 Tensor
    timestamp: int
    log_prob_old: float   # ← 注意：存 float 标量，不是 Tensor（虽然改为 actor forward 算，但存时仍需 .item()）
    # 绝对禁止任何 torch.Tensor 字段！
```

### 6.27 `ref_model` 未分发导致 64GB 显存冗余

**问题**：若 `ref_model` 没有经过 `accelerator.prepare()`，每个 Rank（0~7）都会在自己的卡上加载一个完整的 4B 模型实例。这不仅使分布式显存管理完全失效，还凭空消耗 **8×8GB = 64GB** 的冗余显存。

**修正**：必须用 Accelerate 分发 `ref_model`。由于 ref_model 不参与梯度计算，最优雅的方案是 CPU Offload——仅在需要计算 `log_prob_ref` 时分块加载到 GPU：

```python
# train.py 初始化阶段
actor     = accelerator.prepare(actor)      # FSDP 分布式训练
judge     = accelerator.prepare(judge)      # FSDP 分布式训练
ref_model = accelerator.prepare(ref_model)  # 也必须分发！否则每卡各一份完整模型

# 或者更省显存的做法：CPU Offload（ref_model 永不参与 backward）
# ref_model 保持在 CPU，计算时临时移到 GPU，用完立刻释放
with torch.no_grad():
    ref_model.to(accelerator.device)
    log_prob_ref = compute_log_probs(ref_model, input_ids, attention_mask, action_mask)
    ref_model.to("cpu")
    torch.cuda.empty_cache()
```

---

## 7. 消融实验设计

| 实验组 | 开关组合 | 目标 | 预期结果 |
|---|---|---|---|
| 完整系统 | 默认配置 | 基准 | 最好效果 |
| A: w/o Judge Update | `freeze_judge=True` | 验证 Judge 更新的必要性 | 性能下降（Judge 无法校准） |
| B: w/o Dense Reward | `dense_reward_alpha=0` **+** `freeze_judge=True`（或直接跳过 Step 2） | 验证稠密奖励的价值 | 收敛变慢，低分策略奖励信号消失 |
| D: w/o KL Penalty | `kl_penalty_beta=0` | 验证 KL 防灾难遗忘 | 可能出现 reward hacking 或乱码输出 |
| E: w/o UCB Replay | `disable_ucb_replay=True` | 验证反思机制 | Judge 遗忘早期策略，灾难性遗忘 |

> **实验 B 的隐性作弊陷阱**：若仅设 `dense_reward_alpha=0`，Actor 的 Reward 不再包含 Judge 打分，但 Judge 仍会在后台执行 Step 2（ODVA）更新，消耗大量显存和算力训练一个对 Actor 毫无影响的模型。这使得实验 B 与完整系统相比资源消耗不对等，结论不可信且极其浪费算力。  
> **正确做法**：运行实验 B 时，必须同时禁用 Judge 训练（设 `freeze_judge=True` 或在代码中直接跳过 Step 2 的 judge_trainer.train_step 调用）。实验 B 的语义应该是"只用稀疏奖励 + 冻结 Judge"，与实验 A 的区别在于 A 的 Judge 在线更新但不参与 Reward，而 B 连 Judge 都不更新。

---

## 8. 评估指标体系

### 结果指标

- **`Pass@1_InDomain`**：同域验证集（test-id-subtask）答案正确率
- **`Pass@1_OOD`**：跨域测试集（test-ood-task、test-bbh）答案正确率
  - 证明策略具有跨域泛化能力

### 过程指标

- **`JOA`（Judge-Outcome Agreement）**：
  - Judge 高分样本（σ(V) > 0.6）中 y=1 的比例
  - Judge 低分样本（σ(V) < 0.4）中 y=0 的比例
  - 两者曲线均应随训练上升（证明 Judge 校准准确）

### 机制指标

- **`ECR`（Execution Consistency Rate）**：抽样检查答案 A 是否严格遵循策略 S 中的步骤（通过关键词匹配或 LLM Judge），监控 Reward Hacking
- **`Buffer Polarity Distribution`**：各 epoch 中 y=1/0/-1 的比例变化
- **`Average_Strategy_Length`**：每个 epoch 所有生成策略的平均 token 长度（**必须监控，防止 Length Hacking**）

### 长度作弊（Length Hacking）预警机制

RLHF 中一个经典问题：模型会学到"只要策略写得足够长，Judge 就会认为步骤详实而打高分"。一旦 Judge 染上这种偏好，Actor 会疯狂生成又臭又长且毫无意义的策略（Reward Hacking）。

**监控方法**：在 Wandb 面板中，如果发现 `Pass@1` 没有提升，但 `Average_Strategy_Length` 呈陡峭上升趋势，说明 Judge 已被长度欺骗。

**应对方案**（按级别从轻到重）：
1. **Pairwise 采样时的短文偏好**：在 `judge_trainer.py` 的 Bradley-Terry 采样中，若两条 experience 正确性（y 值）相同，强制将较短的策略设为 win，向 Judge 注入"相同质量下更简洁更好"的先验。
2. **轻度长度惩罚**：在 `env.py` 的 Reward 函数中加入 `length_penalty = -0.01 * max(0, strategy_len - 500)`，超出合理长度后线性惩罚。
3. **触发阈值**：只有当 `Average_Strategy_Length` 在连续 3 个 epoch 上升超过 10% 时才启用惩罚，避免过早干预。

所有指标通过 `wandb` 记录并可视化，实验名称格式：`actor_judge_{消融条件}_{时间戳}`。

---

## 9. 快速启动命令

### Step 0: 环境准备

```bash
conda activate agl
pip install deepspeed   # 若尚未安装
```

### Step 1: Checkpoint 转换

```bash
cd /home/test/test16/chenlu/projects/agent-lightning
python examples/actor-judge/convert_checkpoint.py \
    --checkpoint-dir checkpoints/global_step_400/actor \
    --output-dir checkpoints/global_step_400/actor_hf
```

### Step 2: 启动 Phase II 训练

```bash
accelerate launch --config_file examples/actor-judge/fsdp_config.yaml \
    examples/actor-judge/train.py \
    --actor-sft-checkpoint checkpoints/global_step_400/actor_hf \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --strategy-dir /home/test/test16/chenlu/projects/fs/strategy \
    --K 8 \
    --cross-domain-ratio 0.5 \
    --wandb-project ActorJudgeCoEvolution
```

### Step 3: 消融实验示例

```bash
# 实验A：冻结 Judge
python examples/actor-judge/train.py --freeze-judge

# 实验B：纯稀疏奖励
python examples/actor-judge/train.py --dense-reward-alpha 0.0

# 实验E：退化为 FIFO Buffer
python examples/actor-judge/train.py --disable-ucb-replay
```

---

*文档版本：2026-03-19*  
*基于 Review 意见（算法修正 + 工程修正 5 条）更新*

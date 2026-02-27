# Early Experience：无需 Reward 的 Agent 学习范式，连接模仿学习与强化学习

> 论文：**Agent Learning via Early Experience**  
> 作者：Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao 等（Meta Superintelligence Labs / FAIR at Meta / Ohio State University）  
> 链接：<https://arxiv.org/abs/2510.08558>（arXiv:2510.08558 v2，2025-10-13）  
> 机构：Meta + OSU NLP Group，30 位作者的大型合作

---

## 0. 先说清楚：这篇论文在做什么

这篇论文提出了一个新的训练范式，叫做 **Early Experience（早期经验）**，试图解决一个长期困扰 language agent 研究的核心矛盾：

> **模仿学习（IL）**：需要专家数据，成本高，泛化差，agent 只被动学。  
> **强化学习（RL）**：理论上最强，但很多环境没有可验证的奖励信号（网页、工具调用等），而且长轨迹的 credit assignment 非常困难。

Early Experience 是一个"中间地带"——让 agent **自己执行动作、观察结果**，把这些"后续状态"直接作为**无奖励监督信号**，既不依赖专家高质量标注，也不依赖环境 reward。

**三种范式的定位图**：

```
┌────────────────────────────────────────────────────────────────────┐
│  Era of Human Data  │   Early Experience   │  Era of Experience    │
│  （模仿学习时代）    │       （本文）        │  （强化学习时代）     │
│                     │                      │                       │
│  依赖专家 demo       │  Agent 自己探索产生   │  环境提供可验证 reward │
│  不需要 reward       │  的"未来状态"作为监督 │  稀疏/延迟 reward     │
│  数据成本高          │  无需 reward          │  基础设施不成熟        │
│  泛化差             │  数据可扩展           │  训练不稳定            │
└────────────────────────────────────────────────────────────────────┘
         ←────────── Early Experience 是这两者的桥梁 ──────────►
```

---

## 1. 摘要提炼

### 1.1 核心问题

| 训练范式 | 优点 | 局限 |
|---------|------|------|
| 监督微调（SFT/IL） | 简单稳定，不需要 reward | 专家数据贵；泛化差；agent 不了解自己行动的后果 |
| 强化学习（RL） | 从试错中学，理论上最强 | 很多环境无验证 reward；长轨迹 credit assignment 困难；基础设施不成熟 |
| **Early Experience（本文）** | **无需 reward；数据可扩展；连接 IL 和 RL** | 依赖环境交互；短轨迹信号更有效 |

### 1.2 两种核心方法

1. **Implicit World Modeling（IWM，隐式世界建模）**：让 agent 学习"我做了动作 A 之后，世界变成了什么样"——通过预测未来状态来内化环境动力学。
2. **Self-Reflection（SR，自我反思）**：让 agent 学习"为什么专家选 A 而不是 B"——通过对比自己提出的替代动作和专家动作的结果，生成 chain-of-thought 解释，再用这些解释训练自己。

### 1.3 核心结论

- **8 个不同环境**（体感/科学仿真/网页导航/工具调用/长程规划）全部有提升
- **3 种模型**（Llama-3.2-3B、Qwen-2.5-7B、Llama-3.1-8B）全部有效
- Early experience 训练后的 checkpoint 作为 RL warm-start，**后续 RL 效果更强**
- 只需 **1/8 的专家数据**，early experience 就能媲美甚至超过用全量数据的 IL

---

## 2. 整体框架：Early Experience 的数据流

整个系统建立在专家轨迹 + 自主探索的双轨数据上：

```
Expert Dataset:  D_expert = {(s_i, a_i)}  ← 专家状态-动作对（已有）

                           ↓ agent 自主在每个 s_i 提出 K 个替代动作 a_i^j

Rollout Dataset: D_rollout = {(s_i, a_i^j, s_i^j)}  ← (状态, 替代动作, 结果状态)
                                                        不需要 reward！

                ┌──────────────────────────────────────────┐
                │         两种 Early Experience 方法          │
                ├──────────────────────┬───────────────────┤
                │  Implicit World      │  Self-Reflection   │
                │  Modeling (IWM)      │  (SR)              │
                │                     │                    │
                │ 训练目标：给定        │ 训练目标：给定       │
                │ (s_i, a_i^j)         │ (s_i, 专家动作,     │
                │ 预测结果状态 s_i^j   │  替代动作+其结果)    │
                │                     │ 生成 CoT + 专家动作  │
                └──────────────────────┴───────────────────┘
```

---

## 3. 方法详解

### 3.1 数据构建：Rollout Dataset

对 expert dataset 中的每个状态 sᵢ，让初始 policy（instruction-tuned LLM）采样 K 个替代动作：

```python
A_i = {a_i^1, a_i^2, ..., a_i^K}   # K 个替代动作，均不同于专家动作 a_i

# 实际执行每个替代动作，收集结果状态
D_rollout = {(s_i, a_i^j, s_i^j) | i∈[N], j∈[K]}
```

每条三元组 `(状态, 替代动作, 结果状态)` 就是 agent 的一次"试探性经验"，不依赖任何 reward 评分。

**关键参数：分支因子 K（Branching Factor）**

| K 大小 | IWM 效果 | SR 效果 |
|--------|---------|---------|
| K=1 | 基础提升 | 基础提升 |
| K=2~4 | 持续提升 | **最优** |
| K=8 | **最优** | 边际递减（对比信号变弱）|

IWM 倾向更大 K（更多状态转移覆盖），SR 在中等 K 最好（对比太多替代方案会降低推理质量）。

---

### 3.2 方法一：Implicit World Modeling（IWM）

**核心思路**：用 `(状态, 动作) → 预测下一状态` 作为辅助训练任务，让 policy 内化环境动力学——"做了 A 之后世界会变成什么样"。

**训练目标**：

```
L_IWM = -Σ log p_θ(s_i^j | s_i, a_i^j)
```

即用 next-token prediction 训练模型，给定当前状态和动作，预测结果状态（纯文本）。

**两阶段训练流程**：

```
Stage 1：用 L_IWM 在 D_rollout 上训练（内化环境动力学）
   ↓
Stage 2：继续用 L_IL 在 D_expert 上训练（学专家决策）
总训练步数与纯 IL baseline 相同（不多用计算）
```

**举例（WebShop 购物）**：

```
状态：产品详情页，有 "non-ears blue" 和 "click[< prev]" 等选项

IWM 训练样本：
- click[non-ears blue] → "进入新的产品详情页，显示颜色选项、尺码等属性..."
- click[< prev] → "返回搜索结果页，显示多款耳机产品列表..."
- click[buy now] → "进入结账确认页，显示购买成功信息和奖励分数..."
```

通过预测这些状态转移，模型学会"点击不同按钮各会发生什么"，无需任何奖励信号。

---

### 3.3 方法二：Self-Reflection（SR）

**核心思路**：比较专家动作和替代动作的结果，生成自然语言解释"为什么专家选 A 而不是 B"，再用这些 CoT 解释 + 专家动作来训练模型。

**训练目标**：

```
L_SR = -Σ log p_θ(c_i^j, a_i | s_i)
```

其中 c_i^j 是 chain-of-thought 解释（为什么专家动作优于替代动作 a_i^j）。

**数据构建流程**：

```
对每个专家状态 s_i：
1. 执行专家动作 a_i → 观察 s_{i+1}
2. 执行替代动作 a_i^j（j=1..K）→ 观察 s_i^j

3. 把这些信息喂给 LLM，生成解释 c_i^j：
   "为什么专家选 a_i 而不是 a_i^j？"（基于 s_{i+1} 和 s_i^j 的对比）

4. 收集 D_refl = {(s_i, a_i^j, c_i^j)}
5. 混合 D_refl 和 D_expert 做 SFT
```

**SR 的核心优势**：反思基于**实际执行结果**（grounded），而非凭空推理（ungrounded）——这解释了为什么 STaR-style 方法（不执行替代动作，直接生成理由）效果远不如 SR。

---

## 4. Prompt 设计：Self-Reflection 完整模板

这是 SR 方法数据合成的核心 prompt，也是全文最关键的工程设计：

```
[系统/用户 Prompt]

You will be presented with a situation where you need to choose between 
multiple possible actions. Your task is to analyze the situation and 
provide reasoning about why we decide to take the expert action.

• Situation Description (s_i): {当前状态描述}

• Expert Action (a_i): {专家动作}

• Expected Outcome (s_{i+1}): {执行专家动作后的结果状态}

• Alternative Actions:
  1. Action a_i^1: {替代动作1}, resulting state s_i^1: {状态1}
  2. Action a_i^2: {替代动作2}, resulting state s_i^2: {状态2}
  3. ...

Provide a detailed self-reflection as an internal monologue that 
demonstrates your reasoning process for the current situation. 
Your monologue should:
  1. Analyze the situation and the goal.
  2. Compare the possible actions, explaining why each may be less optimal.
  3. Justify why the expert action is most suitable, grounded in the 
     expected outcome.
  4. Highlight any relevant clues, constraints, or consequences.

Guidelines:
  • Stay strictly within the provided information.
  • Avoid meta-commentary about being an AI.
  • Use natural, step-by-step reasoning.
  • Focus on logical decision-making.

Output: Directly write the self-reflection monologue, no extra headings, 
disclaimers, or external notes.
```

**Prompt 设计要点**：

| 设计决策 | 原因 |
|---------|------|
| 提供结果状态（Expected Outcome / resulting state） | 让 CoT **基于实际发生的事**，而非凭空假设 |
| 要求 "internal monologue"（内心独白）| 鼓励自然的 step-by-step 推理，而非 AI 格式化回答 |
| "Stay strictly within the provided information" | 防止幻觉（hallucination） |
| "Compare actions, explain why each may be less optimal" | 强制显式对比，学到决策边界 |
| 不要 headings/disclaimers | 保持输出格式与实际推理一致 |

### 各环境 SR 示例对比

**WebShop（购物）**：

```
Situation: 产品详情页，任务需要 "blue wireless bluetooth headphones, price < $130"
Expert Action: click[non-ears blue]
Alternative 1: click[< prev] → 返回搜索结果
Alternative 2: click[buy now] → 直接购买

SR Output:
"While the red shirt matches the color preference, it exceeds the $20 budget 
constraint specified in the query. The blue shirt satisfies both the style 
requirement and budget limit. This teaches the model to prioritize constraints."
```

**TravelPlanner（旅行规划）**：

```
Situation: Day1 交通待定，预算 $1700，唯一可选航班 F3573659：$474
Expert Action: 选择 Flight F3573659
Alternative: SKIP_TRANSPORTATION（跳过交通）

SR Output:
"Flight F3573659 is a valid option at $474. It satisfies all constraints and 
gets me to my destination efficiently. The only alternative is skipping 
transportation, which prevents reaching the destination. Given all constraints 
and optimization factors, Flight F3573659 is indeed the best choice."
```

**BFCLv3（工具调用）**：

```
Situation: 在 workspace 目录，需要把 log.txt 移到 archive 目录
Expert Action: 使用 mv 工具
Alternative 1: ls → 列出文件（无法完成任务）
Alternative 2: rm → 删除文件（错误操作）

SR Output:
"The 'rm' command would remove the file instead of moving it. Creating a 
directory using 'mkdir' is redundant since 'archive' already exists. 
Therefore, moving the file to the 'archive' directory is the best action."
```

---

## 5. 训练范式：无 Reward 设计

Early Experience 的 **reward 设计**是一个"反设计"——明确选择**不依赖任何外部 reward**，这是整个范式最核心的立场。

### 5.1 监督信号来源

| 方法 | 监督信号来源 |
|------|------------|
| 纯 IL | 专家动作标签 |
| IWM | **未来状态本身**（环境反馈，无需打分）|
| SR | **LLM 基于状态对比生成的 CoT 解释**（无需外部 reward）|
| RL | 环境 reward（很多场景不可用） |

### 5.2 为什么"未来状态"足以作为监督？

两个机制：

1. **隐性负反馈**：若 agent 点击了错误按钮，看到了"Error: Invalid action"或"Page not found"——这本身就是一种无 reward 的惩罚信号，只不过用"预测这个错误状态"来传递。
2. **对比性优势**：SR 中，agent 同时看到专家动作和替代动作的结果，自然形成"哪个更好"的判断，无需外部评分。

### 5.3 对比 STaR 的差距（核心消融）

| 方法 | WebShop | ALFWorld |
|------|---------|---------|
| IL（Llama-8B）| 47.3% | 80.5% |
| +Long CoT（推理时增强）| 0%（崩溃）| 25.8%（崩溃）|
| +STaR 风格（无接地推理）| 25.0% | 74.2% |
| **Ours-IWM** | **58.6%** | **85.9%** |
| **Ours-SR** | **58.2%** | **85.2%** |

STaR 的问题：生成的推理未经实际执行验证，容易**幻觉化工具/事实**，fine-tune 上去反而有害。
Early Experience 的优势：CoT 完全基于**观察到的实际结果**，不会凭空捏造。

---

## 6. Evaluation Metric 设计

### 6.1 多环境、多模型的大规模评估

| 环境 | 任务类型 | 主要指标 |
|------|---------|---------|
| ALFWorld | 具身指令跟随（家庭）| Success Rate (%) |
| ScienceWorld | 科学实验仿真 | Success Rate (%) |
| TravelPlanner | 长程旅行规划 | Final Pass Rate (%) |
| BFCLv3 | 多轮工具调用 | Success Rate (%) |
| Tau-Bench | 客服多轮 API | Success Rate (%) |
| SearchQA | 多跳问答+检索 | F1 Score |
| WebShop | 网页购物导航 | Success Rate (%) + Score |
| WebArena-Lite | 真实网页任务 | Success Rate (%) |

### 6.2 三个维度的评估

1. **in-domain effectiveness**（任务有效性）：同分布测试集上的 SR
2. **OOD generalization**（跨域泛化）：跨任务/跨设定的 SR
3. **RL warm-start quality**（下游 RL 质量）：作为 GRPO 初始化后的最终 SR

### 6.3 三种模型家族

- Llama-3.2-3B-Instruct（小模型）
- Qwen-2.5-7B-Instruct（中模型）
- Llama-3.1-8B-Instruct（中模型）
- 额外：Llama-3.3-70B（大模型，WebArena-Lite）

---

## 7. 核心实验结果

### 7.1 全环境有效性（Table 2 摘要）

| 环境 | IL 基线 | IWM | SR | 最大提升 |
|------|---------|-----|-----|--------|
| ALFWorld | 78~80% | +4.7~5.5% | +3.9~7.8% | **+7.8%** |
| ScienceWorld | 52~55% | +2.3~5.5% | +3.9~13.3% | **+13.3%** |
| TravelPlanner | 17~19% | +5.5~8.9% | +12.8~15.0% | **+15.0%** |
| BFCLv3 | 16~27% | +2.6~4.0% | +4.0~8.0% | **+8.0%** |
| Tau-Bench | 24~36% | +1.8~4.9% | +4.4~5.8% | **+5.8%** |
| SearchQA (F1) | 38~41% | +0.9~3.3% | +0.6~2.1% | **+3.3%** |
| WebShop | 42~52% | +4.6~18.4% | +10.6~10.9% | **+18.4%** |
| WebArena-Lite | 4~6% | +2.4~3.6% | +1.2~3.6% | **+3.6%** |

**规律**：
- 有长程约束推理（TravelPlanner、ScienceWorld）→ SR 效果更好
- 有稳定状态转移（WebShop、ALFWorld）→ IWM 效果更好
- 总体 SR > IWM，但两者互补

### 7.2 数据效率（关键发现）

仅需 **1/8 的专家数据**，Early Experience 就能超过用全量数据的纯 IL：

| 专家数据比例 | IL 成功率 | IWM 成功率 | SR 成功率 |
|------------|---------|-----------|---------|
| 1/8 | 25.8% | 38.3% | 46.9% |
| 1/4 | 33.6% | 43.0% | 54.7% |
| 1/2 | 44.6% | 51.6% | 55.5% |
| 1 (全量) | 45.3% | **58.6%** | **59.4%** |

WebShop 上，SR(1/8 专家数据) > IL(全量专家数据)。

### 7.3 作为 RL Warm-Start 的优势（图 3）

```
WebShop（Llama-3.1-8B）：
  IL → GRPO = 47.3% → 80.5%    (RL提升约 +33%)
  IWM → GRPO = 58.6% → 91.4%   (RL提升约 +33%, 最终更高)
  SR → GRPO = 58.2% → 89.8%    (RL提升约 +32%, 最终更高)

ALFWorld：
  IL → GRPO = 80.5% → 93.8%
  IWM → GRPO = 85.9% → 97.7%
  SR → GRPO = 85.2% → 98.5%   ← 最高！
```

Early experience 不只是替代 RL 的选项，而是让 RL 启动点更高、最终性能天花板更高的**基础**。

### 7.4 OOD 泛化（Table 3）

| 环境 | IL（OOD）| IWM（OOD）| SR（OOD）|
|------|---------|----------|---------|
| ALFWorld | 63~74% | +3.1~14.8% | +3.1~9.4% |
| BFCLv3 | 5.3~9.3% | +0.9~5.3% | +0.7~8.5% |
| SearchQA | 40~47% | +2.2~4.9% | +3.3~4.2% |

OOD 提升有时**比 in-domain 更大**，说明 early experience 学到的是可迁移的决策原则，不只是记忆特定场景。

### 7.5 模型规模（图 5，WebArena-Lite）

| 模型规模 | IL SR | IWM SR | SR SR |
|---------|-------|--------|-------|
| 3B | 6.1% | 8.5% | 7.3% |
| 8B | 4.9% | 8.5% | 8.5% |
| **70B** | **13.3%** | **16.4%** | **15.2%** |

70B 模型上优势保持，说明 early experience 的价值不随模型变大而消失。

---

## 8. 算法总结

```python
# Early Experience 完整流程

# Step 1：构建 Rollout Dataset（两种方法共享此步骤）
D_rollout = []
for (s_i, a_i) in D_expert:
    alt_actions = policy.sample(s_i, K=8, exclude=a_i)  # K 个替代动作
    for a_j in alt_actions:
        s_j = env.step(s_i, a_j)                          # 实际执行，观察结果
        D_rollout.append((s_i, a_j, s_j))

# ─────────── Method 1: Implicit World Modeling ───────────

# Stage 1：世界建模（预测未来状态）
for (s_i, a_j, s_j) in D_rollout:
    L_IWM += -log p_θ(s_j | s_i, a_j)  # next-token prediction on s_j

model = finetune(model, L_IWM, data=D_rollout)

# Stage 2：模仿学习（继续，总步数不变）
model = finetune(model, L_IL, data=D_expert)

# ─────────── Method 2: Self-Reflection ───────────

# Stage 1：合成反思数据
D_refl = []
for (s_i, a_i) in D_expert:
    s_next = env.step(s_i, a_i)                          # 专家动作结果
    alts_with_results = [(a_j, s_j) for (s_i, a_j, s_j) in D_rollout if same_state]
    c_i = llm.reflect(s_i, a_i, s_next, alts_with_results, prompt=SR_PROMPT)
    D_refl.append((s_i, a_i, c_i))

# Stage 2：联合训练（反思数据 + 专家数据混合）
combined = D_refl + D_expert
model = finetune(model, L_SR, data=combined)
# L_SR = -log p_θ(c_i, a_i | s_i)  # 预测 CoT + 专家动作
```

---

## 9. 关键消融

| 消融维度 | 关键结论 |
|---------|---------|
| Early Exp vs STaR | STaR 无接地推理有时降低性能；Early Exp 始终提升 |
| Early Exp vs Long CoT | Long CoT 在 fine-tuned 模型上几乎不起作用（推理链崩溃）|
| 分支因子 K | IWM 偏好大 K；SR 在 K=2~4 最优 |
| 专家数据量 | 1/8 数据量的 EE 能追上全量 IL |
| 模型规模 | 3B 到 70B 全部有效，绝对提升随规模增大 |
| RL 初始化 | EE warm-start 显著优于 IL warm-start（最终性能天花板更高）|

---

## 10. 局限性

| 局限 | 说明 |
|------|------|
| 短轨迹信号更有效 | 目前方法聚焦单步探索，长程 credit assignment 仍未解决 |
| 需要与环境交互 | 不适用于离线/只有静态数据的场景 |
| SR 依赖 LLM 反思质量 | 若反思 LLM 质量差，生成的 CoT 可能有误 |
| 分支因子设计 | K 的选择对不同环境有所不同，没有一键最优设置 |

---

## 11. 结论总结

| 维度 | 核心结论 |
|------|---------|
| **范式定位** | Early Experience 是 IL 和 RL 之间的实用桥梁：无需 reward，数据可扩展 |
| **方法 IWM** | 通过预测未来状态来内化环境动力学；结构化环境中最有效 |
| **方法 SR** | 通过接地 CoT 反思（基于实际结果对比）学习决策原则；长程约束推理中最有效 |
| **数据效率** | 1/8 专家数据 + EE 可超过全量 IL |
| **RL 协同** | EE 作为 RL warm-start，最终性能显著优于直接 IL warm-start |
| **泛化** | OOD 提升有时 ≥ in-domain，说明学到可迁移原则而非记忆特定场景 |
| **规模** | 3B 到 70B 全部有效，EE 与模型规模协同而非替代 |

---

## 12. 对我们项目（Agent-Lightning 策略提取）的启发

我们的项目是：**从 few-shot 示例中提取解题策略，再应用策略解决新问题**。Early Experience 的核心思想与我们有深度共鸣——都面临"如何在没有大量高质量标注的情况下，让模型学到更好的决策原则"的问题。

### 12.1 Self-Reflection → 策略选择的"接地反思"

SR 的本质是：**基于实际执行结果对比**，生成"为什么专家决策优于替代决策"的解释。

**对我们项目的直接类比**：

```
Agent 场景的 Self-Reflection：
  状态 s → 专家动作 a_expert → 结果 s'
  状态 s → 替代动作 a_alt   → 结果 s_alt
  → 生成：「为什么应该选 a_expert 而不是 a_alt？」

我们项目的类比：
  题目 q → 策略 π_expert → 正确答案
  题目 q → 策略 π_alt    → 错误答案
  → 生成：「为什么应该用 π_expert 而不是 π_alt 来解这道题？」
```

**关键借鉴**：
- 对同一道题，让模型生成 K 条候选策略（分支因子），每条策略实际用来求解
- 把"正确策略 vs 错误策略"的对比结果（答案正确/错误）作为 ground truth
- 用 SR prompt 模板生成 CoT 解释，说明正确策略好在哪里
- 把这些 CoT 解释 + 正确策略混合进 SFT 数据——**接地的 CoT**，不是凭空生成的

这解决了我们目前可能面临的"STaR 式幻觉"问题：如果直接让模型生成"这条策略为什么好"而不实际验证，会产生虚假推理。**先跑、再反思，基于真实结果解释**。

### 12.2 Implicit World Modeling → 策略应用结果预测

IWM 训练模型预测"执行动作后的状态"，让模型内化环境动力学。

**对我们项目的类比**：
- 可以训练模型预测"用策略 π 解题后的推理轨迹/中间答案"，即策略应用的"结果状态"
- 这相当于让模型内化"不同类型策略通常导向什么样的推理路径"
- 具体实现：给定（题目, 候选策略），预测"应用该策略后第一步推理会是什么"
- 这作为辅助预训练任务，帮助策略提取模型更好地"感知策略与推理过程的关系"

### 12.3 "数据效率：1/8 专家数据足够" → 少量高质量 few-shot 的威力

论文证明了：early experience 让 **1/8 专家数据就能超越全量 IL**。

**对我们项目的直接启示**：
- 我们的 few-shot 示例数量通常很少（3~10 条），这是我们的核心约束
- Early experience 的思路表明：**从少量示例中"探索"出更多接地的决策信号**，比无脑扩充示例更有效
- 具体操作：对每个 few-shot 示例，生成 K 条候选策略，每条实际用来解题，利用正确/错误的对比结果产生接地 SR 数据
- 这将 N 个 few-shot 示例 × K 条策略 = N×K 条接地训练样本——大大扩充了数据量而不需要更多人工标注

### 12.4 OOD 泛化提升 → 策略的跨题型迁移

论文发现 OOD 场景的提升有时 ≥ in-domain，说明接地 CoT 教会了模型**可迁移的决策原则**，而不是特定场景的记忆。

**对我们项目的启示**：
- 策略提取的目标本来就是"提取可迁移的通用原则"，而不是"记住特定题目的解法"
- SR 生成的接地 CoT（"这条策略好是因为它识别了等差数列结构，而替代策略误识别为等比数列"）是对**策略泛化机制的显式建模**
- 这样的训练数据会让模型学到"什么特征触发什么策略"的通用规律，而不只是"遇到这道题用这个公式"

### 12.5 "接地 vs 不接地"（Early Exp vs STaR）→ 策略反思必须基于实际执行

论文最重要的发现之一：未经实际执行验证的 CoT（STaR 风格）不仅无效，**甚至可以有害**（fine-tune 后性能反而下降）。

**对我们项目的警示**：
- 不能简单地让模型生成"这条策略好的理由"然后就用来训练——如果策略本身对某道题是错的，生成的理由是虚假的
- **必须先实际执行策略求解，得到真实对比结果**，再基于此生成反思数据
- 这与 R-Zero 的"majority vote 伪标签"思路相同：没有接地验证的 CoT 是不可信的训练信号

### 12.6 Early Experience as RL Bridge → 策略数据的 RL 预热

论文证明了 Early Experience 作为 RL warm-start 更有效：相同 RL 步数，EE 初始化的最终性能天花板更高。

**对我们项目的直接可用策略**：

```
我们的训练流程（参考 Early Experience + RL pipeline）：

Phase 1: Early Experience（无需外部 reward）
  - 从 few-shot 提取初始策略
  - 生成 K 条候选策略 per 题目
  - 实际用各策略求解，得到对比结果
  - SR 数据：（题目, 候选策略, 接地 CoT, 正确策略）
  - IWM 数据：（题目, 候选策略, 推理结果预测）
  - 混合 SFT 训练

Phase 2: RL（有验证信号时）
  - 用 Phase 1 checkpoint 作为初始化
  - GRPO 等 RL 算法，以最终答案正确性为 reward
  - 比直接 SFT → RL 更高的最终天花板
```

---

### 一句话总结：这篇论文对我们项目的意义

> **Early Experience 的"接地自我反思"机制（基于实际执行结果对比生成 CoT，而非凭空推理）直接对应我们的策略选择训练场景：对每道题生成 K 条候选策略并实际求解，用正确/错误对比结果生成"为什么这条策略更好"的接地 CoT，再混合 SFT——这样的数据比 STaR 式无验证 CoT 更可靠，比纯 IL 更能教会模型可泛化的策略选择原则，而且只需要少量 few-shot 示例就能产生大量接地训练信号。**

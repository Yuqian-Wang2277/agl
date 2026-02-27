# R-Zero：从零数据出发的 LLM 自进化推理框架

> 论文：**R-Zero: Self-Evolving Reasoning LLM from Zero Data**  
> 作者：Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang 等（Tencent AI Seattle Lab / WashU / UMD）  
> 链接：<https://arxiv.org/abs/2508.05004>（arXiv:2508.05004 v4，2026-02-13，**ICLR 2026 正式发表**）  
> 代码：<https://github.com/Chengsong-Huang/R-Zero>

---

## 0. 先说清楚：这篇论文在做什么

这篇论文回答了一个极具挑战性的问题：

> **如果连一条人工标注的训练数据都没有，LLM 能不能自己进化出更强的推理能力？**

R-Zero 的答案是"可以"。它从**一个 base LLM** 出发，不依赖任何外部数据集、不需要人工标注，通过 **Challenger（出题者）和 Solver（解题者）的协同进化**，让模型持续自我提升推理能力。

这直接冲击了当前主流训练范式的核心假设——"RLVR（带可验证奖励的强化学习）需要高质量人工标注数据"。

---

## 1. 摘要提炼

### 1.1 痛点

| 现有方法 | 局限 |
|---------|------|
| RLVR（DeepSeek-R1 等） | 依赖大量人工标注的任务和答案，成本高、难规模化 |
| Label-Free RL（无标注 RL） | 不需要标签，但仍需要**预先存在的任务集**（seed dataset） |
| Self-Challenging 方法 | 自生成任务，但通常依赖**外部代码执行器**验证正确性 |

**R-Zero 的定位**：既不需要 seed dataset，也不需要外部 oracle，是目前最"干净"的零数据自进化方案。

### 1.2 核心贡献

1. **R-Zero 框架**：Challenger + Solver 协同进化，完全自监督，**零外部数据**。
2. **不确定性奖励（Uncertainty Reward）**：Challenger 被激励生成"恰好在 Solver 能力边界上"的问题——即 Solver 正确率约 50% 的题目。
3. **难度自适应课程**：随着 Solver 变强，Challenger 自动生成更难的题目，形成持续学习信号。
4. **实验效果**：Qwen3-4B-Base 数学平均分 +6.49，通用推理 +7.54（完全无外部数据）。

---

## 2. 整体框架：Challenger-Solver 协同进化

R-Zero 的结构非常清晰——两个模型，三个阶段，循环迭代：

```
┌─────────────────────────────────────────────────────────────────┐
│                      一次 R-Zero 迭代                           │
│                                                                 │
│  Phase 1: Challenger 训练（GRPO）                               │
│    Challenger ──生成问题──► Solver（冻结）                      │
│    ◄── 计算 Solver 解题不确定性，作为 Challenger 的 reward ───  │
│                                                                 │
│  Phase 2: Solver 数据集构建                                     │
│    Challenger（冻结）──生成 N=8000 道题──► 过滤（难度适中）     │
│    Solver 多次采样──► Majority Vote 生成伪标签 ỹ               │
│                                                                 │
│  Phase 3: Solver 训练（GRPO）                                   │
│    Solver ──在过滤后的题目上训练──► 更强的 Solver               │
│                                                                 │
│  ↓ 下一轮迭代：更强的 Solver → Challenger 需要出更难的题        │
└─────────────────────────────────────────────────────────────────┘
```

两个关键设计：
- **Challenger 和 Solver 是独立的两个模型**（从同一 base LLM 初始化后各自独立训练）
- 每一轮中，**训练一个时，另一个冻结**——避免相互干扰

---

## 3. 方法详解

### 3.1 Challenger 训练：如何出"恰好合适"的题

Challenger 的目标不是出最难的题，而是出"让 Solver 刚好不确定的题"。

#### 3.1.1 不确定性奖励（核心创新）

对 Challenger 生成的问题 x，让当前 Solver 采样 m=10 个答案：

```
Majority Vote → 伪标签 ỹ(x)
经验正确率 p̂(x; Sϕ) = (1/m) Σ 1{yⱼ = ỹ(x)}

不确定性奖励：r_uncertainty(x; ϕ) = 1 - 2|p̂(x; Sϕ) - 0.5|
```

**直觉理解**：
- p̂ = 1.0（太简单）→ r_uncertainty = 0
- p̂ = 0.0（太难）→ r_uncertainty = 0  
- **p̂ = 0.5（恰好在边界）→ r_uncertainty = 1**（最高奖励）

这是一个 ∧ 形奖励，峰值在 50% 正确率处。理论依据（附录 F）：学习效率与奖励方差正相关，二元奖励下方差在 p̂=0.5 时最大，即此时学习信号最强。

```
r_uncertainty
    1 |         ▲
      |        / \
    0 |───────/   \──────
      0      0.5      1.0    p̂ (Solver 正确率)
```

#### 3.1.2 重复度惩罚（保证问题多样性）

一批问题内如果太相似，Solver 会学到重复模式而非泛化能力。用 BLEU 分数做聚类，对同一簇内的题目施加惩罚：

```
两题 xᵢ, xⱼ 的距离：dᵢⱼ = 1 - BLEU(xᵢ, xⱼ)
用 Average-linkage 聚类，阈值 τ_BLEU = 0.5

惩罚：r_rep(xᵢ) = λ × |C_k| / B
  其中 |C_k| 是 xᵢ 所在簇的大小，B 是 batch size，λ = 1
```

#### 3.1.3 格式检查（过滤无效输出）

要求问题必须用 `<question>...</question>` 标签包裹；格式不合格的直接 reward = 0。

#### 3.1.4 综合奖励

```python
r_final = max(0, r_uncertainty - r_rep)
```

用 GRPO 更新 Challenger 的参数 θ。

---

### 3.2 Solver 数据集构建：难度自适应过滤

Challenger 训练完后（冻结），生成 N=8000 道候选题：

```
对每道题 xₖ：
  Solver 采样 m=10 个答案 → Majority Vote → 伪标签 ỹₖ
  计算经验正确率 p̂ₖ

保留条件：|p̂ₖ - 0.5| ≤ δ（δ=0.25，即 p̂ ∈ [0.25, 0.75]）
  实际为：Solver 答对 3~7 次（共10次）的题保留
```

**双重作用**：
1. **难度控制**：排除太简单（p̂ > 0.75）或太难（p̂ < 0.25）的题
2. **隐式质量过滤**：极低 p̂ 往往意味着题目本身有歧义或伪标签不可靠，过滤掉这些题

---

### 3.3 Solver 训练：简单可验证的 reward

Solver 在过滤后的数据集 S 上，用 GRPO 训练，reward 非常简单：

```
对问题 xᵢ ∈ S，其伪标签为 ỹᵢ：

rⱼ = 1  如果 Solver 第 j 次生成的答案 yⱼ 匹配 ỹᵢ
rⱼ = 0  否则
```

用这个二元 reward 计算 GRPO 的 advantage，更新 Solver 参数 φ。

---

## 4. Prompt 设计

### 4.1 Solver Prompt（极简）

```
[System]
Please reason step by step, and put your final answer within \boxed{}.

[User]
{problem statement}
```

设计原则：简洁，强制 chain-of-thought 格式（step by step），最终答案放进 `\boxed{}`，便于自动提取和对比。

### 4.2 Challenger Prompt（精心设计）

```
[System]
You are an expert competition-math problem setter. FIRST, in your private 
scratch-pad, think step-by-step to design a brand-new, non-trivial problem. 
The problem could come from any field of mathematics, including but not limited 
to algebra, geometry, number theory, combinatorics, prealgebra, probability, 
statistics, and calculus. Aim for a difficulty such that fewer than 30% of 
advanced high-school students could solve it. Avoid re-using textbook 
clichés or famous contest problems.

THEN, without revealing any of your private thoughts, output exactly 
the following two blocks:
<question>
{The full problem statement on one or more lines}
</question>
\boxed{final answer}

Do NOT output anything else—no explanations, no extra markup.

[User]
Generate one new, challenging reasoning question now. Remember to format 
the output exactly as instructed.
```

**Prompt 设计要点**：

| 设计 | 目的 |
|------|------|
| "fewer than 30% of advanced high-school students" | 引导生成有难度的题，而非简单题 |
| "private scratch-pad, think step-by-step" | 让 Challenger 先思考再出题，提高质量 |
| "without revealing any of your private thoughts" | 只输出题目和答案，不暴露推理过程（格式干净）|
| 严格的 `<question>` 标签约束 | 配合格式检查 reward，过滤无效输出 |
| "Avoid re-using textbook clichés" | 鼓励出新题，配合重复惩罚提升多样性 |

### 4.3 GPT-4o 评判 Prompt（评测用）

用于验证模型答案是否与标准答案等价（处理数学表达式等价性问题）：

```
[System] You are a math answer checker.

[User]
Hi, there is an answer: {answer},
and the ground truth answer is: {response},
please check whether the answer is correct or not, 
and return the **only** Yes or No.
```

---

## 5. Reward 设计总结

R-Zero 有两套截然不同的 reward 体系，这是全文最核心的工程设计：

### Challenger 的 Reward（复合，鼓励出"边界难度"的多样题）

```
r_Challenger = max(0, r_uncertainty - r_rep)

r_uncertainty = 1 - 2|p̂(x; Solver) - 0.5|    ← 激励 Solver 处于最大不确定性
r_rep         = |cluster_size| / batch_size     ← 惩罚批内重复
```

还有一个门控：格式不合格 → r = 0，直接跳过。

### Solver 的 Reward（简单二元，直接验证答案）

```
r_Solver = 1  如果答案与伪标签一致
r_Solver = 0  否则
```

**两套 reward 的哲学差异**：
- Challenger 的 reward 是间接的、相对的——通过衡量 Solver 的困难程度来评价问题质量
- Solver 的 reward 是直接的、绝对的——对就是对，错就是错

这种设计避免了任何外部评判器（judge LLM），完全自给自足。

---

## 6. 评测指标（Evaluation Metrics）

### 6.1 数学推理基准（7 个）

| 基准 | 难度 | 评测方式 |
|------|------|---------|
| GSM8K | 基础数学 | 精确匹配（greedy decoding） |
| MATH-500 | 竞赛数学 | GPT-4o 判断等价性 |
| Minerva | 大学数学 | GPT-4o 判断等价性 |
| OlympiadBench | 奥林匹克级 | GPT-4o 判断等价性 |
| AMC | 竞赛选拔 | mean@32（32次采样取均值）|
| AIME-2024 | 顶级竞赛 | mean@32 |
| AIME-2025 | 顶级竞赛 | mean@32 |

对 AMC 和 AIME 使用 **mean@32**（32次采样后取均值）而非 pass@1，因为这类题目通过率很低，单次解码方差大。

### 6.2 通用推理基准（3 个）

| 基准 | 内容 | 意义 |
|------|------|------|
| MMLU-Pro | 多领域学术多选题 | 通用知识 + 推理能力 |
| SuperGPQA | 285 个研究生学科问题（不可搜索） | 真实推理（排除记忆） |
| BBEH（Big-Bench Extra Hard） | BIG-Bench 更难版 | 复杂推理能力 |

设计意图：通过数学训练能否提升**非数学领域**的推理能力（transfer learning 检验）。

---

## 7. 核心实验结果

### 7.1 数学推理（Table 1）

| 模型 | Base | Absolute Zero（竞品） | R-Zero（本文） | 提升 |
|------|------|--------------------|--------------|------|
| Qwen3-4B-Base | 42.57 | 46.42 | **49.93** | **+7.36** |
| Qwen3-8B-Base | 48.64 | 52.68 | **53.72** | **+5.08** |
| OctoThinker-3B | 26.64 | 27.23 | **29.32** | **+2.68** |
| OctoThinker-8B | 36.41 | 36.60 | **38.52** | **+2.11** |

**关键对比**：`R-Zero(∅ challenger)` 是用未经训练的 Challenger 出题的消融组——效果明显差于正式 R-Zero，证明 **Challenger 的 RL 训练是关键**，不是随便生成题目就能提升的。

### 7.2 通用推理（Table 2）

| 模型 | Base | Absolute Zero | R-Zero | 提升 |
|------|------|--------------|--------|------|
| Qwen3-4B-Base | 26.34 | 29.33 | **31.15** | **+4.81** |
| Qwen3-8B-Base | 31.98 | 34.40 | **34.50** | **+2.52** |

数学题训练 → 通用推理提升，**跨领域迁移验证成功**。

### 7.3 与人工标注数据的协同效果（图 4）

分三种策略：
1. **人工数据 only**（baseline）：57.03（SuperGPQA for 4B）
2. **R-Zero only**：与人工数据接近
3. **R-Zero 先训，再 SFT 人工数据（Sequential）**：**+2.35 提升**（最优）
4. **混合训练（Concurrent）**：优于单独使用任一，但不如 Sequential

**结论**：R-Zero 是最佳 mid-training 方法——先自进化获得推理基础，再用人工数据精调，效果远优于直接 SFT。

---

## 8. 深度分析

### 8.1 迭代缩放（Iteration Scaling）：为何最终会崩溃？

```
Qwen3-4B-Base 各 step 平均分：
  Base    Step 15   Step 30   Step 45   Step 60
  42.58   48.06     48.44     49.07     46.52 ←下降
```

**规律**：
- 所有规模的模型最终都会崩溃（性能下降）
- **模型越大，崩溃越晚**（0.6B 在 Step 15 就崩，4B 撑到 Step 60）
- 原因不只是伪标签质量下降（数据噪声），更根本的是**模型崩塌（model collapse）**——长期在自生成数据上训练导致多样性丧失、偏差放大

### 8.2 问题难度的演化（Table 5）

| 题目集 | Base 分数 | Solver(step15) 分数 | 伪标签准确率 |
|--------|----------|-------------------|------------|
| D_step15 | 48.0% | 59.0% | **79.0%** |
| D_step30 | 52.5% | 53.0% | **69.0%** |
| D_step45 | 44.0% | 47.0% | **63.0%** |

两个关键趋势：
1. **Challenger 确实在持续出更难的题**（D_step45 对所有模型都更难）
2. **伪标签准确率在下降**（79% → 63%）——题越难，多数投票越不可靠

### 8.3 双模型 vs 单模型（Table 6）

将 Challenger 和 Solver 合并为同一模型（Single-R-Zero）的消融：

| | R-Zero（双模型）| Single-R-Zero（单模型）|
|--|--------------|---------------------|
| Step 15 | 48.06 / 71.0% | 47.31 / **63.4%** |
| Step 45 | **49.12** / 48.8% | 45.57 / 32.6% |

**单模型问题**：同一模型出题和答题，会产生"内部偏见"——它出的题本质上更容易被自己解答（overconfidence），伪标签质量更差，崩溃更早。**Challenger 和 Solver 必须分离。**

---

## 9. 算法伪代码

```python
# R-Zero 完整流程（Algorithm 1）

for iteration in range(num_iterations):
    # Phase 1: Challenger 训练
    for step in range(challenger_steps):
        questions = Challenger.sample(G)              # 生成 G 道题
        for q in questions:
            if not format_check(q):
                r = 0; continue
            answers = Solver.sample(q, m=10)          # Solver 采样10次（冻结）
            pseudo_label = majority_vote(answers)
            p_hat = accuracy(answers, pseudo_label)
            r_uncertainty = 1 - 2 * abs(p_hat - 0.5)
            r_rep = cluster_size(q, batch) / batch_size
            r = max(0, r_uncertainty - r_rep)
        Challenger.update_GRPO(rewards)
    
    # Phase 2: 数据集构建
    S = []
    candidates = Challenger.sample(N=8000)            # 冻结 Challenger
    for q in candidates:
        answers = Solver.sample(q, m=10)              # 冻结 Solver
        pseudo_label = majority_vote(answers)
        p_hat = accuracy(answers, pseudo_label)
        if abs(p_hat - 0.5) <= delta:                 # 难度过滤
            S.append((q, pseudo_label))
    
    # Phase 3: Solver 训练
    for (q, pseudo_label) in S:
        answers = Solver.sample(q, G)
        rewards = [1 if a == pseudo_label else 0 for a in answers]
        Solver.update_GRPO(rewards)
```

---

## 10. 关键超参数汇总

| 参数 | 值 | 说明 |
|------|----|------|
| N（候选题数） | 8,000 | 每轮 Challenger 生成的题目数 |
| m（Solver 采样次数） | 10 | 计算 p̂ 和 majority vote |
| δ（过滤阈值） | 0.25 | 保留 p̂ ∈ [0.25, 0.75] |
| τ_BLEU（聚类阈值） | 0.5 | 重复惩罚的相似性阈值 |
| λ_KL | 1e-2 | GRPO 的 KL 惩罚系数 |
| Challenger max steps | 5 | 每轮 Challenger 训练步数 |
| Solver max steps | 15 | 每轮 Solver 训练步数 |
| Learning Rate | 1e-6 | 两者相同 |

---

## 11. 局限性

| 局限 | 说明 |
|------|------|
| **只适合可验证域** | Majority Vote 依赖答案的客观可比性；创意写作、对话生成等主观任务无法适用 |
| **最终迭代崩溃** | 所有规模模型均出现性能退化；Model Collapse 是根本原因之一，还未解决 |
| **伪标签噪声积累** | 随迭代加深，题目越来越难，多数投票越来越不准，数据质量下滑 |
| **目前聚焦数学** | 虽有 "no math prompt" 实验，但主体框架仍以数学为主要验证场景 |

---

## 12. 结论总结

| 维度 | 核心结论 |
|------|---------|
| **最核心命题** | 无任何外部数据，LLM 可以通过 Challenger-Solver 协同进化自我提升推理能力 |
| **关键设计** | 不确定性奖励（p̂≈0.5 最优）+ 难度过滤（保留边界难度题）+ 双模型分离 |
| **数据来源** | 完全自生成，伪标签来自 majority vote，无人工标注 |
| **效果** | 数学 +6.49、通用推理 +7.54（Qwen3-4B），且与 SFT 协同时效果更强 |
| **最优使用方式** | R-Zero 先训（mid-training）→ 再 SFT 人工数据（sequential 策略最优）|
| **已知瓶颈** | 迭代收敛后崩溃，根本原因尚未完全解明 |

---

## 13. 对我们项目（Agent-Lightning 策略提取）的启发

我们的项目思路是：**从 few-shot 示例中提取解题策略，再将策略应用于新问题求解**。R-Zero 的框架设计与我们面临的若干核心问题有深度共鸣，以下逐点分析。

### 13.1 "边界难度"原则：策略应在 Solver 能力边界处提取

R-Zero 最核心的发现是：**学习效率在 Solver 正确率约 50% 时最大**（不确定性奖励峰值）。这背后有严格的理论支撑（KL 散度下界与奖励方差正相关）。

**对我们项目的启示**：
- 用于策略提取的 few-shot 示例，不应该全是"模型轻松能解的题"（太简单，策略提取训练无效），也不应是"模型完全不会的题"（太难，策略应用无法验证）
- 可以在数据筛选阶段引入类似的**难度过滤**：保留"模型用基础方法正确率在 25~75% 之间"的题目作为策略提取训练数据
- 这直接对应 R-Zero 的 `|p̂ - 0.5| ≤ δ` 过滤条件

### 13.2 Majority Vote 作为免标注伪标签：策略质量的无监督评估

R-Zero 用多次采样的多数投票代替人工标注作为训练标签。我们的策略提取系统同样面临标签稀缺问题：**如何判断提取出的策略是否"好"？**

**可借鉴的做法**：
- 用提取到的策略 π，让 Solver 对同一道题独立求解 m 次（m=5~10）
- **策略一致性**：m 次解答中有多少次能通过策略 π 推导到相同答案？一致性高的策略更可靠
- **策略正确率**：m 次解答的答案多数投票结果与真实答案是否一致？

这等价于 R-Zero 的 p̂ 估计，可作为策略质量的代理指标，**无需额外人工标注**。

### 13.3 Challenger-Solver 对应到我们的"策略生成-策略应用"双角色

R-Zero 的双模型设计本质上是"分离问题生成能力和问题求解能力"。类比到我们的项目：

```
R-Zero:          Challenger（出题） ←→ Solver（解题）
                       ↕ 协同进化

我们的项目:     策略提取模型（Strategy Extractor）←→ 解题模型（Solver）
                       ↕ 互相提升
```

**关键启示**：
- 策略提取模型和解题模型应该**分离训练**（就像 R-Zero 禁止单模型同时出题和解题），否则"自我迎合"问题会导致提取的策略只对提取者自己有效
- 可以考虑用 A 模型提取策略，让 B 模型应用策略解题，用 B 的答对率来反向评估 A 的策略质量——形成类似 Challenger-Solver 的互相约束机制

### 13.4 重复惩罚 → 策略多样性保证

R-Zero 用 BLEU 相似度惩罚来确保 Challenger 在同一 batch 内出多样的题，防止策略空间坍缩到某类问题。

**对我们项目的类比**：
- 从 few-shot 示例提取策略时，应主动检测并惩罚"策略雷同"的情况（可以用嵌入相似度替代 BLEU）
- 训练 batch 内的策略覆盖度越高，Solver 的泛化能力越强
- 可设计类似的**策略多样性奖励**，鼓励模型提取不同视角、不同抽象层次的策略

### 13.5 Sequential 训练策略（R-Zero 先，再 SFT）的强力启示

R-Zero 实验明确证明：**先无监督自进化（R-Zero），再有监督精调（SFT on labeled data）**，效果优于任何单独使用或混合训练。

**直接可借鉴**：
- 我们的训练流程也可以参考 Sequential 策略：
  1. **Phase 1**：用 R-Zero 类似的无监督/弱监督自进化预热，让模型先形成基本的推理能力
  2. **Phase 2**：再用策略提取数据（有质量监督的）进行精调
  3. 这两个阶段的顺序很重要——先让模型"学会推理"，再"学会用策略推理"

### 13.6 迭代崩溃的警示

R-Zero 发现多轮迭代之后性能会下降，根本原因之一是**在自生成数据上反复训练导致 model collapse**。

**对我们项目的警示**：
- 策略提取 → 策略应用 → 策略再提取的自迭代循环，同样有崩溃风险
- 需要引入外部锚点（如人工标注的高质量示例、定期注入新的 few-shot examples）来打破"自我强化偏差"
- 类比 R-Zero 的发现：**大模型更抗崩溃**，应优先在较大的 base model 上测试迭代稳定性

---

### 一句话总结：这篇论文对我们项目的意义

> **R-Zero 的"Challenger 以 Solver 不确定性为 reward 出题 + Majority Vote 伪标签 + 难度过滤"的三件套，可以直接迁移为我们项目的训练数据质量控制方案：用策略应用成功率的一致性估计策略质量，用难度过滤筛选"恰好合适"的训练题目，并严格分离策略提取模型与解题模型，防止自我迎合导致策略退化。**

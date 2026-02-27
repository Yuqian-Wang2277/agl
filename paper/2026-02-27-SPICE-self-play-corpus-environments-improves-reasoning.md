# SPICE：让模型在语料库环境中自博弈以提升推理能力

> **论文**：[SPICE: Self-Play In Corpus Environments Improves Reasoning](https://arxiv.org/abs/2510.24684)  
> **机构**：FAIR at Meta & National University of Singapore  
> **时间**：2025 年 10 月 28 日  
> **核心贡献**：提出 SPICE 框架，通过将大规模文档语料库作为外部环境，让单一 LLM 同时扮演"出题者（Challenger）"和"解题者（Reasoner）"两个角色进行对抗式自博弈 RL 训练，在数学推理（+8.9%）和通用推理（+9.8%）基准上显著超越现有无外部接地的自博弈方法。

---

## 一、摘要提炼：自博弈的痛点与 SPICE 的答案

### 为什么现有自博弈方法会"撞墙"？

自博弈（Self-Play）让模型与自身对弈以持续进步，是一个令人期待的自我改进范式。然而，现有面向 LLM 的自博弈方法往往在初期提升后快速陷入瓶颈，原因在于两个根本性缺陷：

| 问题 | 描述 |
|------|------|
| **幻觉放大（Hallucination Amplification）** | 模型自己生成问题和答案，无法验证事实正确性，错误随训练迭代不断累积 |
| **信息对称（Information Symmetry）** | 出题者和解题者共享同一个知识库，无法产生真正的"意外挑战"，导致问题趋于简单重复 |

例如 R-Zero 方法（本文的重要基线）在 4 次迭代后就出现性能退化，伪标签准确率从 79% 下滑到 63%。

### SPICE 的核心思路

SPICE（Self-Play In Corpus Environments）的解决方案一句话可概括：**将大规模真实文档语料库作为外部环境，引入信息不对称，让出题者从文档中挖掘题目，解题者在不看文档的情况下回答**。

```
             ┌─────────────────────────────────────────┐
             │           同一个 LLM：πθ                   │
             │                                           │
             │  ┌──────────────┐    ┌──────────────┐    │
             │  │  Challenger  │    │   Reasoner   │    │
             │  │  (有文档访问) │    │ (无文档访问)  │    │
             │  └──────┬───────┘    └──────┬───────┘    │
             └─────────┼─────────────────────┼───────────┘
                       │                     │
                 从语料库 D           解题，仅依赖
                 挖掘文档 d，          内化知识，
                 生成 (q, a*)          输出答案 â
                       │                     │
                       └──────── 对抗训练 ────┘
                         Challenger: 奖励 → 难度适中的题
                         Reasoner:   奖励 → 答对得分
```

两个关键优势：
1. **防止幻觉**：答案 `a*` 直接从真实文档提取，事实准确性有保障
2. **真实信息不对称**：Challenger 看到文档，Reasoner 看不到，产生真正的挑战

---

## 二、工具与基础设施

| 组件 | 说明 |
|------|------|
| **训练框架** | [Oat](https://github.com/sail-sg/oat)，提供分布式 Actor-Learner 架构 |
| **推理引擎** | vLLM，高效并发采样 |
| **答案验证** | [Math-Verify](https://github.com/huggingface/Math-Verify)，支持数学表达式等价检查、精确匹配 |
| **评估框架** | [simple-evals](https://github.com/openai/simple-evals) + GPT-4o 等价性判断 |
| **优化算法** | DrGRPO（去除标准差归一化的改进版 GRPO） |
| **计算资源** | 8 × H200 GPUs，学习率 1e-6，ZeRO Stage 2 |

**训练语料库**（共 20,000 文档，各 50%）：
- **Nemotron-CC-Math**：1330 亿 token 级别的高质量数学预训练数据集
- **NaturalReasoning**（来自 DCLM 子集）：覆盖 STEM、人文、社科的广泛推理文档

---

## 三、Agent 设计：双角色单模型架构

SPICE 最独特的设计是**单一模型身兼两职**，通过 `role` 参数切换：

### 3.1 Challenger（出题者）

**目标**：生成对 Reasoner 刚好处于能力边界的挑战性题目。

**工作流程**：

```
文档 d ∼ D
    ↓
① 格式选择：判断该文档适合 MCQ 还是 Free-form
    ↓
② 题目生成：根据格式选择对应 Prompt，生成 (q, a*)
   - 最多尝试 N=1024 次，取至少一道有效题
    ↓
③ 有效性验证：格式正确且可解析 → 有效题
    ↓
④ 计算 Challenger 奖励：从 Reasoner 采样 K=8 个答案，计算方差奖励
```

**题目格式**：
- **MCQ（多选题）**：4 个选项，含 3 个精心设计的干扰项，正确答案来自文档
- **Free-form（自由回答）**：整数（Integer）、数学表达式（Expression）、字符串（String）类型

### 3.2 Reasoner（解题者）

**目标**：仅凭内化知识解答问题，不看原始文档。

**工作流程**：

```
仅接收问题 q（无文档）
    ↓
逐步推理（step-by-step CoT）
    ↓
将最终答案放入 \boxed{} 标签
    ↓
与 a* 比对 → 二值化奖励
```

### 3.3 共权重更新

关键设计：**Challenger 和 Reasoner 共享同一套模型权重 `πθ`**，每次迭代交替：先以 Challenger 身份生成题目，再以 Reasoner 身份解题，然后用各自的 role-specific advantage 统一更新 πθ。

---

## 四、数据合成方法

SPICE 的数据合成是**完全在线（online）**的，无需离线预构建数据集：

```
每个训练迭代：
  for 每批 B=128 个文档:
    Challenger 生成题目 (q, a*)
    Reasoner 生成 G=8 个答案 {â_1, ..., â_8}
    计算 Challenger 方差奖励 r_C
    随机选一道有效题用于 Reasoner 训练
    计算 Reasoner 正确率奖励 r_R
  ↓
  用 DrGRPO 更新共享权重 πθ
```

**核心数据合成设计原则**：
1. **无预定义标注**：原始文档不含任何预设问题或标签，完全从非结构化文本中涌现
2. **格式多样性**：MCQ + Free-form 覆盖不同推理模式，扩展到代码以外的所有领域
3. **信息不对称**：答案 a* 从 Challenger 看到的文档提取，而 Reasoner 无法访问

---

## 五、算法创新

### 5.1 Prompt 设计

SPICE 的 Prompt 分为三类，设计极为精细：

#### ① 题目格式选择 Prompt（Task Type Selection）

```
Analyze this document and decide whether it's better suited 
for a CHALLENGING multiple-choice question (MCQ) or a free-form question.

Document: {document}

For MCQ:
- Needs complex relationships and multi-step reasoning paths
- Should allow creating 3 plausible but wrong distractors
- Requires synthesis of multiple concepts

For Free-form:
- Best for questions requiring specific calculations (Integer)
- Good for deriving formulas (Expression)
- Suitable for conceptual answers (String)

You MUST respond with ONLY a valid JSON object:
{
  "suitable_for_mcq": <true/false>,
  "suitable_for_free_form": <true/false>,
  "best_answer_type": <"Integer"/"Expression"/"String"/null>,
  "reason": "<under 100 characters>"
}
```

#### ② MCQ 生成 Prompt（8 步结构化流程）

这是 SPICE 中最复杂的 Prompt，包含 8 个明确步骤：

| 步骤 | 内容 |
|------|------|
| Step 1 | **复杂信息提取**：聚焦需要跨章节综合的多概念关系，避免单一直接事实 |
| Step 2 | **难度增强策略**：显式陈述难化过程（避免什么简单版本、添加哪些复杂层次） |
| Step 3 | **高级题目生成**：要求多概念连接、多步推理，禁止引用"文档中提到..." |
| Step 4 | **难度目标设定**：HARD（综合4+概念）或 EXTRA HARD（复杂系统分析） |
| Step 5 | **知识整合路径**：列出所需3+条信息及逻辑连接，解释为何简单查找无效 |
| Step 6 | **MCQ 设计规范**：选项长度均衡、单位一致、干扰项基于半知识推导 |
| Step 7 | **自测过滤**：Challenger 以学生身份解自己的题，验证是否真的需要多步推理 |
| Step 8 | **最终复杂度验证**：不可一句话回答、必须连接多文档章节、需要理解关系而非记忆 |

#### ③ Free-form 生成 Prompt（类似 8 步结构）

特别针对不同答案类型有差异化要求：
- **Integer/Float**：多变量计算、序列计算
- **Expression**：多变量关系、可泛化公式
- **String**：1-3 词最大长度，精确术语、命名实体

#### ④ Reasoner Prompt（模型族特定）

```
# Qwen3 家族
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{question}
Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant

# OctoThinker 家族
A conversation between User and Assistant. The user asks a question, 
and the Assistant solves it. The assistant first thinks about the reasoning 
process in the mind and then provides the user with the answer.
User: {question}
You must put your answer inside \boxed{}.
Assistant:
```

---

### 5.2 Reward 设计

SPICE 的奖励设计是本文最精妙的部分，Challenger 和 Reasoner 有完全不同的奖励逻辑：

#### Challenger 奖励：方差基准的高斯形状奖励

$$r_C(q, a^*) = \begin{cases} \exp\!\left(-\dfrac{(\text{Var}(\{l_1,\ldots,l_K\}) - 0.25)^2}{2 \times 0.01}\right) & \text{若 } q \text{ 有效} \\ \rho = -0.1 & \text{若 } q \text{ 无效（格式错误）} \end{cases}$$

其中 $l_i = \mathbf{1}[\hat{a}_i = a^*]$，即第 $i$ 次 Reasoner 回答是否正确。

**关键设计意图**：

| Reasoner 通过率 p | 奖励含义 |
|---|---|
| p ≈ 0%（太难）| 奖励趋近 0，Challenger 不鼓励出不可解题 |
| p ≈ 50%（边界）| 奖励最大（= 1.0），这是最有学习价值的题 |
| p ≈ 100%（太简单）| 奖励趋近 0，Challenger 不鼓励出送分题 |

这一设计创造了**自动课程**：随着 Reasoner 能力提升，Challenger 被激励出更难的题以维持 ~50% 通过率。

**与其他奖励策略对比**：

```
Reasoner Pass Rate p
0        0.25        0.5       0.75       1.0
|---------|-----------|---------|-----------|
                      ↑
                  最优难度

Absolute Zero: r = 1 - p     (越难越好，忽视不可解题)
Threshold:     r = 1 若 0<成功数<8，否则 0  (二值信号)
R-Zero:        r = 1 - 2|p-0.5|  (在 p=0.5 达峰，三角形)
SPICE Variance: r = exp(-((p(1-p)-0.25)²)/0.02)  (高斯形，无间断点)
```

#### Reasoner 奖励：二值正确性奖励

$$r_R(\hat{a}, a^*) = \mathbf{1}[\hat{a} = a^*]$$

基于规则的验证器（Math-Verify），支持数学表达式等价、精确字符串匹配等。

#### Role-Specific Advantage（DrGRPO）

两个角色各自计算 advantage，不混用：

$$\hat{A}_i^C = r_i^C - \text{mean}(\{r_j^C\}_j)$$
$$\hat{A}_i^R = r_i^R - \text{mean}(\{r_j^R\}_j)$$

**去除标准差归一化**：避免因题目难度差异导致的梯度噪声——难题本身方差大，标准化会压制真实学习信号。

联合优化目标：

$$J(\theta) = \mathbb{E}_{d \sim D}\mathbb{E}_{(q,a^*) \sim \pi_\theta(\cdot|d,C)}\big[r_C(q,a^*)\big] + \mathbb{E}_{\hat{a} \sim \pi_\theta(\cdot|q,R)}\big[r_R(\hat{a},a^*)\big]$$

---

### 5.3 Evaluation Metric 设计

#### 数学推理基准

| 基准 | 特点 | 评估协议 |
|------|------|----------|
| MATH-500 | 竞赛数学精选 500 题 | pass@1，贪心解码 |
| AMC | 美国数学竞赛 | pass@1，贪心解码 |
| AIME'24/25 | 美国数学邀请赛 | 32 次采样平均（温度 0.6） |
| OlympiadBench | 奥林匹克级别 | pass@1，贪心解码 |
| Minerva Math | STEM 推理 | pass@1，贪心解码 |
| GSM8K | 小学数学应用题 | pass@1，贪心解码 |

#### 通用推理基准

| 基准 | 特点 |
|------|------|
| GPQA-Diamond | 研究生级别科学题，设计为抵抗模式匹配 |
| SuperGPQA | 285 个学科的研究生推理，抗谷歌搜索 |
| MMLU-Pro | 增强版 MMLU，需要深度理解 |
| BBEH | BigBench Hard 的扩展版，更复杂推理任务 |

#### 答案验证

- 数学：GPT-4o 通过 simple-evals 框架判断等价性（处理分数、小数、代数表达式等格式差异）
- MCQ：精确匹配提取的选项字母（A/B/C/D）
- 全程零样本（zero-shot）评估，使用与训练一致的提示格式

---

## 六、SPICE 完整训练算法

```python
# Algorithm 1: SPICE Self-Play Training
# πθ: 预训练 LLM
# D: 语料库（20k 文档）
# G=8: 每题采样数, B=128: batch size, T=640: 迭代次数, ρ=-0.1: 无效惩罚

for t in range(1, T+1):
    # === Challenger Role ===
    challenger_trajectories = []
    for b in range(B):
        d = sample(D)  # 从语料库随机采样文档
        
        # 最多 1024 次尝试，生成题目
        attempts = πθ(d, role="Challenger")  # 返回 [(q_i, a*_i)]
        
        # 子采样 G 条轨迹，保持 valid:invalid 比例
        T_C = subsample(attempts, G)
        
        for (q_i, a*_i) in T_C:
            if is_valid(q_i):
                # Reasoner 采样 G 个答案，计算方差奖励
                answers = [πθ(q_i, role="Reasoner") for _ in range(G)]
                r_C = gaussian_variance_reward(answers, a*_i)  # 式(1)
            else:
                r_C = ρ  # -0.1 惩罚
            challenger_trajectories.append((q_i, r_C))
    
    # === Reasoner Role ===
    # 随机选一道有效题用于 Reasoner 训练
    valid_q, valid_a* = random_valid_task(challenger_trajectories)
    reasoner_answers = [πθ(valid_q, role="Reasoner") for _ in range(G)]
    reasoner_rewards = [1 if ans == valid_a* else 0 for ans in reasoner_answers]
    
    # === 更新 Phase（DrGRPO）===
    A_C = [r - mean(challenger_rewards) for r in challenger_rewards]  # Challenger advantages
    A_R = [r - mean(reasoner_rewards) for r in reasoner_rewards]     # Reasoner advantages
    
    # 用策略梯度统一更新 πθ
    update(πθ, A_C, A_R)
```

---

## 七、实验结果

### 7.1 主要结果（四个 Base 模型）

SPICE 在所有模型家族上均取得最佳综合表现：

| 模型 | Base | R-Zero | Absolute Zero | Strong Challenger | **SPICE** |
|------|------|--------|---------------|-------------------|-----------|
| Qwen3-4B-Base | 35.8 | 39.5 | 40.7 | 43.0 | **44.9** (+9.1) |
| Qwen3-8B-Base | 43.0 | 46.3 | 46.5 | 45.6 | **48.7** (+5.7) |
| OctoThinker-3B | 14.7 | 20.3 | 21.7 | 21.0 | **25.2** (+10.5) |
| OctoThinker-8B | 20.5 | 29.9 | 29.4 | 28.2 | **32.4** (+11.9) |

*注：Strong Challenger 使用固定的 Qwen3-32B-Instruct 出题，非自博弈方法*

**值得注意的发现**：即便使用 Qwen3-32B-Instruct 这样的强模型作为固定出题者（Strong Challenger），SPICE 的自演化 Challenger 在 Qwen3-4B-Base 上仍能超越它（44.9 vs 43.0），证明了协同进化的价值。

### 7.2 对抗学习动态

```
Reasoner Pass Rate (%)
90 |                                   ●●●●
   |                              ●●●●
80 |                         ●●●●
   |  ✕✕✕                   
70 |       ✕✕✕               固定 Challenger
   |             ✕✕✕         下的 Reasoner
60 |                   ✕✕✕   进步（55%→85%）
   |
50 |  ————————————————
   |
40 |       ●●●
   |  ●●●       ●●●   ●●●   
35 |                          固定 Reasoner
   |                          下 Challenger
   |                          出更难题（55%→35%）
   +——————————————————————————→
   200  310  420  530  640   Training Step
```

两个子图清晰展示了**协同进化**：
- Challenger 进化：在固定 Reasoner 的测评下，通过率从 55% 降到 35%（题越来越难）
- Reasoner 进化：在固定 Challenger 的测评下，通过率从 55% 增到 85%（能力越来越强）

### 7.3 核心消融实验

#### ① 语料库组成对比

| 语料库 | 数学平均 | 通用推理 | 综合 |
|--------|----------|----------|------|
| 仅 NaturalReasoning | 44.4 | 37.0 | 41.7 |
| 仅 Nemotron-CC-Math | 53.4 | 29.8 | 43.2 |
| **两者结合（SPICE）** | **50.6** | **35.0** | **44.9** |

**洞察**：领域专用语料提升对应领域性能，混合使用产生协同效应，整体最优。

#### ② 题目类型对比

| 题型 | 数学平均 | 通用推理 | 综合 |
|------|----------|----------|------|
| 仅 MCQ | 46.9 | 35.7 | 42.0 |
| 仅 Free-form | 52.5 | 31.8 | 43.7 |
| **MCQ + Free-form（SPICE）** | **50.6** | **35.0** | **44.9** |

**洞察**：MCQ 提供可靠验证（有干扰项），Free-form 鼓励灵活推理，混合达到最优均衡。

#### ③ Challenger 奖励策略对比

| 奖励策略 | 数学 | 通用 | 综合 | 形状 |
|----------|------|------|------|------|
| Absolute Zero (1-p) | 48.2 | 30.8 | 40.7 | 线性递减 |
| Threshold (binary) | 48.6 | 31.6 | 41.4 | 阶跃函数 |
| R-Zero (1-2\|p-0.5\|) | 50.0 | 33.9 | 43.6 | 三角形 |
| **Variance/SPICE（高斯）** | **50.6** | **35.0** | **44.9** | **平滑高斯** |

**洞察**：高斯方差奖励提供更好的梯度信号（无不连续点），且综合捕捉答案分布的全貌（而不仅是最高频答案）。

---

## 八、定性分析：Challenger 和 Reasoner 的进化轨迹

### Challenger 的任务演化（同一文档，不同训练阶段）

```
【早期训练 Step 50 - 表面信息提取】
文档：太阳直径 1,391,000 km，月球直径 3,475 km...
生成题目：月球直径是多少？
  A) 1,391 km  B) 3,475 km  C) 34,750 km  D) 347,500 km
  → 直接查找，零推理

【晚期训练 Step 480 - 多步推理题】
文档：（同上）
生成题目：一个外星文明的月球直径 3,475 km，创造完美日食。
          其恒星与太阳同等大小。若月球轨道半径 374,000 km
         （保持与地球相同的角尺寸比），其行星与恒星的距离是多少？
  A) 374,000,000 km  B) 149,600,000 km  ...
  → 需要理解角尺寸关系、设立比例方程、代入计算
```

### Reasoner 的推理模式演化（解答同一道题）

**早期（Step 初期）**：
> "完美日食意味着月亮和恒星看起来一样大。月球在 374,000 km。恒星比月球远得多，也许 1000 倍？那就是 374,000,000 km。答案是 A。"

**晚期（Step 后期）**：
> "## Step 1：识别已知信息 ...
> ## Step 2：理解完美日食条件（角尺寸相等）
> ## Step 3：建立方程 3475/374000 = 1391000/d
> ## Step 4：求解 d = 149,708,489 km
> ## Step 5：验证角尺寸匹配
> 答案是 B：149,600,000 km"

---

## 九、与相关工作的对比定位

```
自我改进方法体系
├── 无外部接地
│   ├── R-Zero：纯自博弈，快速退化
│   ├── Absolute Zero：限定代码领域（Python执行器验证）
│   └── Language Self-Play：数据无关训练，受限于预训练分布
├── 静态数据集
│   ├── MetaMath：bootstrapping，受初始覆盖限制
│   ├── WebInstruct：离线挖掘，静态固定
│   └── NaturalReasoning：离线大规模 QA，不自适应
└── 语料库在线自博弈
    └── SPICE ← 本文（在线、自适应、无幻觉、跨域通用）
```

SPICE 的核心区别：**在线对抗生成**，Challenger 持续从语料库挖掘新文档，生成针对 Reasoner 当前能力校准的题目，避免覆盖限制和质量退化。

---

## 十、结论

SPICE 的核心贡献可以归纳为三点：

1. **诊断了自博弈的根本性缺陷**：幻觉放大 + 信息对称 = 自博弈必然撞墙的两大根源

2. **提出了语料库接地的自博弈框架**：单模型双角色（Challenger/Reasoner）+ 文档信息不对称 + 多格式通用验证

3. **实现了跨域持续自我改进**：不再局限于数学/代码，任何有文档语料库的领域均可应用

**核心结论**：只要有足够丰富的文档语料，LLM 就可以通过与语料库的互动，不断为自己出题、解题，实现真正意义上的"无人类监督的持续自我改进"。

**局限性**：
- 目前语料库采用固定的 20k 文档，更大语料库的收益尚未充分探索
- Challenger Prompt 复杂（8步结构），设计本身存在工程成本
- 对语料库质量有依赖，低质量文档可能生成有误导性的题目

---

## 十一、对 Agent-Lightning 项目的启发

我们的项目核心流程是：**从 few-shot 示例中提取解题策略 → 将策略应用于新问题**。SPICE 的设计对此有多个维度的参考价值：

### 11.1 策略提取的"难度校准"思路

SPICE 最值得借鉴的是**方差基准奖励**的思想：不是追求"最高正确率"的策略，而是追求"最有学习价值"的策略。

对应到策略提取场景：
- **当前做法**：训练模型从示例中提取策略，用固定 rubric 打分
- **借鉴方向**：可以设计一个**策略难度校准机制**——提取出来的策略，如果对各种新问题的解题成功率总在 40%~60% 之间浮动，说明这个策略处于"有效学习边界"，比总是 100% 成功（策略太简单）或 0%（策略太难）更有训练价值

### 11.2 信息不对称的 Challenger 角色设计

SPICE 中 Challenger 看到文档（含答案），Reasoner 看不到。这种**信息不对称**保证了挑战的真实性。

对应到 few-shot 策略提取：
- **示例提取器（Challenger）** 相当于：看到完整的 few-shot 示例（含解题过程+答案）
- **策略应用器（Reasoner）** 相当于：只看到提取出的策略摘要，不看原始示例
- 这正是我们项目的核心信息不对称设计，SPICE 从理论上验证了这种不对称是有效的

### 11.3 自适应课程的策略选择

SPICE 的 Challenger 随训练进展自动出更难的题。对应地，我们可以：
- 从简单问题的 few-shot 示例开始，提取基础策略
- 随着模型能力提升，切换到更复杂问题的示例，提取更高阶策略
- 用 Reasoner 通过率的方差来判断当前训练阶段应该使用什么复杂度的示例

### 11.4 语料库接地防止策略退化

SPICE 发现纯自博弈（R-Zero）会退化，原因在于没有外部知识锚定。类比到策略提取：
- 如果模型只在自己生成的"策略-问题"对上训练，可能出现**策略同质化**（所有题都用同一套策略模板）
- 借鉴 SPICE 的思路：确保训练示例来自**多样化的外部真实数据集**，而不是模型合成数据，是保持策略多样性的关键

### 11.5 多格式验证的通用性

SPICE 用 MCQ + Free-form 两种格式覆盖了几乎所有领域，无需专门的执行器。对于策略提取项目，可以类似地设计：
- 提取的策略不必局限于某一种格式（如纯推理步骤），可以包括选择题的解题模板、计算题的公式路径等多种策略类型

---

> **一句话概括**：SPICE 用"文档接地的对抗自博弈"解决了自博弈的幻觉和信息对称问题，其中**方差基准奖励（保持 50% 通过率的边界挑战设计）**和**信息不对称的 Challenger-Reasoner 架构**，为我们的 few-shot 策略提取项目提供了"策略难度校准"和"防止策略退化"的重要设计参考。

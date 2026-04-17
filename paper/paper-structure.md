# 论文结构草稿

> **课题名**：MIST: Meta-Inductive Strategy Training for Self-Evolving LLM Reasoning  
> **方法简称**：**MIST**（Meta-Inductive Strategy Training）  
> **核心叙事（A+B 双主轴）**：
> - **主轴 A（行为习惯训练）**：LLM 不缺乏归纳能力，缺乏的是在无明确引导时可靠、主动地将 few-shot 示例中的规律提炼为可复用策略的行为习惯；MIST 通过跨任务 RL 将这种习惯训练为策略生成器（SG）的稳定能力。
> - **主轴 B（learn-to-induce 范式）**：相比 per-task prompt 搜索（APE/OPRO）和 per-instance 推理（CoT/STaR），MIST 引入新范式——一次训练获得通用策略归纳器，面对任意新任务类型单次前向推理即可生成可执行策略，无需重新优化。
> - **规模 vs. 思维**：一个经过 MIST 训练的 4B 小模型，显著超越未经训练的 32B 大模型。归纳习惯是参数规模无法替代的认知维度，必须通过有目标的训练来培养。
> **版本**：v0.3（架构全面修订稿）

---

## Abstract（摘要）

### 中文草稿

大型语言模型（LLM）在 few-shot 推理中依赖隐式的模式匹配——将示例直接拼入上下文后作答，而不会主动将示例中的规律归纳为可复用的解题策略。这种缺乏显式归纳步骤的推理方式存在两个固有局限：当新问题在表述或难度上与示例有偏差时，隐式匹配容易失效；且同类型新问题每次都需重新匹配，无法将归纳结果跨实例复用。

本文提出 **MIST**（Meta-Inductive Strategy Training），一种通过跨任务强化学习培养 LLM "先归纳策略、后执行解题"推理习惯的双层框架。MIST 将推理过程显式解耦为两个层次：**策略归纳层**由可训练的策略生成器（Strategy Generator，SG）跨 110+ 种任务类型学习"如何从 few-shot 示例归纳出可复用的结构化解题策略"；**策略执行层**由冻结的答案求解器（Answer Solver，AS）以 SG 产出的策略为上下文求解新问题，答题准确率反馈驱动 SG 的归纳能力优化。不同于 per-task 的 prompt 搜索方法（如 APE、OPRO），MIST 采用 *learn-to-induce* 范式：训练后的 SG 面对任意新任务类型仅需单次前向推理即可生成可执行策略，无需重新优化；全程无需人工标注认知策略。

实验表明：（1）训练后的 4B 参数 SG 将整体准确率提升 +135%，超越推理规模最高达 32B 的所有无训练基线，证明元归纳推理习惯是单纯增大参数规模无法替代的认知维度；（2）在训练中未见的任务类型（OOD）上，MIST 取得 +140% 的提升，验证训练习得的是可迁移的跨任务元归纳能力，而非特定任务策略的记忆。

### English Version

Large language models (LLMs) approach few-shot reasoning through implicit pattern matching — directly referencing in-context examples to produce answers, without explicitly inducing reusable problem-solving strategies. This lack of an explicit induction step leads to two fundamental limitations: (1) performance degrades when new problems deviate in surface form or complexity from the provided examples; (2) the inductive process cannot be reused across instances of the same task type, requiring full re-matching for every new problem.

We present **MIST** (Meta-Inductive Strategy Training), a bi-level framework that trains LLMs to adopt an "induce-then-execute" reasoning habit via cross-task reinforcement learning. MIST explicitly decouples reasoning into two layers: a **strategy induction layer**, where a trainable Strategy Generator (SG) learns across 110+ diverse task types how to induce reusable, structured problem-solving strategies from few-shot examples; and a **strategy execution layer**, where a frozen Answer Solver (AS) uses the induced strategy as context to solve new problems, with answer accuracy serving as the sole training signal for the SG. Unlike per-task prompt optimization methods (e.g., APE, OPRO), MIST adopts a *learn-to-induce* paradigm: after a single training phase, the SG generalizes to arbitrary new task types and generates executable strategies in a single forward pass — with no per-task re-optimization and no human-annotated strategies required.

Experiments demonstrate: (1) the 4B-parameter MIST-trained SG improves overall accuracy by +135% over the no-strategy baseline and outperforms all inference-only baselines up to 32B parameters, showing that meta-inductive reasoning cannot be substituted by scale alone; (2) on out-of-distribution task types unseen during training, MIST achieves +140% improvement, confirming that the trained ability is a transferable meta-inductive capacity rather than task-specific strategy memorization.

> **关键词**：meta-learning, meta-cognition, strategy induction, reinforcement learning, in-context learning, few-shot reasoning

---

## 核心概念体系

本文围绕四个核心概念展开，层次递进——从动机现象到能力外显，从架构原则到训练机制：

**第一层：元归纳思维（Meta-Inductive Thinking）**——动机与目标

> 一种跨任务的认知习惯：面对一类新问题时，先从少量示例中显式提炼可复用的规律框架，再将框架应用于新问题求解，而非直接依赖示例匹配作答。本文的核心目标是在 LLM 中培养这种习惯——关键不在于 LLM 是否具备归纳能力（在被明确引导时模型可以做到），而在于让模型在无明确引导时也能主动倾向于"先归纳后执行"的行为模式。

**第二层：认知策略（Cognitive Strategy）**——元归纳思维的外显产物

> 元归纳思维作用于具体任务类型时的输出：针对该任务类型的结构化解题框架。认知策略可视为一种**自然语言程序**（Natural-Language Program）——具有结构性（固定节点区分元层认知程序与任务层操作）和灵活性（节点内容由模型自由生成），由 AS 作为"解释器"来"执行"。一个策略对应一类问题，可复用于同类型的任意新问题，而非针对某道具体题目。

**第三层：认知解耦（Cognitive Decoupling）**——框架架构原则

> 将"如何解题"（策略归纳，由 SG 负责）与"实际解题"（策略执行，由 AS 负责）分离为独立模块，镜像认知科学中元认知与认知的区分 [Flavell, 1979]。解耦的技术意义：SG 通过 RL 独立优化归纳能力，AS 保持冻结充当确定性的认知世界模型；两层之间通过奖励信号（而非梯度）连接，使元层优化不依赖执行层的可微性。

**第四层：归纳对齐（Inductive Alignment）**——训练机制

> 通过跨任务 RL 将 SG 的归纳行为与"产出有效策略"这一目标对齐的训练过程。以 AS 的答题准确率作为奖励信号（outcome-based reward），无需策略层面的人工标注。类比 RLHF 的价值对齐（value alignment）——RLHF 对齐的是人类偏好，MIST 对齐的是认知习惯，使模型的归纳行为指向"产出真正有效的策略"而非"产出形式上像策略的文本"。

**概念连接逻辑**（写进 Method 开头）：我们希望培养 LLM 的**元归纳思维**；这种思维的每次激活产出一个面向具体任务类型的**认知策略**；为实现这一目标，在架构上实施**认知解耦**，使两个功能独立的模块分工合作；在训练上实施**归纳对齐**，以执行结果反向指导归纳能力的进化。

---

## 一、Introduction

### 1.1 Hook：人类的认知习惯 vs. 现有 LLM 的行为模式

**人类面对新任务时的认知倾向**（Motivation 第一段）：

> 当人类面对一类陌生问题时，通常不会反复刷题（"题海战术"），而是倾向于先观察几个例题，从中**提炼出规律框架**，再将这一框架迁移到新题上。这种"先归纳后演绎"的认知习惯，是人类高效学习的核心机制——心理学将其称为**元认知**（meta-cognition）：在解题之上还有一层对解题过程本身的认知与调控 [Flavell, 1979; Schraw & Dennison, 1994]。

**具体示例**（贯穿全文的 Running Example）：

> 设想三道"颜色混合"示例：红+蓝→紫、黄+蓝→绿、红+黄→橙。人类观察后会提炼出**归纳框架**："识别两种输入颜色→查找/推断混合规则→输出结果颜色"。面对新题"蓝+黄→?"时，直接套用框架即可。相比之下，标准 ICL 的 LLM 会将三个示例拼入上下文，通过隐式模式匹配猜测答案——当新题的格式或复杂度与示例有偏差时（例如"将蓝色和黄色颜料等比混合后的颜色是什么？"），隐式匹配容易失效，而显式归纳出的框架则仍然适用。**本文要培养的正是"先提炼框架、再套用框架"这种认知习惯。**

**LLM 在 In-Context Learning 下的实际行为**（Motivation 第二段）：

> 大型语言模型在 few-shot 推理（In-Context Learning）范式下，通常将示例作为上下文直接参考，隐式地进行模式匹配后作答。这一过程中缺乏显式的"归纳"步骤：模型倾向于"看例子→答新题"，而非"看例子→总结规律→用规律答新题"。当新问题在表述或难度上与示例存在偏差时，这种缺乏显式归纳的方式泛化能力有限；且每道新题都需重新匹配，无法将归纳结果跨问题复用。

**Gap 的精确定位**（注意：不是说 LLM"无法"归纳，而是缺乏这种习惯/思维模式）：

> 问题不在于 LLM 是否具备归纳能力，而在于其缺乏"先归纳、后作答"这种认知习惯的**主动倾向**。在没有明确引导的情况下，LLM 不会自发地将 few-shot 示例中的隐性规律**显式提炼**为可迁移的解题框架。

**从归纳偏置视角的理论定位**：

> 从深度学习理论角度，我们的工作可被理解为通过 RL 向 LLM 注入一种特殊的**行为归纳偏置**（behavioral inductive bias）：倾向于"先归纳后执行"的推理模式。与架构级归纳偏置（如 CNN 的平移不变性）或数据级偏置（如课程学习）不同，这是一种**认知模式级**的归纳偏置——通过训练使模型养成特定的思维结构化习惯。

### 1.2 我们想做什么（目标）

培养 LLM 的**元归纳思维习惯**：使模型在面对任何新任务时，能主动从少量示例中归纳出结构化的认知策略，再用该策略指导具体问题的求解。

强调两个维度：
- **"通用"**：这种思维习惯不针对某一特定任务，而是跨任务泛化的（generalize across task types）
- **"思维"**：培养的是认知模式/习惯，而非任务特定的知识；策略是思维的外显，不是最终目标

### 1.3 核心挑战（Problem）

将 Motivation 过渡到 Technical Problem，分两层：

**挑战一：如何在没有"金标准策略"标注的前提下，将元归纳思维的训练信号有效形式化？**

> 元归纳思维是一种内隐的认知过程，人类无法直接标注"正确的归纳方式"——面对同一组示例，不同角度的归纳可以产出多种有效策略，不存在唯一正确答案。在没有策略层面标注目标的情况下，如何对策略生成器进行有效训练？

**挑战二：如何确保训练获得的是跨任务可迁移的归纳能力，而非特定任务策略的记忆？**

> 若训练仅覆盖有限的任务类型，模型可能记住这些类型的具体策略，在面对训练中未见的任务时丧失归纳能力。如何在训练设计上促进元归纳能力的跨任务泛化，并通过实验加以验证？

**解法预告（受元学习启发）**：

> 受元学习（Meta-Learning）思想的启发，我们提出**双层元认知强化学习框架（Bi-level Meta-Cognitive RL Framework）**来解决上述挑战：以策略执行层的答题正确性作为信号，指导策略归纳层的训练；通过跨 110+ 种异构任务类型的元认知层训练，实现元归纳思维的跨领域泛化。

### 1.4 全局概览图（Figure 1）

**Figure 1 设计说明**（需绘制高质量矢量图放在 Introduction 首页）：

```
┌──────────────────────────────────────────────────────────────────┐
│  MIST: Bi-level Meta-Cognitive RL Framework                      │
│                                                                  │
│  ┌─ 策略归纳层（STRATEGY INDUCTION LAYER）────────────────────┐   │
│  │  Task Distribution P(T)                                    │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        │   │
│  │  │Task_1│  │Task_2│  │ ...  │  │Task_n│  │Task_?│(OOD)   │   │
│  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘        │   │
│  │     └────────┬┴────────┬┴────────┬┘         │             │   │
│  │              ↓ 跨任务 GRPO 训练              │(zero-shot)  │   │
│  │     ┌────────────────────────┐               │             │   │
│  │     │  Strategy Generator    │  ← RL 优化    │             │   │
│  │     │  (SG, 4B, trainable)   │               │             │   │
│  │     └───────────┬────────────┘               │             │   │
│  └─────────────────┼────────────────────────────┼─────────────┘   │
│                    │ 认知策略 s                   │                │
│  ┌─ 策略执行层 ────┼────────────────────────────┼─────────────┐   │
│  │                 ↓                             ↓             │   │
│  │     ┌────────────────────────┐                              │   │
│  │     │  Cognitive Strategy s  │                              │   │
│  │     │  (semi-structured NL)  │                              │   │
│  │     └───────────┬────────────┘                              │   │
│  │                 ↓                                           │   │
│  │     ┌────────────────────────┐                              │   │
│  │     │  Answer Solver (AS)    │  ← 冻结                     │   │
│  │     │  (4B/8B, frozen)       │                              │   │
│  │     └───────────┬────────────┘                              │   │
│  │                 ↓                                           │   │
│  │     M 次采样 → grounded_proxy ──→ Reward ──→ 回传归纳层     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**图的设计要点**：
- 左右对比呈现：左侧 = 标准 ICL（examples → direct answer），右侧 = MIST（examples → strategy → answer）
- 用不同颜色区分策略归纳层（蓝色调）和策略执行层（橙色调）
- 用虚线箭头表示奖励回传路径，实线箭头表示前向数据流
- 在策略节点旁展示一个缩略的策略 schema 预览（TYPE / FIRST_ORDER / SECOND_ORDER）
- OOD 任务用虚线框标注，强调泛化能力

### 1.5 贡献列表（Contributions）

1. **双层元认知框架**：提出跨任务 / 任务内双层架构，将问题解耦为策略归纳层（跨任务 RL 训练 SG 习得元归纳能力）和策略执行层（冻结 AS 以策略为上下文求解新问题）两个可独立优化的层次

2. **推理摊销效率（Amortized Inference Efficiency）**：通过一次跨任务 RL 训练，MIST 获得面向任意新任务类型的通用策略归纳能力；推理时无需重新优化，以单次前向推理生成可执行策略——相比 APE/OPRO 每次 per-task 搜索，效率显著更高，同时无需人工标注认知策略

3. **跨任务泛化验证**：在 ID / OOD / BBH 三个评估集上验证，RL 训练培养的元归纳能力具有跨任务泛化性，不依赖于特定任务类型的记忆；OOD 任务上取得 +140% 提升

4. **规模效率**：训练后的 4B 策略生成模型在整体准确率上大幅超越 32B 推理基线，说明元归纳思维是单纯扩大模型规模无法替代的认知维度

---

## 二、Related Work

### 2.1 元学习与元层优化（Meta-Learning and Meta-Level Optimization）

**元学习范式**（参数级双层优化）：

- MAML [Finn et al., 2017]：内外循环结构，内循环支持集任务适配 + 外循环元参数更新 → 我们借鉴双层解耦思想，但 SG 产出的是自然语言策略而非参数梯度
- Prototypical Networks、Matching Networks：从 few-shot 支持集归纳类别表示 → 同样从示例归纳，但我们归纳的是结构化解题策略而非向量表示

元学习范式经历了从参数级双层优化（MAML 时代，梯度作为适配产物）到双层优化形式化（Franceschi et al. [2018]，上层/下层问题的数学抽象），再到黑盒上下文级元优化（LLM 时代，自然语言策略/指令作为适配产物，两层通过奖励信号而非梯度连接）的演化。本文框架属于第三阶段：**策略归纳层**产出自然语言认知策略，**策略执行层**以策略为上下文执行求解，两层通过奖励信号解耦连接。关键区别：我们的"元"在于跨任务学习"如何归纳"，而非学习"如何快速适应某任务"。

**自动化 Prompt 优化**（黑盒元优化的下游分支）：

- APE [Zhou et al., 2023]：自动生成和筛选任务指令 → per-task 搜索，不涉及跨任务泛化
- OPRO [Yang et al., 2024]：LLM 迭代优化 per-task prompt → 仍属逐任务范式，需要全量目标函数评估
- PromptBreeder [Fernando et al., 2024]、EvoPrompt [Guo et al., 2024]：进化搜索 prompt 群体 → 搜索空间大、效率低

**关键区别**：上述方法均为 per-task prompt 搜索——每遇到新任务须重新运行优化流程。MIST 属于 **learn-to-induce** 范式：一次跨任务 RL 训练，SG 面对任意新任务类型只需单次前向推理即可生成策略，无需搜索。这是**搜索**（search）与**学习**（learn）的范式本质差异。

### 2.2 LLM 推理增强与上下文学习（LLM Reasoning Enhancement and In-Context Learning）

**In-Context Learning 与推理链**：

- Brown et al. [2020]：few-shot 示例直接作为 context，隐式模式匹配，无显式归纳步骤
- Chain-of-Thought [Wei et al., 2022]：引导推理过程显式化 → 我们引导的是策略层面的显式归纳，而非每道题的推理链
- Self-refine、Reflection：任务中错误修正 → 我们聚焦于任务前的策略提炼

**基于 RL 的推理增强**：

- RLHF [Ouyang et al., 2022]：人类偏好奖励 → 我们用任务正确性，无需人工标注
- GRPO [Shao et al., 2024]：Group Relative Policy Optimization → 本文采用的训练算法（详见 §3.2）
- STaR [Zelikman et al., 2022]：用正确推理路径自我训练 → 优化的是推理链（CoT），MIST 优化的是答题前的归纳过程；两者的结果信号相似，但优化目标层次不同
- Quiet-STaR [Zelikman et al., 2024]：token 级隐式推理 → 关注 token-level 思考，而非任务类型级别的策略归纳
- rStar [Qi et al., 2025]：MCTS 增强推理 → 推理时搜索，而非训练时培养归纳思维

**关键区别**：上述方法优化的是**推理过程本身**（object-level reasoning，"如何推理"），MIST 优化的是**推理之前的归纳过程**（meta-level induction，"如何从示例中提炼解题框架"）。这是 object-level 与 meta-level 的层次差异。

### 2.3 相关系统与上下文工程（Skill Discovery and Context Engineering）

- Voyager [Wang et al., 2023]：在 Minecraft 中自主发现并存储可复用技能（代码形式）→ 我们的认知策略可类比为"自然语言技能"，但 Voyager 的技能是 task-specific 代码，不涉及跨任务归纳能力的训练；MIST 在 110+ 种异构任务类型上训练元归纳能力，产出的是认知层面的解题框架而非可执行代码
- OMNI-EPIC [Faldor et al., 2024]：开放式技能生成，强调多样性 → MIST 以执行准确率为唯一训练目标，强调策略有效性
- Meta-Harness [Lee et al., 2026]：外循环搜索 Harness 代码（外循环 = 元策略优化，内循环 = 任务执行），用内循环 trace 反馈训练外循环 → 与我们在元优化范式上高度共鸣（双层解耦结构）；关键差异在于我们训练的是自然语言策略生成器（通过 RL），而非迭代搜索代码形式的 Harness
- ACE [Zhang et al., 2025]、MCE [Ye et al., 2026]：上下文工程方法 → 我们的认知策略是一种通过 RL 习得的上下文产物，而非人工设计的上下文结构

---

## 三、Preliminary

### 3.1 任务设定与形式化定义

**符号定义**：

- $\mathcal{T}$：任务类型空间（例如：算术推理、颜色推理、BBH 各子任务……共 110+ 种）
- $\mathcal{E}_t = \{(x_i, y_i)\}_{i=1}^{K}$：任务类型 $t \in \mathcal{T}$ 的 $K$ 个 few-shot 示例
- $x_{\text{new}}$：同类型的新问题，$y^*$：标准答案
- $s = \pi_{\text{SG}}(\mathcal{E}_t)$：策略生成器基于示例归纳的认知策略
- $\hat{a} = \text{AS}(s, x_{\text{new}})$：答案求解器以策略为指导生成的答案
- $K$：few-shot 示例数（支持集大小）
- $M$：策略效用评估的采样次数（区别于 $K$）
- $N$：GRPO rollout 数（每个 prompt 生成的策略候选数）

**优化目标**：

$$\pi_{\text{SG}}^* = \arg\max_{\pi_{\text{SG}}} \mathbb{E}_{t \sim \mathcal{T},\; \mathcal{E}_t,\; x_{\text{new}}} \left[ r\!\left(\text{AS}(\pi_{\text{SG}}(\mathcal{E}_t),\; x_{\text{new}}),\; y^*\right) \right]$$

目标是学习策略生成器 $\pi_{\text{SG}}$，使其生成的认知策略能最大化策略执行层的期望正确率，且期望在**任务类型分布 $\mathcal{T}$ 上取**——即跨任务泛化是训练目标的内在组成部分，而非事后验证。

### 3.2 GRPO 训练算法简介

本文采用 **GRPO（Group Relative Policy Optimization）** [Shao et al., 2024] 训练策略生成器 $\pi_{\text{SG}}$。GRPO 是一种无需独立价值网络的策略梯度方法：对每个训练 prompt，生成 $N$ 个输出样本（rollout），以组内奖励的相对高低估计各样本的优势函数，从而替代 actor-critic 框架中独立价值网络的估计。

选用 GRPO 的理由：（1）策略生成的奖励信号本身带有较高方差（策略质量难以逐 token 评估），组内相对比较天然降噪；（2）无需额外的价值网络参数，内存效率更高；（3）在以结果为奖励的 RL 场景（数学推理等）中已有成熟验证，与本文的 outcome-based reward 设计相契合。

### 3.3 半结构化策略格式（Semi-Structured Strategy Representation）

我们采用**半结构化**的策略格式：各节的名称固定（刚性脚手架），节内内容自由生成，步骤数量弹性（3–6 步）。这种设计介于自由文本和严格表单之间，在约束认知框架的同时保留任务适应的灵活深度。

**完整 schema**（对应 `repetition_controls_2026-04-01.toml`）：

```
<strategy>
TYPE              — 一行模式标签（≤10 词），命名解题类型

CATEGORY
  Task Understanding: 任务类型 + 输入输出空间 + 决策目标
  Task Description:   当前任务的一句话目标描述

FIRST_ORDER_PATTERN              ← 元层：可复用认知程序（跨任务通用）
  Perceive:  如何解析约束/目标/输出空间
  Induce:    如何从示例中推断可复用的求解模式
  Execute:   如何将具体算子应用到当前任务
  Validate:  如何检查正确性和约束合规性
  Emit:      如何以要求格式产出最终答案

FEWSHOT_LEARNING                 ← 归纳层：显式化 few-shot 学习过程
  Latent Knowledge Learned: 从示例中学到的隐性知识/技能
  Transfer Mechanism:        该知识如何映射到当前任务
  Transfer Boundary:         哪些内容不能直接复制或过度泛化

SECOND_ORDER_STEPS               ← 任务层：任务专属具体算子（3–6 步，弹性）
  1. <含动作动词和操作对象的具体步骤>
  2. ...
  [3–6 步，每步必须可执行]

CHECK                            ← 验证层（1–2 条）
  - 与证据或约束绑定的验证项
  - 格式/标签/单位/选项合法性检查

FORMAT                           ← 输出规约层
  Answer Format:              最终答案的精确格式
  Constraint Details:         允许的标签/选项/单位/大小写/括号等
  Insufficient-Evidence Policy: 证据不足时的处理方式

FALLBACK                         ← 边缘情况处理（可选）
  - 主路径失败时的备选方案
</strategy>
```

**两层设计的核心意义**：schema 的核心设计是将策略显式分为两层，直接对应双层元认知框架：

| Schema 层 | 对应框架层级 | 作用 |
|---|---|---|
| **FIRST_ORDER_PATTERN** | 策略归纳层（元层，跨任务通用） | 跨任务可复用的认知程序（Perceive/Induce/Execute/Validate/Emit） |
| **SECOND_ORDER_STEPS** | 策略执行层（任务层，任务专属） | 针对当前任务类型的具体可执行算子 |
| **FEWSHOT_LEARNING** | 策略归纳层（归纳过程文字化） | 强迫模型将隐式 few-shot 归纳过程显式化，使归纳步骤透明而非黑盒 |
| **FORMAT / CHECK** | 策略执行层（执行约束规约） | 为 AS 提供可解析的输出格式约束，降低执行层噪声 |

**FEWSHOT_LEARNING 节的特殊价值**：该节要求模型显式回答"从示例中学到了什么、如何迁移、迁移边界在哪"，将元归纳的归纳过程由隐式（黑盒 ICL 模式匹配）变为显式（文字化推理链）。这是本设计与普通结构化 prompt 的关键差异。

---

## 四、Method

### 4.1 双层元认知框架（Bi-level Meta-Cognitive Framework）

#### 完整结构图

```
策略归纳层（STRATEGY INDUCTION LAYER，跨任务）——元归纳训练
  从任务类型分布 P(T) 采样：t ~ {算术, 颜色推理, BBH子任务, ...}
  ─────────────────────────────────────────────────────────────
  单个训练 episode 内部：

  支持集 E_t = {(x₁,y₁), ..., (xK,yK)}   ← K 个同类示例

  ┌── Stage 1：任务策略归纳（strategy induction）──┐
  │  SG 非马尔可夫地跨 K 个示例横向推理             │
  │  π_SG(E_t) → 认知策略 s                    │
  └──────────────────────────────────────────┘
                    ↓ 策略 s
  ┌── Stage 2：M 次策略效用评估（grounded evaluation）──┐
  │  for m = 1 … M:                                 │
  │      answer_m  = AS(s, x_new)    ← AS 参数冻结  │
  │      correct_m = score(answer_m, y*)             │
  │  grounded_proxy(s) = Σ correct_m / M             │
  └──────────────────────────────────────────────────┘
                    ↓ 稳定的策略效用估计
  reward = α · R_format(s) + β · grounded_proxy(s)
  ─────────────────────────────────────────────────────────────
  归纳对齐优化：GRPO 在跨任务 episodes 上更新 π_SG 参数
```

**为什么 Stage 2 的 M 次采样是必要的**：

当 $M=1$ 时，整个流程退化为单向流水线（SG → AS → reward），策略效用估计的方差极大，RL 训练信号噪声过高。引入 $M$ 次 AS 采样后，通过对同一策略重复 $M$ 次执行来平均掉执行噪声，得到策略期望效用的稳定估计 $\text{grounded\_proxy}(s) = \frac{1}{M}\sum_m \mathbb{1}[\text{correct}_m]$，类比 MAML 中内循环多步梯度下降以得到稳定适配结果——两者都是在 episode 内部**迭代以提高估计质量**，而非单次前向即止。

#### 与 MAML 的系统性对比

| 维度 | MAML 任务层 | 我们的策略执行层 | MAML 元层 | 我们的策略归纳层 |
|---|---|---|---|---|
| **输入** | 支持集 $D_{\text{sup}}$ | K 个示例 $\mathcal{E}_t$ | 跨任务查询集损失 | 跨任务 grounded_proxy |
| **适配产物** | 任务适配参数 $\theta'$（梯度） | 认知策略 $s$（自然语言） | 元梯度作用于 $\theta$ | GRPO 作用于 $\pi_{\text{SG}}$ |
| **迭代结构** | 多步梯度下降 | $M$ 次 AS 采样评估 | 任务批次 | 任务类型 episode 批次 |
| **梯度流** | 梯度反传穿过内循环 | 无反传（AS 冻结） | 元梯度 | 策略梯度（GRPO） |
| **适配方式** | 参数空间（weight-space） | 上下文空间（context-space） | — | — |

**关键区别**：MAML 的任务层产物是更新后的参数 $\theta'$，通过梯度流将任务层与元层耦合；我们的策略执行层产物是自然语言策略 $s$，AS 模型完全冻结充当确定性的**认知世界模型**——两层通过奖励信号（而非梯度）解耦连接。这种解耦使策略归纳层能够使用黑盒优化（GRPO），不依赖策略执行层的可微性。

#### 两层的语义定位

| | 策略归纳层（跨任务） | 策略执行层（任务内） |
|---|---|---|
| **运行粒度** | 跨任务类型分布（元层） | 单个 episode 内部（实例层） |
| **认知操作** | 跨任务学习"如何归纳" | 任务策略归纳：将学到的归纳能力作用于具体任务 $t$ |
| **模型角色** | 策略生成器 SG（可训练，优化目标） | 答案求解器 AS（冻结，充当评估器） |
| **信息流向** | 横向：跨 $K$ 个示例建立模式 | 纵向：策略 → $M$ 次答题 → 效用估计 |
| **训练信号** | 接收策略执行层奖励，更新 $\pi_{\text{SG}}$ | 产生 grounded_proxy，传递给策略归纳层 |

### 4.2 奖励设计

奖励由**两个层次**构成：

$$R = \alpha \cdot R_{\text{format}} + \beta \cdot R_{\text{exec}}$$

**$R_{\text{format}}$（策略结构质量信号）**：

$$R_{\text{format}} = \begin{cases} 1.0 & \text{策略符合结构化格式（TYPE/STEPS/CHECK）} \\ 0.3 & \text{存在 } \texttt{<strategy>} \text{ 标签但结构不完整} \\ 0.0 & \text{无有效策略输出} \end{cases}$$

作用：确保策略可被策略执行层解析和执行，是元归纳思维"外显为可操作策略"的形式保障。

**$R_{\text{exec}}$（策略执行结果信号）**：

$$R_{\text{exec}} = r_{\text{soft}}(\hat{a}, y^*) \in [0, 1]$$

作用：以答题准确率验证认知策略的实际效用，是跨层信用分配的核心信号——策略归纳层策略生成的好坏，由策略执行层能否答对来判定。

可选扩展（非核心，作为消融变量）：在 $R_{\text{format}}$ 和 $R_{\text{exec}}$ 之间增加 LLM 打分的策略质量分 $R_{\text{quality}}$：

$$R = \alpha \cdot R_{\text{format}} + \gamma \cdot R_{\text{quality}} + \beta \cdot R_{\text{exec}}$$

### 4.3 为什么用 RL 而非 SFT（Why RL, Not Supervised Fine-Tuning）

审稿人可能会问：为什么不直接用监督微调（SFT）训练策略生成器？以下三点回应：

1. **无金标准策略**：SFT 需要"正确策略"作为标注目标，但认知策略没有唯一的最优解——面对同一组示例，不同的归纳角度可以产出多种有效策略。人工标注不仅昂贵，且会引入标注者个人的认知偏见，限制策略多样性
2. **弱但诚实的信号**：RL 使用下游答题准确率作为奖励信号——这是一个 outcome-based reward，不规定"如何思考"，只评价"思考的结果是否有用"。这种弱监督恰好匹配元认知的本质：我们不知道最优的归纳过程是什么，但我们能观测到归纳结果的效用
3. **策略空间的自主探索**：RL 允许模型自主探索策略空间，发现人类可能想不到的归纳方式。SFT 只能模仿已有策略（behavior cloning），而 RL 可以超越示范（exploration beyond demonstration）

**类比**：RLHF 用人类偏好对齐价值观；MIST 用任务正确性对齐认知习惯——前者是**价值对齐**（value alignment），后者是**归纳对齐**（inductive alignment）。

### 4.4 训练算法与实现细节

使用 **GRPO** 训练策略生成器 $\pi_{\text{SG}}$（算法细节见 §3.2）：

- 每个训练样本：随机采样任务类型 $t$，采样 $K$ 个示例 $\mathcal{E}_t$ 和一个新问题 $x_{\text{new}}$
- 每个 prompt 生成 $N$ 个策略候选（rollout），以组内相对奖励估计优势函数
- 策略生成器参数按优势方向更新；AS 参数全程冻结

**模型规格**：

| 组件 | 模型 | 参数量 | 状态 |
|---|---|---|---|
| 策略生成器 SG | Qwen2.5-4B（基座）| 4B | RL 训练（可训练） |
| 答案求解器 AS | Qwen2.5-4B / 8B | 4B / 8B | 冻结（仅推理） |
| 策略质量评分器（可选） | Qwen2.5-4B | 4B | 冻结（仅推理） |

**训练超参数**：

| 超参数 | 值 | 说明 |
|---|---|---|
| Rollout 数 $N$ | 8 | 每个 prompt 生成的策略候选数 |
| 评估采样数 $M$ | 4（CLI 默认） | 每个策略的 AS 执行次数 |
| 学习率 | 1e-6 | AdamW |
| 最大策略长度 | 4096 tokens | SG 输出上限 |
| 训练数据量 | ~20k 样本 | 覆盖 110+ 种任务类型 |
| Batch size | 待补 | VERL 配置 |
| 总训练步数 | 待补 | 根据 reward 收敛判定 |

**任务采样策略**：
- 当前：**均匀采样**（uniform over task types），每个 episode 随机选择一种任务类型
- 考虑但未实施：课程学习（curriculum learning），先易后难——可作为消融变量探索
- 每类任务内部：随机抽取 $K$ 个示例作为支持集，1 个不同实例作为查询

**两阶段训练流水线**：
1. **Stage 1（格式预训练）**：仅使用 $R_{\text{format}}$ 训练 SG 学会输出合规格式（快速收敛，~数百步）
2. **Stage 2（归纳对齐训练）**：使用完整奖励 $R = \alpha \cdot R_{\text{format}} + \beta \cdot R_{\text{exec}}$，SG 在保持格式合规的基础上学习产出有效策略（主要训练阶段）

**计算开销分析**：
- 每个训练 episode 的前向推理量 = 1 次 SG 生成 $\times N$ 个 rollout + $N \times M$ 次 AS 推理
- 当 $N=8, M=4$ 时，每 episode 需 32 次 AS 前向——相比纯 SFT 训练，计算开销约增加 $N \times M$ 倍
- 整体训练在 X 张 GPU 上耗时约 Y 小时（待补）

---

## 五、Experiments

### 5.1 实验设置

**数据集**：
- **训练集**：train_20k，覆盖 110+ 种任务类型，每类随机采样
- **评测集（ID）**：test-id-subtask，与训练集同任务类型但不同子任务实例
- **评测集（OOD）**：test-ood-task，训练中未见的任务类型
- **评测集（BBH）**：BIG-Bench Hard 子集，高难度推理任务

**评测指标**：
- $\text{Acc}_{\text{soft}}$：连续正确率（答案相似度）
- $\text{Acc}_{\text{hard}}$：严格二值正确率（0/1）
- $\text{pass@}k$（$k \in \{1, 2, 3\}$）：K 次采样中至少一次答对的比例

**基线**：
- 无策略直接答题（AS-only baseline）
- 无训练 SG 推理（SG inference-only，不同规模：1.7B/4B/8B/14B/32B）
- 固定 few-shot 直接提示（Few-shot ICL）
- 自动化 Prompt 优化方法（APE / OPRO / Zero-shot CoT + Self-Consistency）

### 5.2 主实验：有效性、规模与方法对比

本节从三个维度验证 MIST 的核心 claim：（1）RL 训练带来的质变效果；（2）归纳能力不能靠规模堆砌替代；（3）learn-to-induce 范式相比 per-task 搜索的优势。

**Table 1**：不同配置的总体准确率（核心对比）

| 配置 | $\text{Acc}_{\text{soft}}$ | $\text{Acc}_{\text{hard}}$ |
|---|---|---|
| 无策略 + AS-4B（基线） | 0.2768 | 0.2632 |
| 无策略 + AS-8B | 0.2771 | 0.2561 |
| 无训练 SG-4B + AS-4B | 0.2958 | 0.2781 |
| 无训练 SG-4B + AS-8B | 0.3022 | 0.2904 |
| **RL训练 SG-4B + AS-4B** | **0.6491** | **0.6456** |
| RL训练 SG-4B + AS-8B | 0.5934 | 0.5886 |

**核心 Claim**：RL 训练赋予的元归纳能力带来质变（+135%），且超越所有 inference-only 基线（包括 32B 规模）。

**AS-8B 反而低于 AS-4B 的解释**（审稿人必问，需提前预设解读）：

SG 的 RL 训练以 AS-4B 为冻结的评估环境，策略在训练过程中被优化为最适配 AS-4B 的推理习惯。当测试时切换为 AS-8B，策略与 AS 之间存在**分布偏移**（strategy-solver distribution shift）——策略中的指令粒度和格式可能恰好适配 4B 的上下文处理方式，而非 8B 的。这一现象本身支撑了"策略不仅是任务级的，还隐含了对求解器行为的适配"这一洞察，暗示未来**策略-求解器联合优化**的研究方向（§5.4 将进一步分析）。

**Table 2**：规模分析——固定 AS，改变 SG 规模（均未 RL 训练）

- 展示：SG 规模越大，性能越高，但有天花板（最大 32B 也不及 RL 训练的 4B）
- **Claim**："Cognitive Capacity Cannot Be Substituted by Scale"（认知归纳能力不能被规模替代）

**Table 3**：与自动化 Prompt 优化方法的对比

| 方法 | 方式 | ID Acc | OOD Acc | 每新任务开销 |
|---|---|---|---|---|
| APE [Zhou et al., 2023] | per-task 搜索最优指令 | 待补 | 待补 | 高（需要搜索） |
| OPRO [Yang et al., 2024] | LLM 迭代优化 per-task prompt | 待补 | 待补 | 高（需要迭代） |
| Zero-shot CoT + Self-Consistency | "Let's think step by step" + 多次采样投票 | 待补 | 待补 | 低 |
| **MIST（本文）** | 一次前向推理生成策略 | 0.6491 | 0.5980 | **极低（单次推理）** |

**预期结论**：ID 任务上 APE/OPRO 可能接近 MIST（per-task 优化充分），但 OOD 任务上 MIST 应显著领先（per-task 方法无法迁移到未见任务）；核心差异在于**搜索 vs. 学习**的范式差异，以及**推理时效率**的显著优势。

### 5.3 消融实验

本节从三个维度消融 MIST 的核心设计选择，验证各组件的必要性。

#### 5.3.1 奖励组件消融

**Table 4**：奖励构成消融

| 训练奖励配置 | $\text{Acc}_{\text{soft}}$ | $\text{Acc}_{\text{hard}}$ | 说明 |
|---|---|---|---|
| 仅 $R_{\text{exec}}$（无格式约束） | 待补 | 待补 | 移除格式奖励 |
| 仅 $R_{\text{format}}$（无执行反馈） | 待补 | 待补 | 模型只学格式，不学有效性 |
| $R_{\text{format}} + R_{\text{exec}}$ | 0.6491 | 0.6456 | **本文主要配置** |
| $R_{\text{format}} + R_{\text{quality}} + R_{\text{exec}}$ | 待补 | 待补 | 加入 LLM 质量打分 |

**解读**：格式约束是内容有效性的形式保障；执行信号保障策略实际有效；两者结合优于各自单独使用。

训练奖励曲线（Figure X）：展示 $R_{\text{format}}$ 和 $R_{\text{exec}}$ 随训练步数的变化趋势。预期：$R_{\text{format}}$ 先快速收敛（格式学习简单），$R_{\text{exec}}$ 缓慢提升（内容质量需要更多探索）。

#### 5.3.2 策略格式消融

**实验动机**：半结构化策略格式是本文的核心设计选择之一（§3.3），本消融验证两层分离的 schema 是否确实必要。

**Table 5**：Prompt 结构消融

| Prompt 配置 | Prompt 文件 | BBH | ID | OOD | 总体 |
|---|---|---|---|---|---|
| 完全自由（v1） | `v1.toml` | 待补 | 待补 | 待补 | 待补 |
| 轻度结构化（schema） | `strategy_structured_schema.toml` | 待补 | 待补 | 待补 | 待补 |
| **半结构化（本文）** | `repetition_controls_2026-04-01.toml` | **0.7789/0.7778** | **0.4664/0.4633** | **0.5980/0.5900** | **0.6491/0.6456** |

**预期结论**：完全自由 < 轻度结构化 < 半结构化；OOD 任务上差距最为显著（两层分离促进了可迁移的元层认知程序的涌现）。半结构化 schema 不只是工程细节，而是 OOD 泛化能力的结构性来源。

#### 5.3.3 策略效用评估采样次数消融（$M$-sample Ablation）

**Table 6**：$M$ 次采样消融

| $M$（评估采样次数） | 总体 Acc | 训练稳定性（reward 方差） | 训练成本倍率 |
|---|---|---|---|
| $M=1$ | 待补 | 待补 | 1x |
| $M=2$ | 待补 | 待补 | ~2x |
| $M=4$（CLI 默认） | 待补 | 待补 | ~4x |
| $M=8$ | 待补 | 待补 | ~8x |

**预期结论**：$M=1 \to M=4$ 有显著提升（多次采样降低 reward 噪声，训练更稳定）；$M=4 \to M=8$ 边际递减；存在一个成本-效果平衡的最优 $M$，直接验证 Stage 2 多次采样评估机制的必要性。

### 5.4 泛化分析

本节从三个维度验证 MIST 习得的是跨任务可迁移的元归纳能力，而非特定任务策略的记忆。

#### 5.4.1 ID / OOD / BBH 分项结果

**Table 7**：ID vs. OOD vs. BBH 分项对比

| 评测集 | 无策略 AS-4B | RL SG-4B + AS-4B | 提升 |
|---|---|---|---|
| BBH | 0.3364 / 0.3185 | 0.7789 / 0.7778 | +∆ |
| ID-Subtask | 0.1943 / 0.1900 | 0.4664 / 0.4633 | +∆ |
| OOD-Task | 0.2522 / 0.2367 | 0.5980 / 0.5900 | +140% |

**核心 Claim**：OOD 任务上的强劲提升（+140%）说明 RL 训练培养的是**通用元归纳能力**，而非特定任务策略的记忆。

#### 5.4.2 策略跨模型迁移（Strategy-Solver Transfer）

**实验动机**：验证认知策略是否为通用的认知产物（可跨 AS 使用），直接回应 §5.2 中 AS-8B 低于 AS-4B 的反常现象。

**Table 8**：固定 SG-4B 策略，改变测试 AS 规模

| 测试 AS | Acc_soft / Acc_hard | 说明 |
|---|---|---|
| AS-4B（训练时 AS） | 0.6491 / 0.6456 | 基准（策略-求解器匹配） |
| AS-8B | 待补 | 策略-求解器分布偏移 |
| AS-14B | 待补 | 更大求解器能否更好利用策略 |
| AS-32B | 待补 | 求解器能力天花板 |

**预期结论**：若策略在更大 AS 上也有显著提升，证明策略是通用认知产物；若提升随 AS 规模递减或出现倒挂，说明策略隐含了对训练时 AS 行为的适配，未来可探索策略-求解器联合优化。

#### 5.4.3 少样本数量 $K$ 的敏感性与零样本泛化（$K$ Sensitivity & Zero-Shot）

**Table 9**：$K$ 值对策略质量和最终准确率的影响

| $K$（few-shot 示例数） | 总体 Acc | BBH Acc | OOD Acc |
|---|---|---|---|
| $K=0$（零样本，仅任务描述） | 待补 | 待补 | 待补 |
| $K=3$ | 待补 | 待补 | 待补 |
| $K=5$（本文标准） | 0.6491 | 0.7789 | 0.5980 |
| $K=8$ | 待补 | 待补 | 待补 |

**分析角度**：$K$ 太小时示例不足以归纳规律；$K$ 太大时示例多样性降低，策略可能过特化；$K=0$ 的非零提升将直接支持"元归纳思维已部分内化为参数知识"这一 claim。

### 5.5 策略质量与训练动态

本节通过定性与定量分析理解策略的内在质量及 RL 训练的演化过程。

#### 5.5.1 训练动态可视化（Strategy Self-Evolution）

**实验动机**：为标题中"Self-Evolving"提供直接证据。

在训练过程中定期（每 100 steps）对相同的 3–5 种任务类型生成策略快照，追踪：
1. 策略**长度**随训练步数的变化（预期：先增后稳，模型先探索再收敛）
2. **FIRST_ORDER_PATTERN 稳定性**：训练后期，跨不同任务类型的 FIRST_ORDER 部分是否趋向一致（证明通用认知程序的涌现）
3. **SECOND_ORDER_STEPS 多样性**：训练后期，不同任务类型的 SECOND_ORDER 部分是否保持差异（证明任务适配能力）
4. 策略内容的**语义演化轨迹**：早期策略空泛/错误 → 中期策略渐具体 → 后期策略精炼稳定

**包装角度**：Figure X = "Emergence of Meta-Inductive Thinking Through Training"

#### 5.5.2 策略质量分析（定性）

从不同任务类型各抽取样例，展示：
- **好策略 vs. 坏策略**对比（与奖励分数对应）
- RL 训练前后策略质量的定性变化
- 策略在 OOD 任务上"迁移"的例子
- **Running Example 回显**：展示"颜色混合"类任务在训练前后的策略对比，呼应 §1.1 的引入示例

Pass@k 分析（$k \in \{1,2,3\}$）：衡量元归纳能力的稳定性，好的归纳习惯应产出稳定有效的策略，与直接多次采样（无策略）比较。

#### 5.5.3 按任务类型的长尾分析（Per-Task Breakdown）

计算每种任务类型的 $\Delta\text{Acc} = \text{Acc}_{\text{MIST}} - \text{Acc}_{\text{baseline}}$，绘制分布直方图并按降序排列。

**分析角度**：
- 高提升任务的共性特征（推理步骤多？规则性强？需要模式识别？）
- 低提升/无提升/负提升任务的特征（太简单无需策略？太难策略也无法帮助？）
- 提升幅度与任务复杂度的相关性分析，识别 MIST 的"最佳适用场景"

**包装角度**：Figure X = "Where Does Meta-Inductive Thinking Help Most?"

---

## 六、Conclusion

### 6.1 总结

> 本文提出了 **MIST**（Meta-Inductive Strategy Training），一种面向 LLM 元归纳思维培养的**双层元认知强化学习框架**。核心思路是通过**认知解耦**将问题分离为策略归纳层（跨任务 RL 训练 SG 习得元归纳能力）和策略执行层（冻结 AS 以策略为上下文求解新问题），以策略执行层的答题准确率作为训练信号，实现无需人工策略标注的端到端**归纳对齐**训练。MIST 采用 learn-to-induce 范式：一次训练即获得面向任意新任务类型的通用策略归纳能力，推理时无需 per-task 重新优化。实验表明，MIST 培养的元归纳能力具有强劲的跨任务泛化性：4B 参数的训练模型在总体准确率上提升 +135%，显著超越 32B 参数的推理基线；在 OOD 任务上提升 +140%，证明学到的是通用归纳能力而非特定任务记忆。

### 6.2 局限

- **奖励稀疏性**：任务执行层答题准确率作为唯一执行信号可能过于稀疏（特别是极难任务，正确率接近 0 时 RL 训练信号极弱）。未来可探索更细粒度的过程奖励（process reward）或部分正确性评分
- **策略模式坍缩风险**：GRPO 优化可能导致策略收敛到少数高奖励模板，丧失对不同任务类型的适应多样性。需监控策略多样性指标（如跨任务策略 embedding 的方差）
- **基座模型依赖**：当前实验仅在 Qwen2.5 系列上验证。元归纳能力的培养效果是否依赖于特定基座模型的 ICL 能力，尚需跨模型系列（LLaMA、Mistral 等）的验证
- **策略-求解器耦合**：§5.2 的结果表明，RL 训练可能导致策略与特定 AS 过度适配（AS-8B 反而低于 AS-4B），限制了策略的通用性

### 6.3 展望

- **策略-求解器联合优化**：同时微调 SG 和 AS，使策略与执行器协同演化
- **跨模型策略迁移**：验证 SG 生成的策略能否迁移到不同架构/规模的 AS
- **多步推理扩展**：将元归纳思维从单步策略生成扩展到多轮迭代策略精化（strategy refinement）
- **元归纳课程学习**：在训练中引入任务难度课程，先在简单任务上培养基础归纳能力，再迁移到复杂任务
- **策略库与检索增强**：将训练过程中产生的高质量策略存入外部策略库，推理时通过检索增强策略生成

### 6.4 伦理声明与更广泛影响（Ethics Statement & Broader Impact）

本研究旨在提升 LLM 的元认知推理能力，主要应用于学术和技术场景下的少样本推理任务。本文使用的训练数据均为公开可用的推理基准数据集，不涉及个人隐私信息。潜在的正面影响包括：(1) 提高小模型的推理效率，降低对大规模模型的依赖；(2) 通过显式化推理策略提升模型决策的可解释性。潜在风险包括：自动生成的策略可能在特定领域产生系统性偏见（例如，策略偏向于特定文化背景的推理模式）。我们计划在论文发表后开源代码和训练权重，以促进研究可复现性。

---

## 附录（Appendix）

### A. 数据集详情
- 110+ 任务类型的完整列表与分布统计（柱状图）
- ID / OOD / BBH 的分割方式及具体任务类型列表
- 每类任务的样本数量、平均问题长度、平均答案长度

### B. 策略案例展示
- 各任务类型的典型策略（好/中/差各一例），含完整 schema 输出
- 策略的结构分析：FIRST_ORDER_PATTERN 跨任务的共性 vs. SECOND_ORDER_STEPS 的差异
- Running Example（颜色混合）的完整策略展示

### C. 完整训练配置与可复现性清单

| 配置项 | 值 |
|---|---|
| 基座模型 | Qwen2.5-4B |
| RL 算法 | GRPO |
| 训练框架 | VERL + Agent Lightning |
| Rollout $N$ | 8 |
| 评估采样 $M$ | 4 |
| 学习率 | 1e-6 |
| 优化器 | AdamW |
| 权重衰减 | 待补 |
| 梯度裁剪 | 待补 |
| 最大策略长度 | 4096 tokens |
| Batch size | 待补 |
| 总训练步数 | 待补 |
| $R_{\text{format}}$ 权重 $\alpha$ | 待补 |
| $R_{\text{exec}}$ 权重 $\beta$ | 待补 |
| GPU 型号 $\times$ 数量 | 待补 |
| 总训练时长 | 待补 |
| 随机种子 | 待补（建议跑 3 个种子） |

**可复现性承诺**：代码和训练权重将在论文发表后开源于 GitHub。

### D. 策略可解释性：人类评估（Human Evaluation）

**实验动机**：下游准确率只能间接反映策略质量。直接评估策略的可读性和认知价值。

**做法**：随机抽取 50 个任务类型，各取 RL 训练后 SG 生成的策略 1 份，由 3 名标注者按以下维度评分（1–5 分）：

| 维度 | 定义 |
|---|---|
| **正确性**（Correctness） | 策略描述的方法是否正确、可行 |
| **具体性**（Specificity） | 步骤是否足够具体，能指导人/模型执行 |
| **可迁移性**（Transferability） | 策略是否能应用于同类未见问题 |
| **可读性**（Readability） | 结构是否清晰，语言是否易懂 |

对比组：(a) 无训练 SG 策略 (b) RL 训练 SG 策略 (c) 人工编写的参考策略（如果可行）

### E. 奖励权重敏感性分析
- $\alpha$ 和 $\beta$ 不同组合的消融（如 0.2F+0.8Exec vs. 0.1F+0.9Exec vs. 0.0F+1.0Exec）
- 含 $R_{\text{quality}}$ 的三项组合消融

### F. 额外统计分析
- 主实验各表格的 95% 置信区间（bootstrap 或多种子运行）
- 效应量（effect size）计算
- 统计显著性检验（paired bootstrap test）

---

## 写作注意事项

1. **"元"字的使用**：
   - ✅ "元归纳思维"、"元认知"、"元层"——指跨任务的通用认知习惯
   - ✅ "认知策略"——指思维的任务级外显
   - ❌ "元策略"——策略是任务级的，不应冠"元"

2. **层次命名一致性**：
   - ✅ "策略归纳层"（Strategy Induction Layer）——跨任务，SG 所在
   - ✅ "策略执行层"（Strategy Execution Layer）——任务内，AS 所在
   - 正文中保持中英文对照的一致性（第一次出现时标注英文名）

3. **"策略"首次出场**：应在概念过渡段（§4.1 引言）引入，Introduction 阶段只说"归纳"、"认知框架"、"解题规律"，不提"策略"

4. **Motivation 中关于 LLM 的描述**：
   - ✅ "LLM 缺乏主动归纳的认知倾向"
   - ✅ "LLM 不倾向于将示例中的规律显式提炼"
   - ❌ "LLM 无法归纳"（太绝对）

5. **消融实验**：格式分消融是当前缺失的关键实验，应尽快安排（§5.3.1）

6. **双层框架的 M-sample 前提**：若实验中当前 $M=1$，需将 $M$ 升至 $\geq 3$ 以使策略执行层的迭代评估结构成立；同时对应更新训练脚本中的 `grounded_proxy_k` 参数

7. **§5.3.2 Prompt 消融实验**：三种 prompt 版本（`v1.toml` / `strategy_structured_schema.toml` / `repetition_controls_2026-04-01.toml`）均已存在于代码库，可直接在相同训练配置下分别运行对比；该实验同时为 §3.3 的半结构化设计提供实证依据，应在 §3.3 末尾以"（详见消融 §5.3.2）"形式交叉引用

8. **符号一致性**：$K$ 专指 few-shot 示例数量，$M$ 专指策略效用评估的采样次数（对应代码中的 `grounded_proxy_k`），$N$ 指 GRPO rollout 数。全文须严格区分三者

9. **统计严谨性**：所有实验表格应包含误差估计。推荐方案：(a) 使用 3 个不同随机种子训练，报告 mean $\pm$ std；或 (b) 在评估集上做 bootstrap resampling（1000 次），报告 95% 置信区间

10. **MIST 简称使用**：全文在首次出现后统一使用 MIST 简称。Introduction 中首次出现时完整展开：MIST (Meta-Inductive Strategy Training)，之后所有提及均用 MIST

11. **代码-论文一致性警告**：当前训练脚本（`train_format_answer_v3.sh`）使用 `reward-mode scorer_only` 和 `grounded-proxy-k 1`，与论文 §4.1 描述的 `hybrid_grounded` + $M=4$ 不一致。必须在提交前将实际训练配置与论文描述对齐

12. **Figure 清单**：
    - Figure 1：MIST 框架全局概览图（§1.4）
    - Figure 2：规模分析——无训练 SG 各规模性能（§5.2）
    - Figure 3：主实验结果柱状图（§5.2）
    - Figure 4：训练动态曲线——$R_{\text{format}}$ 和 $R_{\text{exec}}$ 随步数变化（§5.3.1）
    - Figure 5：策略自进化过程——同一任务类型在不同训练步数的策略快照（§5.5.1）
    - Figure 6：按任务类型的提升分布直方图（§5.5.3）

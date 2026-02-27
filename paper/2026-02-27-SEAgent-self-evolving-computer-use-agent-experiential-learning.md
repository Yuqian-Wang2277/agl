# SEAgent：无需人工标注，让计算机使用 Agent 通过经验自我进化

> 论文：**SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience**  
> 作者：Zeyi Sun, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Tong Wu, Dahua Lin, Jiaqi Wang（上海交大 / 上海 AI Lab / CUHK）  
> 链接：<https://arxiv.org/abs/2508.04700>（arXiv:2508.04700 v2，2025-08-12）  
> 代码：<https://github.com/SunzeY/SEAgent>

---

## 0. 先说清楚：这篇论文在做什么

想象一下：给 AI agent 一个它从未见过的软件（比如一款专业的 3D 建模工具），**没有任何操作教程、没有人工标注数据**，它能不能通过自己不断试错，逐渐学会怎么用这个软件？

这就是 SEAgent 要解决的问题。它的核心思路是：**让 agent 在陌生软件里自主探索 → 自动评估每一步是否有效 → 从试错经验中强化学习 → 越来越熟练**。同时，它还解决了一个更难的问题：如何从多个"软件专家"进化成一个"全能通才"？

---

## 1. 摘要提炼

### 1.1 背景痛点

现有 Computer Use Agent（CUA）的瓶颈：

| 问题 | 说明 |
|------|------|
| **依赖人工标注** | 现有模型（UI-TARS、SeeClick 等）主要靠人工标注的操作 demo 训练 |
| **无法泛化到新软件** | 新软件层出不穷，每个都需要重新标注，成本极高 |
| **稀疏反馈难以学习** | 多步操作任务只有最终成败信号，中间每步的好坏未知 |
| **通才 vs 专才** | 直接训练通才比专才效果差；但训练多个专才又难以合并 |

### 1.2 SEAgent 的核心贡献

1. **World State Model（WSM）**：基于完整轨迹截图序列做步骤级奖励评估，精度接近 GPT-4o 但完全本地运行。
2. **Curriculum Generator + Software Guidebook Memory**：自动为陌生软件生成从易到难的任务，并维护动态更新的"软件使用手册"。
3. **双路学习**：GRPO 强化正确动作 + Adversarial Imitation 惩罚失败动作。
4. **Specialist-to-Generalist 策略**：先训各软件专家 → SFT 蒸馏 → 通才 RL 精调，最终超越专家集成。
5. **成果**：在 OSWorld 的 5 款专业软件上，成功率从 11.3% 提升到 **34.5%（+23.2%）**，超越所有专家集成方案。

---

## 2. 整体框架：三组件协同自进化

```
┌──────────────────────────────────────────────────────────────────┐
│                    SEAgent 自进化循环                             │
│                                                                  │
│  ┌─────────────┐  任务指令   ┌───────────────┐                  │
│  │ Curriculum  │ ──────────► │  Actor Model  │ 执行动作          │
│  │ Generator   │            │  (UI-TARS-7B) │ ──────►  截图序列 │
│  │ (Qwen2.5-  │ ◄────────── └───────────────┘          │       │
│  │    72B)     │  更新任务                              │       │
│  │  + Guidebook│                                        ▼       │
│  └─────────────┘                              ┌──────────────┐  │
│        ▲                                      │ World State  │  │
│        │  任务执行结果 + 状态变化描述          │   Model      │  │
│        └──────────────────────────────────────│ (Qwen2.5-   │  │
│                                               │   VL-7B FT) │  │
│                                               └──────────────┘  │
│                                                   │             │
│                          步骤级标注 (aT/aF)        │             │
│                                                   ▼             │
│                              GRPO (正确步) + Adversarial Imitation (失败步)│
│                                     ↓ 更新 Actor                │
└──────────────────────────────────────────────────────────────────┘
```

**三个核心组件**：

| 组件 | 模型 | 职责 |
|------|------|------|
| Actor Model π | UI-TARS-7B-DPO | 执行 GUI 操作，被持续训练的策略模型 |
| World State Model Mstate | Qwen2.5-VL-7B（微调） | 评估轨迹每步是否正确，提供步骤级 reward |
| Curriculum Generator Mtask | Qwen2.5-72B（零射推理） | 生成任务 + 维护 Software Guidebook |

---

## 3. 核心组件详解

### 3.1 World State Model（WSM）：步骤级奖励的关键

这是整个系统最关键的基础设施。**没有可靠的步骤级奖励，后续所有 RL 训练都无从谈起。**

#### 3.1.1 为什么要用"全轨迹"评估

现有奖励模型只看"最终截图"判断成败。但这有严重缺陷：

> 例如订机票任务，最终页面显示"订票成功"——但时间选错了、座位选错了，单看最后一帧根本发现不了。

SEAgent 让 WSM 看**整个轨迹的所有截图序列**，做全局判断。

#### 3.1.2 开源模型的问题 vs WSM 的解决

当直接把多张截图输入 Qwen2.5-VL-72B 时，随着截图数量增加，**精度反而下降**——因为这超出了 32K context 的优化范围，模型无法有效处理高分辨率的长截图序列。

**WSM 的解法**：从 GPT-4o 蒸馏，用 Qwen2.5-VL-7B 微调，专门针对"长轨迹截图序列"做训练：

```
训练数据构建：
1. 用 UI-TARS + Gemini-2.5-Pro 在 OSWorld 的 Chrome 环境采样轨迹
2. 用 GPT-4o 标注每条轨迹（判断 + 逐帧描述）
3. 保留 860 条 GPT-4o 判断与规则评估一致的高质量样本
4. 额外 1000 对"操作前/后"截图 + GPT-4o 描述变化（Change Description 数据）

微调：Qwen2.5-VL-7B + LoRA（rank=128），2000 步，8×A100
```

#### 3.1.3 WSM 的输出结构

WSM 对每条轨迹输出一个结构化 JSON：

```json
{
  "Correctness": true/false,       // 任务是否完成
  "Redundant": [3, 5],             // 哪些步骤是多余的（步骤号列表）
  "Optimized": true/false,         // 执行是否最优
  "First_Error_Step": 4,           // 第一个错误步骤
  "Error_Type": "clicked wrong UI element",
  "Correct_Action": "should click the Save button instead"
}
```

基于此，每步动作被动态标注为：

| 轨迹类型 | 标注规则 |
|---------|---------|
| 完全成功（无冗余） | 所有步骤 → aT（正确） |
| 成功但有冗余（步骤 k 开始冗余）| 步骤 k 之前 → aT；k 之后的冗余步骤忽略 |
| 失败（第 e 步出错）| 步骤 e 之前 → aT；步骤 e → aF（失败）|

#### 3.1.4 WSM 精度对比

| 模型 | 输入 | OSWorld Precision | OSWorld NPV |
|------|------|-------------------|------------|
| GPT-4o | 仅最后截图 (LS) | 46.3% | 88.2% |
| GPT-4o | 完整截图序列 (ES) | **74.6%** | **95.2%** |
| Qwen2.5-VL-72B | ES | 26.8% | 83.0% |
| **World State Model** | ES | **73.9%** | **90.5%** |

WSM（7B）以 ES 输入接近 GPT-4o 的水平，远超同规模的 Qwen2.5-VL-72B——说明精调的关键性。

---

### 3.2 Curriculum Generator + Software Guidebook：自进化课程

#### 3.2.1 整体设计思路

普通任务生成器（如 WebRL、NNetNav）的问题：生成的任务同质化，覆盖不了软件的全部功能。

SEAgent 的 Curriculum Generator 维护一个**持续更新的软件使用手册（Software Guidebook）**，每轮根据 agent 的执行结果不断扩展知识，生成更多样更难的任务。

#### 3.2.2 三阶段自进化课程（以 LibreOffice Impress 为例）

```
Phase 0（初始）：
  WSM 对 GUI 截图做密集标注（按钮、菜单解析）
  Curriculum Generator 生成初始任务集 I₀ + 软件手册 U₀

  I₀ 示例任务：
  - "Add a Rectangle"
  - "Type text in the first box"
  - "Add a new page"

Phase 1（初阶探索）：
  Actor 执行 I₀ 中的任务 → WSM 评估 → 记录状态变化
  例如"Add a Rectangle"后，WSM 观察到：
    "出现了矩形属性面板，包含填充色、线条、透明度等属性"
  Curriculum Generator 更新 U₁，生成更难任务 I₁：
  - "Draw a green rectangle"
  - "Draw a rectangle with 50% transparency"
  - "Create a title 'GUI RL!' and put it in the center of the page"

Phase 2（进阶操作）：
  I₂：
  - "Draw a green rectangle with 50% transparency"
  - "Create a title 'GUI RL!' with green background in the center"

Phase 3（复杂任务）：
  I₃：
  - "Create a new file"
  - "Open settings"
  - "Change the style of vscode to 'Light+'"
  - "Install python extension"
```

#### 3.2.3 Curriculum Generator 的 Prompt 设计

```
[System]
You are now a teacher training a Computer Use Agent (CUA). This CUA is 
exposed to a new software environment and undergoes multiple rounds of 
iterative training. Your task is to issue new tasks for the agent to 
explore and train on, based on the feedback from the agent's actions. 
You are also responsible for summarizing a software usage manual to 
help the agent remember knowledge about the software.

The agent has provided the following feedback on its operations 
within the software:
{action_description_list}

Here is the software usage document you summarized in the previous round:
{document}

Here is the agent's performance on the task you provided in the 
previous round:
{exam}

Please:
- Analyze the agent's performance.
- Integrate new knowledge from the feedback.
- Update the usage manual accordingly.
- Design a new set of tasks (with increased difficulty) (30 or more) 
  that reinforce the concepts the agent struggled with in the last round.
- Each task must be concise and specific, targeting a concrete atomic 
  action.
- Each task must be executable from software initial state.
- Decompose and target previous errors in a more focused way.

Output JSON format:
{
  "software_document_new": "...",
  "exam_new": [[subtask1, subtask2, ...], [task], ...]
}
```

**Prompt 设计要点**：

| 设计 | 目的 |
|------|------|
| 注入上轮 guidebook | 保持知识记忆的连续性，不重复学已会的操作 |
| 注入上轮任务评估结果（exam） | 让生成器知道 agent 哪里还不会，针对性出题 |
| 注入状态变化描述（action_description_list） | 让生成器发现软件新功能（如"发现了透明度属性"）|
| 要求 30+ 任务、从初始状态可执行 | 保证任务多样性和可执行性 |
| sequential 依赖 vs 独立任务分开列 | 区分有前置条件的任务和独立任务 |

---

### 3.3 Reinforcement Learning from Experience：双路学习

SEAgent 把每步动作分为"正确（aT）"和"失败（aF）"两类，用两种不同的损失函数分别优化。

#### 3.3.1 GRPO：鼓励正确动作

对 WSM 标记为 aT 的步骤，用 GRPO 做正向强化，reward 设计细致到每种动作类型：

```python
r(a, aT) = I[type(a) == type(aT)] + rdist(a, aT)
```

其中 rdist 针对不同动作类型分别定义，均归一化到 [0, 1]：

| 动作类型 | 距离 reward 计算方式 |
|---------|-------------------|
| click / hover | 预测坐标与真实坐标的归一化 L1 距离（越近越高）|
| drag / select | 预测框与真实框的 IoU |
| type（输入文字）| 字符级 BLEU score |
| hotkey / press / scroll | 字符级 BLEU |
| highlight | 预测框与真实框的 IoU |
| wait / finished | 固定 reward +1 |

**两项合计**：类型匹配（0 或 1）+ 精度 reward（0~1），最高 2 分。

再用 GRPO 计算组内 advantage：

```
A^(i) = (r^(i) - mean(r)) / std(r)
```

#### 3.3.2 Adversarial Imitation：惩罚失败动作

对 WSM 标记为 aF 的失败步骤，用对抗模仿损失明确"拉开"与失败动作的距离：

```
L_AI(πθ) = E[-log(πθ(a|s,I) / πref(aF|s,I))]
```

**直觉理解**：最大化当前策略与失败动作的 log-ratio，即**主动排斥**失败行为的分布。

这本质上是 DPO 损失的"负向部分"——只惩罚坏动作，不需要配对的好动作。

#### 3.3.3 综合损失

```python
L_total = L_GRPO + γ * L_AI

γ = 0.2  # 消融实验选定的最优值
```

消融对比（VSCode 成功率）：

| γ | Success Rate |
|---|-------------|
| 0.0（无 AI）| 34.8% |
| 0.1 | 36.2% |
| **0.2（最优）** | **37.7%** |
| 0.3 | 31.9% |
| 0.5 | 26.1% |
| 0.8 | 23.1% |

过大的 γ 会让模型过于保守（过度回避失败动作），反而降低性能。

---

## 4. Reward 设计汇总

SEAgent 的 reward 体系是全文最精密的部分，值得专门梳理：

### 4.1 奖励来源：World State Model（不依赖外部 API）

关键设计决策：**全程不调用 GPT-4o API**，只用本地微调的 WSM（7B）。

原因：
- 避免推理效率问题（大量 trajectory 评估）
- 开源可复现
- 精度已接近 GPT-4o（73.9% vs 74.6% Precision）

### 4.2 奖励粒度：步骤级 vs 轨迹级

| 方法 | 奖励粒度 | 问题 |
|------|---------|------|
| WebRL / DigiRL | 轨迹级（成功/失败）| 稀疏信号，难以定位哪步出错 |
| **SEAgent** | **步骤级（每步 aT/aF）** | 精准定位错误，密集反馈 |

### 4.3 奖励内容：类型 + 坐标精度

不同于只判断"有没有点击"，SEAgent 的 reward 同时检查：
1. 动作类型是否正确（click 还是 type？）
2. 坐标/文本是否精确（IoU、L1 距离、BLEU）

这让模型不只学"做什么操作"，还学"精确在哪里操作"。

---

## 5. Prompt 设计汇总

SEAgent 共有三类核心 Prompt：

### 5.1 WSM 评估 Prompt（步骤级轨迹判断）

```
I am evaluating the performance of a UI agent. The images provided are 
sequential keyframes that represent the full execution trajectory of the 
agent when attempting to follow a command.

These keyframes correspond to the instruction: [INSTRUCTION].

Please thoroughly analyze the sequence to assess the following aspects:
1. Correctness — Did the agent successfully complete the task?
2. Redundant Steps — Identify unnecessary or repeated actions.
3. Optimization — Did the agent follow an efficient plan?
4. First Error Step — If incorrect, determine the index of the first 
   keyframe where a mistake occurred.
5. Error Analysis — Brief explanation of the mistake.
6. Correct Action Suggestion — What should the agent have done instead?

Important Instructions:
- Unless the task is fully and correctly completed, set 'Correctness' 
  to False.
- Missing confirmation screens, skipped inputs, or wrong UI elements 
  clicked all count as errors.

Return your evaluation as:
{
  "Correctness": True/False,
  "Redundant": [step numbers],
  "Optimized": True/False,
  "First_Error_Step": step number or None,
  "Error_Type": "brief description",
  "Correct_Action": "what should have been done"
}
```

**设计要点**：
- 要求"除非完全正确否则标 False"——严格标准保证奖励质量
- 逐步分析：从 Correctness → 冗余 → 第一错误步骤，形成链式推理
- 结构化输出 JSON，便于程序解析

### 5.2 Curriculum Generator Prompt（任务自进化）

见 §3.2.3。核心是把"软件手册 + 上轮评估结果 + 状态变化描述"三路信息综合，生成有针对性的下一轮任务。

### 5.3 AgentRewardBench 评估 Prompt（奖励模型评测）

对 web agent 轨迹的四问评估：
- Q1：动作序列是否完成目标？（成功/未成功）
- Q2：agent 是否执行了不必要的动作？（是/否）
- Q3：执行是否最优？（完全失败/次优/较优/完全最优）
- Q4：agent 是否陷入无进展循环？（是/否）

---

## 6. Evaluation Metric 设计

### 6.1 主要指标：成功率（Success Rate, SR）

```
SR = 成功完成任务数 / 总任务数
```

每个结果取 **3 次运行均值**（减少随机性），在 5 款专业软件上分别评估：VSCode, GIMP, LibreOffice Impress, VLC, LibreOffice Writer。

### 6.2 奖励模型评估：Precision + NPV

评估 WSM 判断正确性的精度：

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **Precision（精确率）** | 判断"成功"中真正成功的比例 | 避免把失败误判为成功（过度奖励导致策略崩塌）|
| **NPV（负预测值）** | 判断"失败"中真正失败的比例 | 避免把成功误判为失败（错误惩罚导致模型退化）|

**为什么 Precision 比 Recall 更重要？**  
在 RL 训练中，错误的正向奖励（False Positive）比漏掉正向样本（False Negative）危害更大——前者会强化错误行为，后者只是浪费一些正确轨迹。

### 6.3 基准对比（OSWorld 5 软件）

| 方法 | VSCode | GIMP | Impress | VLC | Writer | 平均 |
|------|--------|------|---------|-----|--------|------|
| 人类基准 | 73.9% | 73.1% | 80.9% | 70.6% | 73.9% | 74.5% |
| GPT-4o | 4.35% | 3.85% | 6.77% | 16.1% | 4.35% | 7.08% |
| Claude 3.7 Sonnet | 18.8% | 24.4% | 10.6% | 27.5% | 17.4% | 19.7% |
| Gemini 2.5 Pro | 21.7% | 26.9% | 9.92% | 25.5% | 24.6% | 21.7% |
| UI-TARS-7B-DPO（基线）| 13.0% | 23.1% | 4.26% | 11.8% | 4.35% | 11.3% |
| DigiRL（专家 RL 集成）| 21.7% | 32.1% | 12.8% | 23.5% | 18.8% | 21.8% |
| WebRL（专家 RL 集成）| 27.5% | 29.5% | 10.6% | 25.5% | 15.9% | 21.8% |
| **SEAgent 专家 RL 集成** | **37.7%** | **38.5%** | **22.0%** | **33.3%** | **29.0%** | **32.2%** |
| SEAgent 通才 RL | 36.2% | 39.7% | 19.9% | 31.4% | 26.1% | 30.6% |
| **SEAgent 专才→通才** | **40.5%** | **42.3%** | **22.7%** | **35.3%** | **31.8%** | **34.5%** |

SEAgent 专才→通才不仅超过所有 baseline，还超越了自身的专家集成——这是最重要的结果。

---

## 7. Specialist-to-Generalist 策略：三步进化

这是论文的另一大贡献。直接训练通才效果（30.6%）比专家集成差（32.2%），但 SEAgent 设计了三步策略解决这个问题：

```
Step 1: 训练 5 个软件专家
  分别在 VSCode、GIMP、Impress、VLC、Writer 上运行 SEAgent
  每个专家独立学习该软件的操作策略

  ↓

Step 2: 专家知识蒸馏（SFT）
  收集所有专家在各自领域的成功轨迹（共 3500 条）
  用 SFT 把这些成功轨迹微调到 base 模型 UI-TARS-7B
  得到一个"懂得多但不精通"的初始通才

  ↓

Step 3: 通才 RL 精调
  在全部 5 个软件上运行 SEAgent RL
  通才从 SFT 初始化出发，在多软件环境中强化学习
  得到最终的通才模型（34.5% > 32.2% 专家集成）
```

**为什么这比直接训通才更好？**

SFT 蒸馏给了通才：
- 各软件的基础操作知识（commonsense）
- 多样化的推理和规划模式
- 更好的初始化，让后续 RL 有更好的起点

这与 R-Zero 论文中"Sequential 策略（先自进化再 SFT）最优"有异曲同工之妙。

---

## 8. 消融实验

### 8.1 各组件消融（VSCode 成功率）

| Reward Model | SFT(BC) | GRPO | Adversarial Imitation | VSCode SR |
|-------------|---------|------|-----------------------|-----------|
| Qwen2.5-VL-72B | ✓ | | | 10.1% |
| Qwen2.5-VL-72B | | ✓ | | 11.6% |
| **World State Model** | | ✓ | | **23.2%** |
| World State Model | ✓ | ✓ | | 30.4% |
| World State Model | | ✓ | ✓ | **34.8%** |
| **World State Model** | | ✓ | **✓** | **37.7%** |

**关键结论**：
1. WSM 对比 Qwen2.5-VL-72B：**+11.6%**——高质量 reward 是最大的性能瓶颈
2. GRPO alone（23.2%）→ 加入 SFT（30.4%）→ 去掉 SFT 加 AI（34.8%）→ 完整（37.7%）
3. Adversarial Imitation 独立带来的增益比 SFT 更大（+11.6% vs +7.2%）

### 8.2 超参灵敏度

| 生成任务数 | VSCode SR | 状态变化描述数 | VSCode SR |
|---------|-----------|-------------|-----------|
| 30 | 31.88% | 30 | 33.33% |
| 50 | 36.23% | 50 | 37.68% |
| **100** | **37.68%** | **100** | **37.68%** |
| 200 | 37.68% | 200 | 34.78% ↓ |

- 任务数从 50 到 100 有提升，100 以上不再提升（多样性饱和）
- 状态描述数 100 最优；200 反而下降（上下文过长，LLM 质量下降）

---

## 9. 局限性

| 局限 | 说明 |
|------|------|
| **依赖 WSM 奖励信号** | 系统质量上限受 WSM 精度约束，还不是真实环境的直接奖励 |
| **任务相对简单** | 当前测试的任务（<20 步）与专家小时级工作流差距很大 |
| **仅 GUI 类软件** | 文本/代码类软件（LaTeX、Lean）不适用当前框架 |
| **视觉模态依赖** | 截图序列处理的计算开销大 |

---

## 10. 算法伪代码

```python
# SEAgent 专家自进化主循环（Algorithm 1）

# 初始化
C0 = WSM.caption_GUI(initial_screenshot)      # 解析 GUI 结构
I0, U0 = Curriculum.generate(C0)              # 生成初始任务集 + 软件手册

for phase in range(P):                        # P=3 阶段
    Dtraj = []
    
    # 2.1 自主探索
    for task in I_phase:
        trajectory = Actor.execute(task)       # Actor 执行任务
        
        # 2.2 效果评估
        judgment, state_captions = WSM.evaluate(trajectory)
        Dtraj.append((trajectory, judgment, state_captions))
    
    # 2.3 策略更新
    D_pos = [steps labeled aT from Dtraj]     # 正确步骤
    D_neg = [steps labeled aF from Dtraj]     # 失败步骤
    
    # GRPO on 正确步骤
    for (s, aT) in D_pos:
        r = type_match(a, aT) + rdist(a, aT)  # 类型+精度 reward
    L_GRPO = GRPO_loss(D_pos, rewards)
    
    # Adversarial Imitation on 失败步骤
    L_AI = mean([-log(π(a|s,I) / πref(aF|s,I)) for (s,aF) in D_neg])
    
    # 综合更新
    L_total = L_GRPO + 0.2 * L_AI
    Actor = update(Actor, L_total)
    
    # 2.4 任务更新：基于执行结果生成更难任务
    I_next, U_next = Curriculum.evolve(U_phase, I_phase, judgments, captions)

# 输出：专精于目标软件的 Actor Policy
```

---

## 11. 结论总结

| 维度 | 核心结论 |
|------|---------|
| **核心命题** | CUA 可以在无任何人工标注的情况下，通过自主探索陌生软件学会操作 |
| **关键创新 1** | World State Model：全轨迹输入的步骤级奖励，精度接近 GPT-4o |
| **关键创新 2** | Curriculum Generator + Software Guidebook：自维护的软件知识 + 自适应课程 |
| **关键创新 3** | GRPO + Adversarial Imitation 双路学习：正向强化 + 负向排斥 |
| **关键创新 4** | Specialist-to-Generalist：专家蒸馏 → 通才 RL，最终超越专家集成 |
| **数字成果** | UI-TARS-7B 成功率 11.3% → 34.5%（+23.2%），超越所有 baseline |
| **已知瓶颈** | 依赖 WSM 精度；任务相对简单；仅 GUI 软件适用 |

---

## 12. 对我们项目（Agent-Lightning 策略提取）的启发

我们的项目核心是：**从 few-shot 示例中提取解题策略，再应用策略解决新问题**。SEAgent 解决的是"如何让 agent 在陌生环境中自主学习"——两者都面临"如何在无充分先验知识下产生有效的学习信号和学习内容"的共同挑战。

### 12.1 Software Guidebook Memory → 策略知识库

SEAgent 最核心的创新之一是**动态累积的 Software Guidebook**：agent 每次执行一个任务，就从 WSM 的分析结果中提炼出新知识（"发现了矩形有透明度属性"），更新到 Guidebook，下轮 Curriculum Generator 据此生成更复杂的任务。

**对我们项目的类比**：
- 我们的"策略库"（extracted strategies）类似于这里的 Software Guidebook
- 每次从 few-shot 示例中提取策略后，应该**主动分析这条策略"解决了哪类问题的哪个子结构"**，并更新到策略知识库中
- 下次遇到类似结构的新题时，优先调用知识库中已有的相关策略，而非重新提取
- 这形成一个"策略积累 → 策略应用 → 策略补充"的正向循环

### 12.2 Adversarial Imitation → 从失败策略中学习

SEAgent 不只学成功案例，还明确**惩罚失败动作**（Adversarial Imitation），效果甚至优于单纯 SFT 成功轨迹。

**对我们项目的直接启示**：
- 当某条策略被提取出来但**应用失败**（模型按照策略推理得到了错误答案），不应简单丢弃
- 可以设计类似的"对抗损失"：让模型的策略选择分布主动**远离这条坏策略**
- 具体形式：对（题目, 坏策略）对施加负向 KTO/DPO 信号，明确告诉模型"遇到这类题不要选这种策略"

### 12.3 步骤级 Reward → 策略内部步骤的质量评估

WSM 能定位到轨迹中**第一个出错的步骤**，而不只说"整体失败"。这大大提升了 reward 的精度。

**对我们项目的类比**：
- 我们的策略是多步推理过程，一条策略通常包含"识别题型 → 提取关键条件 → 构造解题框架 → 推导答案"等子步骤
- 当最终答案错误时，应该尝试定位是**哪个子步骤出了问题**（策略识别错？条件提取错？推导错？）
- 这等价于 SEAgent 的 First_Error_Step 定位，可用小模型或规则对中间推理步做细粒度判断
- 粒度更细的错误信号 → 更精准的训练信号 → 更快收敛

### 12.4 Curriculum Generator 的课程思路 → 策略难度递进

SEAgent 的课程从"Add a Rectangle"进化到"Create a title with green background centered on page"，难度递进有明确规律：操作复杂度 + 组合性 + 依赖关系。

**对我们项目的启示**：
- 训练策略提取模型时，可以设计类似的难度递进课程：
  1. **Phase 1**：单步策略（e.g., 等差数列求和公式直接套用）
  2. **Phase 2**：两步组合策略（e.g., 先识别数列类型，再选公式）
  3. **Phase 3**：多步复合策略（e.g., 分情况讨论 + 多个公式嵌套）
- 每阶段根据模型当前的"策略提取成功率"自动决定是否进入下一阶段

### 12.5 Specialist-to-Generalist → 题目类型专家再泛化

SEAgent 的三步策略（专家训练 → SFT 蒸馏 → 通才 RL）可以直接启发我们的训练范式：

```
我们项目的类比：

Step 1: 训练各题型专家
  分别在数学/代码/逻辑推理等不同题型上训练专门的策略提取模型

Step 2: 专家策略蒸馏
  收集各专家提取的高质量策略样本，SFT 到通用策略提取模型

Step 3: 通才策略 RL 精调
  在混合题型上做 RL，让通才在多题型间泛化策略提取能力
```

**预期收益**：专题训练确保各领域策略质量，蒸馏保留专业知识，最终通才 RL 融合跨域泛化——比直接混合训练效果更好（SEAgent 实验验证了 34.5% > 30.6%）。

### 12.6 完整轨迹评估 → 策略应用过程的全程监控

WSM 看**完整轨迹**而非只看最终答案，发现了很多"最终看起来正确但过程有问题"的情况（如订票最终成功但日期选错）。

**对我们项目的警示**：
- 仅检查最终答案正确与否是不够的——即使答案对，如果是"歪打正着"（没有通过策略真正推导出来）
- 应该同时评估**策略的应用过程**是否符合策略的结构（类似 WSM 检查动作序列的合理性）
- 这要求设计一个能评估"推理链与策略一致性"的 Process Reward Model

---

### 一句话总结：这篇论文对我们项目的意义

> **SEAgent 的"自进化软件手册 + 步骤级精准 reward + 失败动作对抗惩罚 + 专才→通才蒸馏"四件套，分别对应我们项目的"策略知识库积累"、"策略子步骤质量评估"、"坏策略主动排斥"和"分题型专家训练再泛化"，其中最具参考价值的是：把成功/失败信号精细化到步骤级，以及用 Adversarial Imitation 显式排斥已知坏策略，而非只从成功案例中学习。**

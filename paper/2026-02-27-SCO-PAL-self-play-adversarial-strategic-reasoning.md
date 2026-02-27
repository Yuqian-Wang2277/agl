# SCO-PAL：通过自博弈在对抗游戏中提升语言 Agent 战略推理能力

> 论文：**Enhancing Language Agent Strategic Reasoning through Self-Play in Adversarial Games**  
> 作者：Yikai Zhang, Ye Rong, Siyu Yuan, Jiangjie Chen, Jian Xie, Yanghua Xiao（复旦大学 / ByteDance Seed / OSU）  
> 链接：<https://arxiv.org/abs/2510.16761>（arXiv:2510.16761，2025-10-19）

---

## 0. 先说清楚：这篇论文在做什么

这篇论文不是做通用 Agent 训练，而是专注在 **对抗性博弈游戏**（adversarial games）场景下提升 LLM agent 的**战略推理**能力。核心问题是：**一个 LLM agent，能否通过和对手玩游戏这件事情本身，来自我提升战略水平——而不依赖专家标注数据？**

他们给出的答案：可以。而且"和自己玩"（self-play）是最有效的方式。

---

## 1. 摘要提炼

### 核心痛点

现有 LLM agent 在动态对抗游戏中表现差，根本原因是**战略推理能力不足**。已有两条路：

| 路径 | 方法 | 问题 |
|------|------|------|
| 路径 A | 模仿专家数据（imitation learning） | 专家数据获取成本高、难以规模化 |
| 路径 B | Play-and-Learn（交互学习） | 现有工作忽略了**对手选择**对学习效果的关键影响 |

### 核心贡献

1. **SCO-PAL 框架**：Step-level poliCy Optimization through Play-And-Learn，通过大规模游戏交互 → 步骤级 reward 估计 → 策略精炼，三阶段提升 agent 战略能力。
2. **对手选择的系统性分析**：第一次在 LLM agent 领域系统研究了"和谁玩"对学习效果的影响，并定量证明 **self-play 是最优选择**。
3. **效果**：在 6 个对抗游戏上，对比 4 类对手，平均胜率提升约 **30%**；对战 GPT-4 达到 **54.76%** 胜率。

---

## 2. 问题建模

### 2.1 游戏环境（MDP 形式化）

对抗游戏被建模为一个情节式马尔可夫决策过程（episodic MDP）：

```
状态 s：当前游戏局面
观察 o：玩家 i 可见的部分信息
动作 a：语言 agent 输出的自然语言行动
转移函数 T: (S, A) → S
终局 reward r：赢(+1) / 输(-1) / 平(0)
```

每个 agent 的目标是优化自身策略 πᵢ，最大化累积 reward。值得注意：LLM agent 的"动作空间"是**高维自然语言**，和传统 RL 的离散/连续控制有本质区别。

### 2.2 为什么对手选择是个大问题

在传统 RL 中，对手选择（self-play、curriculum learning 等）已是成熟议题。但在 LLM 场景下，额外复杂性来自：

- **太强的对手**：策略分布偏移过大，agent 难以生成有效学习信号
- **太弱的对手**：胜负可预测，动作多样性极低，策略陷入局部最优
- **不均衡**：不同对手强弱导致有利/不利动作比例严重失衡，学习不稳定

---

## 3. SCO-PAL 框架详解

整个框架分三个阶段：

```
Stage I ──► Stage II ──► Stage III
游戏交互    步骤级 reward 估计    策略精炼（BC + KTO）
```

### 3.1 Stage I：Game Interaction（游戏交互）

两个 agent 在游戏环境中对弈，自动积累大量轨迹：

```
轨迹 τ = (s₀, a₁, s₁, a₂, ..., aₙ, sₙ, r)
收集轨迹集合 T = {τ₁, τ₂, ..., τₙ}
```

游戏环境自动判断胜负，**无需任何人工标注**。这是"不依赖专家数据"的关键。

### 3.2 Stage II：Step-Wise Reward Estimation（步骤级 Reward 估计）

这是全文最关键的创新点之一。环境只给终局 reward，但每一步动作的好坏完全未知。作者用 **Monte Carlo 估计**来解决：

```
对每个 (state, action) 对 (sᵢ, aᵢ)，
在所有轨迹 T 中统计：
  Nₐₗₗ = (sᵢ, aᵢ) 出现的总次数
  Nwᵢₙ = (sᵢ, aᵢ) 出现后最终获胜的次数

步骤级 reward：r(sᵢ, aᵢ) = Nwᵢₙ / Nₐₗₗ
```

**通俗理解**：统计在局面 s 采取动作 a 之后，最终赢了多少次。赢得越频繁，这个动作就越"有利"（advantageous）。

作者还对比了三种 reward 估计方式（见消融实验 §7.2）：

| 方法 | 说明 | 平均胜率 |
|------|------|---------|
| Win Rate（本文） | 直接统计胜率 | **50.08%** |
| Discounted Reward | 折扣因子加权 γ=0.8 | 48.66% |
| Beta 分布估计 | 贝叶斯胜率估计，先验 α₀=β₀=1 | 36.77% |

Win Rate 胜出的原因：直接对齐游戏目标，稳定可解释，不需调 γ 等超参。

### 3.3 Stage III：Strategy Refinement（策略精炼）

得到步骤级 reward 后，用**两阶段**训练精炼策略：

**阶段 A：Behavioral Cloning（BC）**

筛选 reward > 阈值 δ（=0.5）的有利动作，做有监督微调：

```python
# 只取高质量动作做 BC
advantageous_data = [(s, a) for (s, a) in T if r(s, a) > δ]
loss_BC = -E_{(s,a)~D} [log π_θ(a|s)]
```

BC 的作用是让模型**快速适应游戏环境**，建立一个合理的初始化基础。

**阶段 B：KTO Optimization**

Kahneman-Tversky Optimization（KTO，Ethayarajh et al., 2024）是一种不需要 pair-wise 对比数据的偏好优化方法，只需要每条样本的"是否可取（desirable/undesirable）"二元标签：

```
r_θ(x, y) = log(π_θ(y|x) / π_ref(y|x))      # 相对于参考模型的 log ratio

z₀ = KL(π_θ || π_ref)                          # KL 散度基准

v(x, y) = {
  λ_D · σ(β · (r_θ - z₀))    如果 y 是 desirable
  λ_U · σ(β · (z₀ - r_θ))    如果 y 是 undesirable
}

L_KTO = E[(λ_y - v(x, y))]
```

把 reward > δ 的 (s, a) 标记为 desirable，< δ 的标记为 undesirable，让模型倾向选有利动作，回避不利动作。

**为什么用 KTO 而不是 DPO？**

DPO 需要对同一个 state 下生成 pair-wise 比较，在博弈场景中难以构建（不同时刻的局面千变万化）。KTO 只需单条样本 + 二元标签，适配性更强，且实验证明胜率高出 DPO **9.56%**。

**消融：两阶段 vs 单阶段**

| 训练策略 | 说明 | 平均胜率 |
|---------|------|---------|
| Direct KTO | 直接 KTO，无 BC | 44.89% |
| Joint Loss | BC + KTO 联合损失 | 45.49% |
| Two-Stage（本文） | BC → KTO 串行 | **50.08%** |

先 BC 后 KTO 优于联合训练，说明 BC 提供了更稳定的初始化，再做偏好优化效果更好。

---

## 4. Prompt 设计

论文对 6 个游戏分别设计了 **系统提示（System Prompt）+ 游戏提示（Game Prompt）+ 步骤提示（Step Prompt）**，形成三层结构。

### 4.1 系统提示

```
You are a powerful gaming agent who can make proper decisions to beat 
the user in gaming tasks. You are a helpful assistant that strictly 
follows the user's instructions.
```

角色定位明确：竞技博弈 agent，目标是"赢"。

### 4.2 游戏提示（以 Nim 为例）

```
In Nim, a strategic game with a set of four piles containing 1, 3, 5, 
and 7 matches respectively, players aim to avoid taking the last match. 
During each turn, a player may take any number of matches from a single 
pile, but must take at least one and cannot exceed the number remaining 
in that pile. The objective is to force the opponent to pick up the 
final match, thereby winning the game. The action is presented in 
<pile:x, take:y>, which means take y match(es) from the x-th pile.
```

规则描述 + 动作格式说明，让模型理解游戏结构。

### 4.3 步骤提示（以 Nim 为例）

```
Currently, the 1st pile has <PILES[0]> match(es), the 2nd pile has 
<PILES[1]> match(es), the 3rd pile has <PILES[2]> match(es), 4th pile 
has <PILES[3]> match(es).
```

动态注入当前局面信息（状态观察），让模型做出当步决策。

### 4.4 Prompt 设计小结

| 层级 | 内容 | 作用 |
|------|------|------|
| System Prompt | 角色定位（竞技 agent） | 固定基本行为模式 |
| Game Prompt | 游戏规则 + 动作格式 | 让模型理解结构约束 |
| Step Prompt | 当前局面状态（动态） | 提供决策所需观察信息 |

---

## 5. Reward 设计

这是全文另一个核心设计点，值得专门拆解：

### 5.1 挑战：稀疏终局 reward

游戏只在结束时给 reward（赢/输/平），中间每一步动作没有即时奖励信号——这是典型的**信用分配（credit assignment）**问题。

### 5.2 解决方案：Monte Carlo 胜率

```
r(s, a) = N_win / N_all
```

用大量交互轨迹统计经验胜率。胜率 > 0.5 为"有利动作"，< 0.5 为"不利动作"。

### 5.3 与 SPAG 对比

竞品 SPAG 用了**手工折扣 reward**：

```
r(sₜ, uₜ) = (1-γ)γ^{T-t} / (1-γ^{T+1})  如果 player1 赢
           = -r(sₜ, uₜ)                    如果 player2 赢
```

问题：γ 需要手动调整，且对不同游戏难以通用。SCO-PAL 的胜率估计无超参依赖，更通用、更稳定。

---

## 6. 评测指标（Evaluation Metrics）设计

### 6.1 Win Rate（胜率）

```
w = (N_win + 0.5 × N_tie) / (N_win + N_lose + N_tie)
```

平局计 0.5 分，既公平又直观，直接对齐游戏目标。

### 6.2 对手矩阵（四类评测对手）

| 对手 | 描述 |
|------|------|
| Random | 完全随机选动作（弱基线） |
| MCTS(1000) | 1000 次模拟的蒙特卡洛树搜索（强符号 AI） |
| GPT-3.5 | OpenAI GPT-3.5-turbo-0125 |
| GPT-4 | OpenAI GPT-4-turbo-2024-04-09 |

### 6.3 泛化评测（Unseen Games）

在 3 个训练时未见过的游戏（Blind Auction、Pig、Prisoner's Dilemma）上测试，评估战略能力的迁移性。

### 6.4 Regret 分析

在 Nim 和 Tic-Tac-Toe 上计算 regret（距最优策略的差距），验证模型向博弈均衡逼近。

---

## 7. 对手选择分析：为何 Self-Play 最优

这是本文最有价值的分析实验，值得重点拆解。

### 7.1 实验设置

- 对手范围：从 Random（弱）到 MCTS(1000)（强），共 8 种级别
- Self-Play 单独作为一类
- 分析两个维度：(a) 数据量和有利/不利动作比 (b) 训练后胜率

### 7.2 关键发现

**发现 1：数据多样性与对手强度呈倒 U 形关系**

- 对手太弱（Random）：agent 总赢，不利动作极少，数据单调
- 对手太强（MCTS-1000）：agent 总输，有利动作极少，同样单调
- **Self-Play**：对战双方实力相当，有利/不利动作比例最均衡，数据最丰富

**发现 2：数据量 ≠ 数据质量**

实验对低质量数据进行两种上采样：
- Scale 1：等比例扩充总量 → 胜率**下降**（过拟合低质量模式）
- Scale 2：强制对齐有利/不利数量 → 胜率进一步**下降**（打乱优化轨迹）

结论：**靠数据堆量无法弥补质量缺失，多样性和平衡性必须来自数据本身。**

**发现 3：Self-Play 训练效果最优**

各对手训练后平均胜率：

```
Self-Play > MCTS(5) > MCTS(10) > ... > MCTS(1000) > Random
```

越均衡的对手，训练后效果越好。

### 7.3 直觉解释

Self-play 是一种隐式的 **curriculum learning**：随着模型变强，对手（即模型自身）也在同步变强，保持"刚好合适的难度"，持续提供有效学习信号。

---

## 8. 实验结果

### 8.1 主实验（6 游戏 × 4 对手）

| 方法 | vs GPT-4 | vs GPT-3.5 | vs Random | vs MCTS | 平均 |
|------|---------|-----------|---------|---------|------|
| Base | 28.40% | 42.09% | 52.44% | 5.75% | 32.17% |
| BC | 42.58% | 53.52% | 67.68% | 10.22% | 43.50% |
| SPAG | 40.31% | 50.51% | 65.98% | 10.00% | 41.70% |
| **SCO-PAL** | **54.76%** | **64.84%** | **70.15%** | **10.57%** | **50.08%** |

SCO-PAL 平均胜率 50.08%，比 Base 提升 **+17.91%**，比最好 baseline 提升 **+6.58%**，对战 GPT-4 的胜率从 28.40% 提升到 **54.76%**。

### 8.2 Head-to-Head 对战（方法间互相比赛）

| 方法 | vs Base | vs BC | vs SPAG | vs SCO-PAL | 平均 |
|------|---------|-------|---------|-----------|------|
| SCO-PAL | **77.03%** | **54.71%** | **52.69%** | 50.00% | **58.61%** |

SCO-PAL 全面压制其他所有方法。

### 8.3 泛化能力（Unseen Games）

| 游戏 | Base | SCO-PAL |
|------|------|---------|
| Blind Auction | 71% | 67% |
| Pig | 88% | **92%** |
| Prisoner's Dilemma | 37% | **60%** |
| 平均 | 65% | **73%** |

未见游戏平均提升 **+8%**，说明 SCO-PAL 提升的是通用战略推理能力，不只是针对特定游戏的记忆。

### 8.4 通用能力保持（MMLU）

| 模型 | MMLU 准确率 |
|------|------------|
| Base | 70.54% |
| SCO-PAL | 70.65% |

战略训练对通用能力几乎无损耗，MMLU 几乎持平。

---

## 9. 消融实验总结

| 维度 | 最优设置 | 次优设置 | 结论 |
|------|---------|---------|------|
| 训练策略 | Two-Stage (BC→KTO) 50.08% | Direct KTO 44.89% | BC 先适应环境，KTO 再优化偏好 |
| BC 数据来源 | Reward-Based 过滤 46.87% | Trajectory-Based 43.50% | 成功轨迹包含次优动作，需精筛 |
| 优化方法 | KTO 50.08% | DPO 40.52% | KTO 无需 pair 比较，更灵活 |
| Reward 估计 | Win Rate 50.08% | Discounted 48.66% | 胜率直接对齐目标，最稳定 |
| 对手选择 | Self-Play | MCTS(5) | 均衡最关键 |
| 温度 | 0.7 | 0.5 / 1.0 | 稳定性和多样性的平衡 |

---

## 10. 迭代训练（Iteration）

实验还测试了多轮迭代 SCO-PAL：

```
Iter1: Base 自博弈 → SCO-PAL → 得到模型 M1（50.08%）
Iter2: M1 vs Base → SCO-PAL → M2（50.77%）
Iter3: M2 vs M1 → SCO-PAL → M3（48.93%）
```

Iter2 略有提升，但 Iter3 开始下降——RL 多轮迭代的过拟合风险显现。这与 RLHF/RLAIF 中普遍观察到的"RL 训练轮数饱和"一致。

---

## 11. 局限性

论文诚实地指出了三点局限：

1. **游戏类型受限**：只做了回合制、规则明确的符号博弈，不涉及开放式/部分可观测环境（如谈判、网页操作）。
2. **对手多样性有限**：只用了脚本 agent 和 LLM 变体，未引入人类对手或风格多样化 LLM。
3. **迭代轮数有限**：多轮 SCO-PAL 存在过拟合风险，需要额外机制（如 replay buffer、对手池等）。

---

## 12. 结论总结

| 维度 | 核心结论 |
|------|---------|
| **最核心发现** | Self-play 是对抗游戏中 play-and-learn 的最优对手策略 |
| **为什么** | 保证有利/不利动作均衡分布，提供持续有效的学习信号 |
| **算法贡献** | 三阶段 SCO-PAL 框架：大规模交互 + MC 胜率估计 + BC/KTO 两阶段精炼 |
| **Reward 设计** | MC 胜率估计（无超参，直接对齐目标）优于手工折扣 reward |
| **优化选择** | KTO > DPO（无需 pair 数据，step-level 覆盖更广）|
| **实验结论** | +30% 平均胜率，54.76% 战胜 GPT-4，泛化到未见游戏 |

---

## 13. 对我们项目（Agent-Lightning 策略提取）的启发

我们的项目思路是：**从 few-shot 示例中提取解题策略，再将策略应用于新问题**。这与 SCO-PAL 的出发点（"如何产生更优质的策略数据"）有深层共鸣，以下是几个具体启发点：

### 13.1 步骤级 Credit Assignment（最直接的借鉴）

SCO-PAL 的核心创新之一是把终局 reward 拆解到每一步动作。我们的项目同样面临类似问题：**从 few-shot 示例中提取出来的策略，在解新题时每个推理步骤的贡献是什么？**

借鉴方向：
- 可以参考 Monte Carlo 胜率估计的思路，对策略的每个关键子步骤（如"识别问题类型" → "选取策略" → "执行推理"）用统计方式估计各子步骤对最终答题正确率的贡献。
- 这有助于在训练中对"好策略步骤"和"坏策略步骤"差异化加权。

### 13.2 数据质量 > 数据数量（策略提取的关键警示）

SCO-PAL 的对手实验证明：**简单扩充低质量数据（数量翻倍）不仅无效，还会降低性能。** 有利/不利动作的均衡性和多样性才是核心。

对我们项目的启示：
- 从 few-shot 示例中提取策略时，应优先保证策略的**覆盖多样性**（不同题型、不同解题路径），而不只是堆数量。
- 若某类策略在样本中过度集中（类比"和弱对手玩"产生的数据单调性），需主动下采样或加入对比性负样本。

### 13.3 两阶段精炼思路（BC 先适应，偏好优化再提升）

SCO-PAL 证明了 BC → KTO 两阶段训练优于直接偏好优化。原因是 BC 先建立"领域知识的初始化"，KTO 再做行为的精细区分。

对我们项目的类比：
- **第一阶段**：用提取到的策略做 SFT，让模型先"学会"用策略框架思考（适应策略格式和解题流程）。
- **第二阶段**：再用策略质量的偏好信号（如基于最终答题正确率）做偏好优化，区分"好策略" vs "坏策略"。

### 13.4 Self-Play → 策略间"对抗"数据合成

SCO-PAL 的 self-play 机制自动产生多样且均衡的训练数据。对我们项目的延伸思考：

- 可以构造**策略对比场景**：同一道题，用不同策略解答，比较哪种策略更有效。这等价于 self-play 中"有利 vs 不利动作"的对比。
- 这种策略间的"竞争"可以产生高质量的偏好对（preference pairs），用于后续 KTO/DPO 训练。

### 13.5 Reward 设计直接参考

SCO-PAL 用答题胜率（win rate）作为步骤级 reward，简洁有效。我们项目可以：
- 用**策略应用成功率**（用某策略解题的最终正确率）作为策略质量的度量。
- 在提取到多条候选策略时，用成功率对策略排序，产生策略偏好数据。

---

### 一句话总结这篇论文对我们项目的意义

> **SCO-PAL 证明了"无标注数据、通过自我交互学习"的可行性，其步骤级 MC 胜率估计 + BC/KTO 两阶段精炼的设计范式，可以直接迁移到我们"策略质量评估 → 策略偏好优化"的训练流程中：用策略应用成功率作为步骤级 reward，BC 先适配策略格式，KTO 再区分好坏策略，形成低成本的策略自改进闭环。**

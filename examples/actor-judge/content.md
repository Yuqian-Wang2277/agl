
# 🚀 自进化 Agent 代码实现工程蓝图

## 1. 全局架构与模型选型配置 (Global Architecture)

在动代码前，我们需要在框架中初始化并管理三个核心模型实例。

* **Actor Model ($\pi_\theta$) & Reference Model ($\pi_{SFT}$)**
    * **基座模型**：选用 Qwen3-4B。选择它的原因是其在长文本推演（CoT）上表现优异，避免了 BERT 类模型 512 tokens 的截断问题 。
    * **Reference Model**：由 Phase I SFT 训练后的 Actor 副本初始化，在 Phase II 中**完全冻结参数**，仅在推理时做 Forward 计算，用于产出 Log-Prob 从而计算 KL 散度 。
* **Judge Model ($V_\phi$)**
    * **初始化策略（关键 Trick）**：绝对不能随机初始化！必须使用 Phase I SFT 训练好的 Actor 权重来初始化 Judge 。因为 SFT 后的 Actor 内部的 Attention 已经学会了什么是好的推理步骤 ，这保证了 Judge 和 Actor 的“语义空间对齐” 。
    * **网络结构魔改**：在代码中，需要将 Qwen3-4B 最后一层 `LM Head`（输出维度为词表大小）截断，替换为一个 `Scalar Head`（例如 `nn.Linear(hidden_size, 1)`） 。前向传播的最后需加上 Sigmoid 激活函数，确保输出 $v \in [0,1]$。

## 2. 核心数据模块：Reflective Experience Replay Buffer

这是区别于传统 RL 的核心数据结构，不能用简单的 FIFO 队列 。

* **数据结构设计**：
    * 建议在 Python 中实现一个以问题 $Q$ 的 ID 为 Key 的字典嵌套结构：`Dict[Q_id, List[Dict]]`。
    * 每一条经验记录必须包含：问题 $Q$、策略 $S$、环境真实反馈 $y \in \{0, 1\}$ 。注意：不需要存储答案 $A$，因为解耦假设下 $y$ 的成败主要归因于 $S$ 。
    * **必须追踪的动态属性**：
        1.  `n_i`：被 Judge 采样的历史次数（初始为 0） 。
        2.  `v_pred`（即 $\hat{y}_i$）：Judge 对该策略的最新预测概率 $\sigma(V_\phi(Q,S))$（初始可设为默认值） 。
        3.  加入时间戳或 step 索引，用于在队列超过 $N_{max}$ 时淘汰老数据 。

## 3. Phase I: 能力冷启动 (SFT 代码实现)

本阶段的难点在于实现**自定义的 DataCollator** 以支持复杂的 Mask 机制 。

* **数据格式**：严格要求模型按照 `<Strategy>...</Strategy> <Answer>...</Answer>` 格式输出 。
* **双重 Loss 掩码构造**：
    需要修改 `labels` tensor（通常将不需要计算 Loss 的 token ID 设为 `-100`）：
    1.  **$L_{gen}$ (策略生成损失)**：学习 $P(S|Q)$。在 `labels` 中，将 Prompt ($Q$) 和 Answer 部分全部设为 `-100`，**仅保留 Strategy 部分**的 token ID 计算交叉熵 。
    2.  **$L_{exec}$ (策略执行损失)**：学习 $P(A|Q,S)$（Teacher Forcing）。在 `labels` 中，将 Prompt ($Q$) 和 Strategy ($S_{gold}$) 部分设为 `-100`，**仅保留 Answer 部分**的 token ID 。
* **最终 Loss**：$L_{SFT}(\theta) = \lambda_1 \cdot L_{gen} + \lambda_2 \cdot L_{exec}$ 。代码中需对外暴露 $\lambda_1, \lambda_2$ 超参。

## 4. Phase II: 协同进化 RL (Main Loop 代码实现)

这是整个框架的引擎，需要严格按照以下步骤编写训练循环。

### Step 0: Judge Warm-up (代码可选项)
* **实现逻辑**：使用 Phase I 的 $D_{SFT}$ 数据集 。将 $S_{gold}$ 视为正样本，随机采样或扰动 $S$ 视为负样本 。单独写一个微调脚本，先训 Judge 几个 Epoch ，防止初始阶段与 Actor 互不收敛（瞎猜） 。

### Step 1: Rollout (探索与采样)
* **生成阶段**：对于 Batch 内的 $Q$，Actor 进行 $K$ 次采样，生成 $S_k$ 和 $A_k$ 。
* **沙盒评测**：调用 $Env$ 模块，执行 $y_k = Env(A_k) \in \{0, 1\}$ 。
* **入库**：将 $(Q, S_k, y_k)$ 存入 UCB Buffer 。

### Step 2: ODVA (Judge 模型更新) 这是代码实现中最复杂的数据抽取与 Loss 计算环节。
* **UCB 权重计算**：在采样前，遍历 Buffer 计算每条数据的优先级分数：
    $Score(i) = \lambda_{err} \cdot |y_i - \sigma(V_\phi(Q_i, S_i))| + \lambda_{exp} \cdot \sqrt{\frac{\ln T}{n_i}}$
    *(代码细节：分母 $n_i$ 要加一以防除零报错；$T$ 是全局总采样次数 )*
* **Pairwise 采样**：基于上述 $Score(i)$ 进行加权随机采样 。为了计算 Loss，必须保证针对同一个 $Q$，抽取出至少一对 $(S_{win}, S_{lose})$（即一个 $y=1$，一个 $y=0$） 。
* **Bradley-Terry Loss 计算**：
    将 $Q$ 分别与 $S_{win}$ 和 $S_{lose}$ 拼接输入 Judge。
    $$L_{Judge}(\phi) = - \mathbb{E} [\log \sigma(V_\phi(Q,S_{win}) - V_\phi(Q,S_{lose}))]$$ 
* **状态回写**：完成更新后，务必将 Judge 当前给出的最新打分 $\sigma(V_\phi)$ 更新回 Buffer 的 $v\_pred$ 字段，并将被抽样数据的 $n_i$ 加 1 。

### Step 3 & 4: 奖励计算与 GRPO Actor 更新 * **Reward 组合 (逐样本计算)**：
    $$R(S_k) = (1-\alpha) \cdot y_k + \alpha \cdot \sigma(V_\phi(Q,S_k)) - \beta \cdot D_{KL}(\pi_\theta || \pi_{SFT})$$ 
    *(代码细节：$y_k$ 是硬标签 0/1 ；$\sigma(V_\phi)$ 是 Judge 的稠密信号 ；KL 惩罚使用 Actor 和 Reference 的 Log-prob 差值近似计算，防止灾难性遗忘 )*
* **组内 Advantage 归一化 (Z-Score)**：
    **极易出错点**：Z-Score 标准化**必须在同一个 $Q$ 的 $G$ 个采样样本内部**进行，绝对不能在整个全局 Batch 上做 。
    $$\hat{A}_i = \frac{R_{raw}(i) - mean(\{R_{raw}\})}{std(\{R_{raw}\}) + \epsilon}$$ 
* **GRPO 目标函数优化**：
    实现 PPO 风格的 Ratio Clipping 函数：
    $$J_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum \min \left( \rho_i(\theta) \hat{A}_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$ 
    *(这里核心就是提升高 Advantage 样本的生成概率，将 Actor 往 Judge 认为的高分区域推 )*

## 5. 评估与日志记录体系 (Logging & Evaluation)

在代码库中，需要提前埋点记录以下指标，这直接关系到你论文的“证据金字塔”。

* **结果指标 (Result)**：
    * `Pass@1_On_Distribution`：训练同分布集上的准确率 。
    * `OOD_Generalization_Score`：必须设计一个跨域测试集（如 GSM8K 训，MATH 测），评估通用解题策略的泛化性 。
* **过程指标 (Process)**：
    * `ECR (Execution Consistency Rate)`：定期抽样，使用外部工具或代码逻辑判断 $A$ 是否严格遵从 $S$ 推导，监控模型是否发生 Reward Hacking 。
* **机制指标 (Mechanism)**：
    * `JOA (Judge-Outcome Agreement)`：在验证集上监控 Judge 打分。高分策略最终 $y=1$ 的比例，与低分策略最终 $y=0$ 的比例，需在 TensorBoard 中以曲线呈现。

## 6. 实验管控 (用于消融实验的开关)

为了能顺利跑通第 5.2 节的消融实验，你的代码配置系统（如 `yaml` 或 `argparse`）需要具备以下开关：
* `--freeze_judge`: (对应实验 A) 控制 Phase II 中 Judge 是否更新参数 。
* `--dense_reward_alpha`: (对应实验 B) 控制 Reward 公式中 $\alpha$ 的值，设为 0 即为纯稀疏奖励 。
* `--kl_penalty_beta`: (对应实验 D) 设为 0 用于验证是否会生成乱码或 Reward Hacking。
* `--disable_ucb_replay`: (对应实验 E) 将 Buffer 退化为普通 FIFO 队列并采用均匀采样，用于验证反思机制的有效性 。

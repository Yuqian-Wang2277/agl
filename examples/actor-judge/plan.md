这份实现指南将涵盖从显存统筹、分布式通信到定制化数据流的各个关键节点。实现一个带反思机制的协同进化多智能体系统工程量庞大，我们需要自顶向下地梳理所有的技术细节。

以下是为你整理的 Markdown 格式的全面考量清单：

***

# 实验方案代码实现全景指南：基于 `agentlighting` 的自进化 Agent 构建

本指南旨在梳理“执行-评价协同进化”框架（包含 UCB 反思经验池）在代码落地前必须解决的工程与算法问题，确保在 `agentlighting` 框架下的开发工作有的放矢。

## 1. 显存统筹与模型调度 (VRAM & Model Management)

多模型同驻显存是强化学习微调（RLHF/GRPO）中最大的工程瓶颈。我们需要明确框架的并行策略。

* **模型清单与状态**：
    * Actor Model ($\pi_\theta$)：需计算梯度并更新，占用显存最大（参数 + 梯度 + 优化器状态）。
    * Reference Model ($\pi_{SFT}$)：冻结参数，仅用于推理计算 KL 散度。
    * Judge Model ($V_\phi$)：需计算梯度并更新，且在 Rollout 阶段需参与打分。
* **并行策略选型**：明确 `agentlighting` 是否底层支持 DeepSpeed ZeRO-3 或 FSDP。如果是全参数微调，强烈建议开启 ZeRO-3 以切分优化器状态；如果资源受限，需提前决定是否采用 LoRA/QLoRA 进行高效微调 。
* **卸载与调度 (Offloading)**：在 Actor 的 Rollout 阶段，Judge 和 Reference 模型是否可以暂时 Offload 到 CPU 内存？在 Judge 更新阶段，Actor 是否可以挂起？设计模型调度流水线可以极大突破显存上限。

## 2. 核心数据模块：Grouped UCB Buffer 实现细节

反思性经验回放池是本方案的特色，其数据结构的健壮性直接决定了系统能否收敛。

* **存储介质与格式**：绝对禁止在 Buffer 中存储 Tokenize 后的庞大 Tensor 矩阵（如 `input_ids`, `attention_mask`），这会导致严重的 CPU OOM。Buffer 中仅存储纯文本（Raw Text）和极少量的标量（$y$, $n_{sampled}$, $v\_pred$）。
* **哈希分组机制**：内部数据必须以问题 $Q$ 为键（Key）进行字典化存储，确保后续能以 $O(1)$ 或 $O(\log N)$ 的复杂度快速提取同组的 $(S_{win}, S_{lose})$ 对比数据。
* **惰性更新 (Lazy Update)**：计算 UCB 优先级时需要最新的 Judge 打分 $v\_pred$。千万不能在每次采样前用 Judge 对全量 Buffer 做一遍前向推理。应在 Judge 训练当前 Batch 时，将计算出的最新预测值顺便写回 Buffer，未被抽样的历史数据直接使用旧值计算 UCB。
* **单极性数据兜底**：如果某个 $Q$ 下的采样结果全对（只有 $y=1$）或全错（只有 $y=0$），将无法构成 Pairwise Loss 所需的对比对。此时需要抛弃该 $Q$ 并在日志中记录。如果此类情况占比过高，需调整 Actor 的采样温度（Temperature）。

## 3. Phase I: SFT 阶段的定制化 Loss 计算

Phase I 要求实现解耦假设，这就要求我们对传统的 SFT 代码进行魔改。

* **DataCollator 的掩码定制**：标准 SFT 会对 Prompt 计算 Loss 掩码（置为 -100）。我们需要编写一个高度定制的 `DataCollator`。
    * 计算 $L_{gen}$ 时：只对 $<Strategy>$ 标签内的 Token 计算交叉熵。
    * 计算 $L_{exec}$ 时：将 $Q$ 和 $<Strategy>$ 内容视为 Prompt，仅对 $<Answer>$ 标签内的 Token 计算交叉熵。
* **Teacher Forcing 校验**：由于 $L_{exec}$ 的目的是强制模型听从策略，数据集中必须包含“Teacher 给出错误策略，模型推导出错误答案”的鲁棒性样本，以防止模型在训练中走捷径（直接根据 $Q$ 猜答案而无视 $S$）。

## 4. Phase II: Rollout 与环境交互 (Env Interaction)

这是产生动态经验数据的源头。

* **批量生成策略 (Batched Generation)**：Actor 针对 Batch 内的问题 $Q$ 生成 $K$ 个轨迹。需利用 vLLM 等高性能算子加速这一过程，否则 Rollout 的时间开销会远大于网络反向传播的时间。
* **停止词 (Stop Words) 阻断**：生成时需严格设置停止词（如 `</Strategy>`, `</Answer>`），防止模型无休止地输出无关字符消耗算力。
* **沙盒环境 (Sandbox Env)**：答案验证 $y=Env(A, A_{gold})$ 不能仅靠简单的字符串匹配（Exact Match），因为 $A$ 往往包含冗余文本。需实现一个安全的 Python 沙盒或调用外部工具（如基于 SymPy 的数学验证器）来提取和比对最终答案。

## 5. Judge 模型的改造与训练

Judge 模型承担着提供密集奖励的重任。

* **网络结构修改**：Judge 由 Actor (如 Qwen) 初始化，必须用代码截断其最后一层 LM Head（输出维度为 `vocab_size`），替换为一个随机初始化的 Linear 层（输出维度为 1），并添加 Sigmoid 激活函数以输出标量预测概率 $\sigma(V_\phi)$。
* **Pairwise 批处理构建**：从 Buffer 拿到的数据是 $(Q, S_{win})$ 和 $(Q, S_{lose})$。我们需要将它们拼接成两组 `input_ids` 输入 Judge，得到两组 Logit，再代入 Bradley-Terry 公式计算 Loss。需注意 Pad Token 的屏蔽逻辑。

## 6. Actor 模型的 GRPO 更新

利用组内相对优势更新策略生成逻辑。

* **组内 Z-Score 标准化**：在计算 $A^i$（Advantage）时，必须确保是在同一个 $Q$ 产生的那一组 $K$ 个样本内部计算均值和标准差，而不是在整个 Batch 全局计算。
* **KL 惩罚项的快速近似**：严格的 KL 散度计算开销大。在 PPO/GRPO 中，通常使用近似计算 $\log \pi_\theta - \log \pi_{SFT}$。这要求我们维护好 Actor 和 Reference 模型的 Log-Prob 输出同步。
* **PPO Clip 机制防崩溃**：确保代码中正确实现了 Ratio Clip（如限制在 $[0.8, 1.2]$），防止 Actor 在单次更新中步伐过大，破坏 Phase I 积累的执行连贯性。

## 7. 分布式通信与全局同步 (Distributed Sync)

* **经验池的数据聚合**：在多 GPU 训练时，每张卡只生成了一部分 $Q$ 的 Rollout 轨迹。必须在 Env 打分完毕后，使用 `torch.distributed.all_gather_object` 将所有 Rank 的轨迹汇总，统一写入全局的 UCB Buffer 中，保证 Judge 看到的是完整的经验池。


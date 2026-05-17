# Figure-4 热力图归因分析流水线

## 核心目标

通过 LOO（Leave-One-Out）归因分析，定量证明 GIST 训练从根本上改变了策略生成模式：

| 指标 | Pre-GIST（期望值） | Post-GIST（期望值） | 解读 |
|------|-------------------|-------------------|------|
| H_ex（归因熵） | 低 | 高 | Post-GIST 归因更均匀 |
| D_max（最大单例支配度） | 高 | 低 | Pre-GIST 依赖单一示例 |
| S_total（总敏感度） | 低 | 高 | Post-GIST 对全部示例更敏感 |
| Silhouette Score | 低 | 高 | 规则在表征空间中任务族内聚性更强 |

## 环境准备

```bash
pip install -r requirements.txt
```

bge-m3 模型（约 2GB）首次运行时自动下载至 HuggingFace 本地缓存。

## 运行步骤

### Step 1：数据提取（约 1~5 分钟）

从原始评测 shard 中按 problem_type 分层采样 N=100 条任务：
- Pre-GIST：取 reward.correctness 最低的 100 条（表现最差的典型）
- Post-GIST：取 reward.correctness 最高的 100 条（表现最好的典型）

```bash
python 1_extract_tasks.py
# 默认读取 ../../checkpoints_eval_no_verl/ 下的 shard 目录
# 输出：data/pre_gist_tasks.json  data/post_gist_tasks.json
```

如需指定路径：

```bash
python 1_extract_tasks.py \
    --pre-dir /path/to/habit/Qwen3-4B/habit/Qwen3-4B \
    --post-dir /path/to/MIST/Qwen3-semi/MIST/Qwen3-semi
```

### Step 2：M=5 规则生成（需要本地 vLLM 服务）

**前提**：需要分别启动 Qwen3-4B 和 Qwen3-semi 两个 vLLM 服务。

```bash
# 终端1：启动 Pre-GIST 模型（Qwen3-4B）
vllm serve Qwen3-4B --port 8000

# 终端2：启动 Post-GIST 模型（Qwen3-semi）
vllm serve Qwen3-semi --port 8001
```

然后运行规则生成：

```bash
# Pre-GIST 规则
python 2_gen_rules.py --split pre \
    --input data/pre_gist_tasks.json \
    --api-base http://localhost:8000/v1 \
    --model Qwen3-4B \
    --output generated_rules/pre_gist_rules.json

# Post-GIST 规则
python 2_gen_rules.py --split post \
    --input data/post_gist_tasks.json \
    --api-base http://localhost:8001/v1 \
    --model Qwen3-semi \
    --output generated_rules/post_gist_rules.json
```

支持断点续跑，每 10 条自动保存一次。

### Step 3：LOO 归因计算（核心，约 2~6 小时）

对每个任务的每条规则（共 5 条），移除每个示例后重新生成并计算 embedding 差异：

```bash
python 3_loo_attribution.py --split pre \
    --tasks data/pre_gist_tasks.json \
    --rules generated_rules/pre_gist_rules.json \
    --api-base http://localhost:8000/v1 \
    --model Qwen3-4B \
    --output loo_results/pre_gist_loo.json

python 3_loo_attribution.py --split post \
    --tasks data/post_gist_tasks.json \
    --rules generated_rules/post_gist_rules.json \
    --api-base http://localhost:8001/v1 \
    --model Qwen3-semi \
    --output loo_results/post_gist_loo.json
```

每个任务输出：
- `a_bar`：K=3 个示例的归因权重（热力图的一行）
- `H_ex`：归因熵（归一化，越高越均匀）
- `D_max`：最大单例支配度（越低越好）
- `S_total_task`：总敏感度

### Step 4：LLM-as-Judge 四维评分（需要闭源 API）

使用 GPT-4o / Gemini / Claude 三个模型独立打分，自动计算 Cohen's κ：

```bash
export GEMINI_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"

python 4_llm_judge.py --split pre \
    --tasks data/pre_gist_tasks.json \
    --rules generated_rules/pre_gist_rules.json \
    --apis gemini,openai,anthropic \
    --output judge_results/pre_gist_judge.json

python 4_llm_judge.py --split post \
    --tasks data/post_gist_tasks.json \
    --rules generated_rules/post_gist_rules.json \
    --apis gemini,openai,anthropic \
    --output judge_results/post_gist_judge.json
```

如果只有部分 API：

```bash
python 4_llm_judge.py --split pre ... --apis gemini,openai
```

四个评分维度（1-5 分）：
- **abstraction_score**：任务级抽象性（高=好）
- **coverage_score**：多示例覆盖性（高=好）
- **copying_score**：单例复制程度（高=差，自动取反为 copying_score_inv）
- **transferability_score**：可迁移性（高=好）

### Step 5：表征空间分析 + 汇总（约 30~60 分钟）

```bash
python 5_analyze.py \
    --pre-tasks  data/pre_gist_tasks.json \
    --post-tasks data/post_gist_tasks.json \
    --pre-rules  generated_rules/pre_gist_rules.json \
    --post-rules generated_rules/post_gist_rules.json \
    --pre-loo    loo_results/pre_gist_loo_metrics.json \
    --post-loo   loo_results/post_gist_loo_metrics.json \
    --pre-judge  judge_results/pre_gist_judge.json \
    --post-judge judge_results/post_gist_judge.json \
    --judge-api  gemini \
    --judge-key  $GEMINI_API_KEY \
    --output     results/summary.json
```

如需跳过混淆矩阵（节省 API 调用）：

```bash
python 5_analyze.py ... --skip-confusion
```

## 输出文件说明

| 文件 | 内容 |
|------|------|
| `data/pre_gist_tasks.json` | Pre-GIST 100 条任务（含系统提示、示例、策略） |
| `data/post_gist_tasks.json` | Post-GIST 100 条任务 |
| `generated_rules/pre_gist_rules.json` | Pre-GIST 每任务 5 条规则 |
| `generated_rules/post_gist_rules.json` | Post-GIST 每任务 5 条规则 |
| `loo_results/pre_gist_loo.json` | 完整 LOO 结果（含每条规则的 s_ij / a_ij） |
| `loo_results/pre_gist_loo_metrics.json` | LOO 汇总指标（H_ex / D_max / S_total / 归因矩阵） |
| `judge_results/pre_gist_judge.json` | 三模型四维评分结果 |
| `embeddings/pre_gist_emb.npy` | bge-m3 规则向量（N×M × D） |
| `results/summary.json` | 所有阶段汇总指标（直接用于论文绘图） |

## 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 100 | 每 split 采样任务数 |
| M | 5 | 每任务生成规则数（增加鲁棒性） |
| K | 3 | few-shot 支持示例数 |
| Embed | BAAI/bge-m3 | 规则 embedding 模型 |
| LOO workers | 2 | LOO 并发（计算密集，不宜过高） |
| Judge workers | 8 | Judge 评分并发 |
| LOO temperature | 0.7 | 保证重生成的多样性 |

## 数据流

```
Pre-GIST shard (64 JSON) ─┐
                           ├─► 1_extract_tasks.py ─► data/
Post-GIST shard (64 JSON) ─┘

data/ ─► 2_gen_rules.py (Qwen3-4B/Qwen3-semi vLLM) ─► generated_rules/

generated_rules/ ─► 3_loo_attribution.py (bge-m3 + vLLM LOO) ─► loo_results/

generated_rules/ ─► 4_llm_judge.py (GPT-4o/Gemini/Claude) ─► judge_results/

loo_results/ + judge_results/ ─► 5_analyze.py ─► results/summary.json
```

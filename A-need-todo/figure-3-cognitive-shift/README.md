# Figure-3 认知迁移散点图分析流水线

## 核心目标

证明 GIST 训练**真正改变了模型的思考方式**，而非仅提升答题准确率。
通过对模型自发 CoT 轨迹（`<think>` 段）的逐句认知标注，量化五项归纳行为指标，
可视化为散点图，展示"认知迁移"（cognitive shift）。

## 五项认知指标

| 指标 | 含义 | Pre 期望 | Post 期望 |
|------|------|---------|---------|
| ID（Inductive Density） | 归纳性单元（P+R）占比 | 低 | 高 |
| IE（Inductive Earliness） | 首个归纳单元出现越早越接近 1 | 低 | 高 |
| PIR（Pre-execution Induction Ratio） | 归纳发生在执行步骤之前的比例 | 低 | 高 |
| RGR（Rule Grounding Rate） | 执行步骤有归纳依据的比例 | 低 | 高 |
| AHS（Abstraction Hierarchy Score） | 归纳单元的抽象层次均值（0-3） | 低 | 高 |

## 四类思考单元标签

| 标签 | 全称 | 归纳性 | 说明 |
|------|------|--------|------|
| E | Example-recitation | 否 | 直接复述/改写支持示例内容 |
| D | Direct-application | 否 | 直接对当前问题应用知识，无归纳 |
| P | Pattern-abstraction | 是 | 提炼跨示例的通用规则/模式 |
| R | Rule-application | 是 | 将前面的归纳规则用于当前问题 |

## 数据来源

- **Pre-GIST**：`checkpoints_eval_no_verl/few-shot/Qwen3-4B/open-think/` (Qwen3-4B 基础模型，自由 CoT)
- **Post-GIST**：`checkpoints_eval_no_verl/few-shot/Qwen3-4B-trained/open-think-all/` (GIST 微调后，自由 CoT)
- **OOD 过滤**：仅保留 `validation_split` 为 `test-ood-task` 或 `test-bbh` 的样本

## 环境准备

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key_here"
```

## 运行步骤

### Step 1：数据提取（约 2-5 分钟）

从 OOD 评测 shard 中提取 `<think>` 内容，按正确率分层采样：
- Pre：取表现最差的 100 道题（最低 reward.correctness）
- Post：取表现最好的 100 道题（最高 reward.correctness）

```bash
python 1_extract_think.py
# 输出：data/pre_ood_tasks.json  data/post_ood_tasks.json
```

若需指定路径：

```bash
python 1_extract_think.py \
    --pre-dir /path/to/pre/open-think \
    --post-dir /path/to/post/open-think-all \
    --n 100
```

### Step 2：逐单元标注（约 1-3 小时，取决于 API 并发）

用 Gemini 对每条轨迹的命题单元打标签（E/D/P/R），并对 P 单元评 AHS 分：

```bash
# Pre 标注
python 2_annotate_units.py --split pre --api-key $GEMINI_API_KEY --workers 8

# Post 标注
python 2_annotate_units.py --split post --api-key $GEMINI_API_KEY --workers 8
```

标注流程：
1. 按句号/分号切分 `<think>` 为命题单元
2. Prompt 1 v1 标注（round 1）
3. Prompt 1 v2 标注（round 2，不同措辞）
4. 计算 Cohen's kappa；若 kappa < 0.7，对分歧单元第三次裁决
5. 对最终标签为 P 的单元用 Prompt 2 评 AHS 分（0-3）

断点续跑：重新运行相同命令即可，已完成的任务自动跳过。

### Step 3：计算五项指标（约 1 分钟）

```bash
python 3_compute_metrics.py
# 输出：metrics/all_metrics.json  metrics/summary_stats.json
```

输出包含每条轨迹的 5 项指标，以及 Pre/Post 的均值对比和 Mann-Whitney U 显著性检验。

### Step 4：生成散点图

```bash
python 4_visualize.py
# 输出：results/cognitive_shift.pdf  results/cognitive_shift.png
```

图表元素说明：
- **X 轴** = PIR（执行前归纳比例）
- **Y 轴** = ID（归纳单元密度）
- **形状** = IE（★ IE≥0.7 / ● 0.3≤IE<0.7 / ▲ IE<0.3）
- **点大小** = RGR（越大越有规律依据）
- **颜色深浅** = AHS（越深抽象层次越高）
- **红色** = M_pre，**紫色** = M_post

如需包含所有数据点（不过滤噪声）：

```bash
python 4_visualize.py --no-filter
```

## 输出文件

| 文件 | 内容 |
|------|------|
| `data/pre_ood_tasks.json` | Pre 100 道 OOD 题的 think 内容 |
| `data/post_ood_tasks.json` | Post 100 道 OOD 题的 think 内容 |
| `annotations/pre_annotations.json` | Pre 每单元 E/D/P/R 标签 + AHS 分 |
| `annotations/post_annotations.json` | Post 每单元 E/D/P/R 标签 + AHS 分 |
| `metrics/all_metrics.json` | 每条轨迹的 5 项指标 |
| `metrics/summary_stats.json` | Pre/Post 对比表（均值/标准差/p 值） |
| `results/cognitive_shift.pdf` | 主图（矢量格式） |
| `results/cognitive_shift.png` | 主图（位图，200 DPI） |

## Prompt 文件

| 文件 | 用途 |
|------|------|
| `prompts/annotate_unit_v1.txt` | Prompt 1 round 1：E/D/P/R 四类标注 |
| `prompts/annotate_unit_v2.txt` | Prompt 1 round 2：同语义不同措辞，用于 Cohen's kappa |
| `prompts/abstract_score.txt` | Prompt 2：仅对 P 单元的 AHS 0-3 评分 |

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| N | 100 | 每 split 采样题数 |
| OOD splits | test-ood-task, test-bbh | 只分析 OOD 任务 |
| Gemini model | gemini-2.0-flash | 标注模型 |
| workers | 8 | API 并发线程数 |
| kappa 阈值 | 0.7 | 低于此值对分歧单元做第三次裁决 |
| 点过滤 | n_units>=5, ID>0 or AHS>0 | 去除无归纳行为的噪声轨迹 |

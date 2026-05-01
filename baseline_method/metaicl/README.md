# MetaICL Baseline

本目录实现了 **MetaICL**（Meta In-Context Learning）对比基线，用于与 MIST 方法进行公平比较。

MetaICL 是一种 ICL 风格的 SFT 方法：将训练数据组织成 k-shot 情节（episode），模型从上下文中隐式学习任务模式，损失仅在最终测试样本的答案上计算。

---

## 目录结构

```
metaicl/
├── data_formatter.py        # BIG-bench 数据 → MetaICL k-shot 格式转换
├── train_metaicl.py         # MetaICL SFT 训练（TRL SFTTrainer）
├── train_metaicl_cot.py     # MetaICL-CoT 训练（带 Think: 字段）
├── eval.py                  # 统一评测脚本（5 个 benchmark 一键评测）
├── configs/
│   ├── train_qwen3_4b.yaml      # MetaICL 训练配置
│   └── train_qwen3_4b_cot.yaml  # MetaICL-CoT 训练配置
└── scripts/
    ├── run_train.sh         # MetaICL 训练启动脚本
    ├── run_train_cot.sh     # MetaICL-CoT 训练启动脚本
    └── run_eval.sh          # 评测启动脚本（自动管理 vLLM 进程）
```

---

## 方法说明

### MetaICL

训练格式（每条样本包含 k 个 shot + 1 个测试查询）：

```
Input: <shot_1_input>
Output: <shot_1_answer>

Input: <shot_2_input>
Output: <shot_2_answer>

...

Input: <test_input>
Output: <test_answer>   ← 仅此处计算 loss
```

使用 TRL 的 `DataCollatorForCompletionOnlyLM`，`response_template="\nOutput:"`，确保 loss 只在最后一个 `Output:` 之后的答案上计算（前导 `\n` 保证匹配的是最后一个 Output 而非 few-shot 中的 Output）。

### MetaICL-CoT

在测试查询的 `Output:` 之前插入一个 `Think:` 字段：

```
Input: <test_input>
Think: <reasoning>
Output: <test_answer>   ← 仅此处计算 loss
```

`Think:` 内容来源（`think_source`）：

| 来源 | 说明 |
|------|------|
| `empty`（默认）| Think 字段为空，模型学习格式但不监督推理内容；推理时模型自由生成 |
| `file` | 从外部 JSONL 文件加载预生成的推理链，每行格式：`{"input": "...", "think": "..."}` |

### 与 MIST 的计算公平性

| 方法 | 推理计算量 | 主要对比 |
|------|-----------|---------|
| MetaICL | 1× | ← 与 MIST-single (1×) 主对比 |
| MetaICL-CoT | ~1.5× | ← 与 MIST-A (2×) 次对比 |
| MIST-single | 1× | 单步同时输出策略+答案 |
| MIST-A | 2× | 两步：策略生成 → 答案生成 |

所有方法统一使用 **pass@1 + greedy decode（temperature=0）**，保证与闭源模型比较的一致性。

---

## 环境要求

训练和评测统一使用 `agl` 环境。`agl` 默认没有 `trl`，首次使用前需安装一次：

```bash
conda activate agl
pip install trl
```

安装后 `agl` 具备：torch 2.8.0、transformers 4.57.5、trl、vllm、openai，训练和评测全部覆盖。

---

## 数据准备

训练数据读取自 `../../data/train_all_project_suitable/`（相对于本目录）。目录结构：

```
train_all_project_suitable/
├── <task_type_1>/
│   ├── *.json    # BIG-bench 格式：{"task": ..., "examples": [{"input": ..., "target": ...}]}
│   └── ...
├── <task_type_2>/
│   └── ...
...
```

评测数据路径（已在代码中设置默认值，可通过参数覆盖）：

| Benchmark | 默认路径 |
|-----------|---------|
| test-id-subtask / test-ood-task / test-bbh | `../../data/test-*/` |
| HARDMath2 | `../../banchmark/HARDMath2/data/` |
| Linguini | `../../banchmark/linguini/dataset.jsonl` |

---

## 快速开始

### 1. 训练 MetaICL

```bash
# 使用默认配置（4 GPU）
bash scripts/run_train.sh

# 自定义参数
GPUS=0,1 MAX_STEPS=500 bash scripts/run_train.sh

# 指定 checkpoint 输出目录
OUTPUT_DIR=./checkpoints/my-run bash scripts/run_train.sh
```

### 2. 训练 MetaICL-CoT

```bash
# 默认：Think 字段为空（模型学习格式，推理时自由生成）
bash scripts/run_train_cot.sh

# 使用预生成推理链文件
THINK_SOURCE=file THINK_FILE=/path/to/think_labels.jsonl bash scripts/run_train_cot.sh
```

### 3. 评测

```bash
# 评测所有 benchmark（自动启动/停止 vLLM）
CHECKPOINT=./checkpoints/metaicl-qwen3-4b/best bash scripts/run_eval.sh

# 只评测部分 benchmark
BENCHMARK="id-ood bbh" CHECKPOINT=./checkpoints/metaicl-qwen3-4b/best bash scripts/run_eval.sh

# 评测 CoT 变体
CHECKPOINT=./checkpoints/metaicl-cot-qwen3-4b/best \
MODEL_NAME=metaicl-cot \
bash scripts/run_eval.sh

# 冒烟测试（每个 benchmark 只取 10 条）
MAX_SAMPLES=10 bash scripts/run_eval.sh
```

结果保存在 `./results/` 目录，文件名格式：`<benchmark>_<timestamp>.json`。

---

## 详细使用

### 训练脚本参数

#### `scripts/run_train.sh`

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `CONFIG` | `configs/train_qwen3_4b.yaml` | 训练配置文件路径 |
| `GPUS` | `0,1,2,3` | 使用的 GPU（逗号分隔） |
| `MAX_STEPS` | 配置文件值 (10000) | 覆盖最大训练步数 |
| `K_SHOT` | 配置文件值 (4) | 覆盖 few-shot 数量 |
| `LR` | 配置文件值 (1e-5) | 覆盖学习率 |
| `MODEL_PATH` | 配置文件值 | 覆盖模型路径 |
| `DATA_DIR` | 配置文件值 | 覆盖数据目录 |
| `OUTPUT_DIR` | 配置文件值 | 覆盖输出目录 |

#### `scripts/run_train_cot.sh`

在 `run_train.sh` 的所有参数基础上，额外支持：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `THINK_SOURCE` | `empty` | Think 标签来源：`empty` 或 `file` |
| `THINK_FILE` | 无 | 预生成推理链 JSONL 文件（`THINK_SOURCE=file` 时必填） |

#### `scripts/run_eval.sh`

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `CHECKPOINT` | `./checkpoints/metaicl-qwen3-4b/best` | 模型 checkpoint 路径 |
| `MODEL_NAME` | `metaicl` | vLLM 服务名称 |
| `BENCHMARK` | `all` | 评测集（空格分隔，可选值见下） |
| `VLLM_GPUS` | `0,1` | vLLM 服务使用的 GPU |
| `VLLM_PORT` | `8300` | vLLM API 端口 |
| `TENSOR_PARALLEL` | GPU 数量 | vLLM tensor parallel size |
| `K_SHOT` | `4` | few-shot 数量（必须与训练一致） |
| `OUTPUT_DIR` | `./results` | 结果输出目录 |
| `CONCURRENCY` | `32` | 并发 API 请求数 |
| `MAX_SAMPLES` | 无限制 | 每个 benchmark 最多评测条数（调试用） |

`BENCHMARK` 可选值：

| 值 | 说明 |
|---|------|
| `all` | 运行所有 benchmark |
| `id-ood` | test-id-subtask + test-ood-task |
| `bbh` | test-bbh |
| `hardmath` | HARDMath2 |
| `linguini` | Linguini |

### 直接调用 `eval.py`

若 vLLM 已在外部启动，可直接调用：

```bash
conda activate agl

# 评测所有 benchmark
python eval.py \
    --benchmark all \
    --model-url http://localhost:8300/v1 \
    --model-name metaicl \
    --k-shot 4 \
    --output-dir ./results

# 评测部分 benchmark
python eval.py -b id-ood bbh --model-name metaicl

# 查看所有参数
python eval.py --help
```

`eval.py` 完整参数：

```
--benchmark / -b     benchmark 名称（可多选），默认 all
--model-url          vLLM API 地址，默认 http://localhost:8300/v1
                     （也可通过 MODEL_URL 环境变量设置）
--model-name         vLLM 服务的模型名称，默认 metaicl
                     （也可通过 MODEL_NAME 环境变量设置）
--api-key            API Key（vLLM 本地部署填 dummy），默认 dummy
--k-shot             few-shot 数量（需与训练一致），默认 4
--samples-per-subtask  BBH/ID-OOD 每个子任务采样的题目数，默认 20
--seed               随机种子，默认 42
--concurrency        并发请求数，默认 32
--max-samples        每个 benchmark 最多处理条数（None = 全量），默认 None
--output-dir         JSON 结果输出目录，默认 ./results
--data-base          test-* 数据根目录
--hardmath-data      HARDMath2 数据目录
--linguini-data      Linguini dataset.jsonl 路径
```

### 训练配置文件

#### `configs/train_qwen3_4b.yaml`

```yaml
model_name_or_path: /home/test/test16/chenlu/model/Qwen3-4B
data_dir: ../../data/train_all_project_suitable
k_shot: 4
samples_per_task: 2000          # 每个任务类型生成 2000 条训练样本（44 任务 × 2000 = 88000）
val_ratio: 0.05
max_steps: 10000
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # effective batch = 4 × 4 × n_gpus
learning_rate: 1.0e-5
max_seq_length: 2048
eval_steps: 500
save_steps: 500
save_total_limit: 3
```

#### `configs/train_qwen3_4b_cot.yaml`

与上面相同，但：
- `output_dir: ./checkpoints/metaicl-cot-qwen3-4b`
- `think_source: empty`（可改为 `file`）
- `max_seq_length: 2560`（Think 字段需要更长的序列）

---

## 模块 API

### `data_formatter.py`

```python
from data_formatter import (
    build_metaicl_dataset,
    make_metaicl_text,
    make_metaicl_cot_text,
    make_metaicl_prompt,
    load_task_examples,
)

# 构建训练/验证集
train_samples, val_samples = build_metaicl_dataset(
    data_dir=Path("../../data/train_all_project_suitable"),
    k=4,
    samples_per_task=2000,
    val_ratio=0.05,
    seed=42,
    cot_reasoning_fn=None,    # 不传则为标准 MetaICL；传入则为 CoT
)
# train_samples: List[{"text": str}]

# 构造推理 prompt（不含答案，用于 eval）
prompt = make_metaicl_prompt(
    shot_examples=[{"input": "...", "target": "..."}, ...],
    test_input="...",
    k=4,
    cot=False,   # True 时以 "Think:" 结尾
)
```

### 冒烟测试（验证数据格式）

```bash
conda activate agl
cd baseline_method/metaicl
python data_formatter.py
# 输出：Train: 88,000  Val: 4,400  及示例文本
```

---

## 结果格式

评测结果以 JSON 格式保存，每个 benchmark 一个文件：

```
results/
├── bbh_id-subtask_ood-task_20260430_183000.json
├── bbh_bbh_20260430_183500.json
├── hardmath_20260430_184000.json
└── linguini_20260430_184500.json
```

BBH / ID-OOD 结果结构：
```json
{
  "summary": {
    "test-id-subtask": {"pass@1": 0.6532, "n": 420},
    "test-ood-task":   {"pass@1": 0.5214, "n": 380},
    "overall":         {"pass@1": 0.5901, "n": 800}
  },
  "details": [
    {
      "split": "test-id-subtask",
      "subtask": "logical_deduction",
      "hard_correct": 1,
      "soft_score": 1.0,
      "prediction": "A",
      "ground_truth": "A"
    }, ...
  ]
}
```

HARDMath2 / Linguini 结果结构：
```json
{
  "summary": {"pass@1": 0.4230, "n": 200},
  "details": [
    {"type": "algebra", "hard_correct": 1, "soft_score": 1.0, "prediction": "...", "ground_truth": "..."}, ...
  ]
}
```

---

## 手动启动 vLLM

若需要独立管理 vLLM 服务（例如对多个 checkpoint 重复评测），可手动启动：

```bash
conda activate agl

# MetaICL
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/metaicl-qwen3-4b/best \
    --served-model-name metaicl \
    --tensor-parallel-size 2 \
    --port 8300

# MetaICL-CoT（另一个端口）
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/metaicl-cot-qwen3-4b/best \
    --served-model-name metaicl-cot \
    --tensor-parallel-size 2 \
    --port 8301
```

然后直接调用 `eval.py`，不需要通过 `run_eval.sh`：

```bash
conda activate agl
python eval.py --benchmark all --model-url http://localhost:8300/v1 --model-name metaicl
python eval.py --benchmark all --model-url http://localhost:8301/v1 --model-name metaicl-cot --output-dir ./results_cot
```

---

## 常见问题

**Q: 训练时 loss 只在最后一个答案上计算还是所有 Output: 上？**

只在最后一个 `Output:`（测试查询的答案）上计算。`DataCollatorForCompletionOnlyLM` 会找到序列中 `response_template="\nOutput:"` 的**最后一次**出现位置，并屏蔽其之前的所有 token 的 loss。前导 `\n` 是必要的，确保在 few-shot 的中间 `Output:` 处也能正确区分。

**Q: 评测时 MetaICL 与 MIST 的信息量是否公平？**

公平。MetaICL 在评测时看到 k 个 few-shot 示例后直接输出答案；MIST 在评测时看到相同的 k 个 few-shot 示例，先生成策略再回答（答案模型只见策略，不再重复见 few-shot）。两者起点相同（都从 k 个示例出发），属于计算公平对比。

**Q: 为什么 `run_eval.sh` 等待 vLLM 就绪的最长时间是 120s？**

大模型加载时间通常在 60–90s 之间，120s 已足够。若加载更大的模型（如 72B）超时，可设置 `MAX_WAIT=300` 修改脚本中对应变量。

**Q: 如何只评测单个 benchmark 以节省时间？**

```bash
BENCHMARK=hardmath bash scripts/run_eval.sh
BENCHMARK="id-ood bbh" bash scripts/run_eval.sh
```

**Q: `--samples-per-subtask` 和 `--max-samples` 有什么区别？**

`--samples-per-subtask`（默认 20）：BBH 系列每个子任务文件中随机采样的题目数，控制各子任务均衡采样。`--max-samples`：在所有采样完成后再做一次全局截断，主要用于快速调试。

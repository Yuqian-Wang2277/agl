# Strategy Scorer Bootstrap (Qwen3-8B)

这个目录用于两件事：

1. 先做零样本打分（你当前阶段）：直接调用 Qwen3-8B 按 rubric 打分。  
2. 后续补标注数据后，升级为监督训练打分模型。

## 目录

- `score_with_qwen.py`: 批量调用模型打分（输入 JSONL，输出 JSONL）。
- `rubric.py`: rubric 维度、权重和分数归一化逻辑。
- `scripts/score_qwen3_8b.sh`: 一键示例脚本。
- `data/example_input.jsonl`: 示例输入数据。

## 输入格式（JSONL）

每行一个样本，字段：

- `id`（可选）
- `task`（可选）
- `examples`（可选，few-shot 列表）
- `strategy`（必填，待打分策略）

示例见 `strategy_scorer_training/data/example_input.jsonl`。

## 先启动模型服务

示例（vLLM）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/test/test16/chenlu/model/Qwen3-8B \
  --served-model-name qwen3-8b-scorer \
  --port 8100
```

## 运行打分

```bash
bash examples/strategy_extraction/strategy_scorer_training/scripts/score_qwen3_8b.sh
```

或直接调用：

```bash
python -m examples.strategy_extraction.strategy_scorer_training.score_with_qwen \
  --input-jsonl examples/strategy_extraction/strategy_scorer_training/data/example_input.jsonl \
  --output-jsonl examples/strategy_extraction/strategy_scorer_training/data/example_output.jsonl \
  --base-url http://localhost:8100/v1 \
  --model qwen3-8b-scorer \
  --prompt-version v2 \
  --overwrite
```

## 输出说明

输出 JSONL 每行包含：

- `result.status`: `ok` / `ok_numeric_fallback` / `parse_error` / `api_error`
- `result.parsed.weighted_total_0_100`: 0-100 总分
- `result.parsed.score`: 0-1 分数（可直接用于 `reward/v2.py`）
- `result.parsed.dimension_scores`: 8 个维度分
- `result.parsed.deductions`: 扣分项

## 下一步（你补标注后）

你补充人工标注数据后，可以把这批零样本打分结果当作弱标签起点，再做监督训练和校准。


# StrategyQA

跨数据集压力测试之一（mist_main.tex §3）。

## 来源
- HuggingFace: `ChilleD/StrategyQA`
- 原始论文: Geva et al. 2021, "Did Aristotle Use a Laptop?" (`geva2021strategyqa`)

## 数据
| 文件 | Split | 样本数 |
|------|-------|--------|
| `data/test.json` | test | 687 |

## 字段
```json
{
  "qid": "...",
  "term": "...",
  "description": "...",
  "question": "...",
  "answer": true/false,
  "facts": ["..."]
}
```

## 下载时间
2026-05-14，via hf-mirror.com

数据集	文件	样本数	字段
MATH-500	banchmark/MATH-500/data/test.json	500	problem, solution, answer, subject, level
StrategyQA	banchmark/StrategyQA/data/test.json	687	question, answer(bool), facts, term
ReClor	banchmark/ReClor/data/test.json	500	context, question, answers(4选1), label
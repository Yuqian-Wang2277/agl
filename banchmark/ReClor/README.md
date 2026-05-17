# ReClor

跨数据集压力测试之一（mist_main.tex §3）。

## 来源
- HuggingFace: `metaeval/reclor`
- 原始论文: Yu et al. 2020, "ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning" (`yu2020reclor`)

## 数据
| 文件 | Split | 样本数 | 说明 |
|------|-------|--------|------|
| `data/test.json` | validation | 500 | 官方 test 集标签不公开，validation 为标准评测集 |

## 字段
```json
{
  "context": "...",
  "question": "...",
  "answers": ["A", "B", "C", "D"],
  "label": 0-3,
  "id_string": "..."
}
```

## 下载时间
2026-05-14，via hf-mirror.com

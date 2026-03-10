# 无策略基线脚本的准确率计算说明（`reward/v3.py`）

本文说明使用脚本 `examples/strategy_extraction/scripts/eval_no_strategy_baseline.sh` 时，准确率/正确性是如何得到的。

---

## 1. 这次基线的核心配置是什么

脚本关键参数：

- `--reward-version v3`：答案判分走 `reward/v3.py`
- `--no-strategy-for-answer`：答案阶段不把生成的策略传给答案模型（无策略基线）
- `--answer-prompt-version no_strategy_baseline`：答案提示词不包含策略
- `--format-weight 0.0 --scorer-weight 0.0 --grounded-proxy-weight 0.0 --correctness-weight 1.0`
  - 最终 reward 完全由 correctness 决定

对应最终 reward 公式（在 `reward/v3.py`）：

`final_reward = format_weight * format_reward + correctness_weight * correctness`

在本脚本下等价于：

`final_reward = correctness`

---

## 2. 验证集 980 条是如何采样出来的

你现在用的是：

- `--val-sampling-mode per_subtask_fixed`
- `--val-samples-per-subtask 20`
- `--val-sampling-seed 42`
- `--val-subdirs test-id-subtask test-ood-task test-bbh`

采样函数是 `train_strategy_generation.py` 里的 `_create_strategy_generation_dataset_per_subtask(...)`，规则：

1. 遍历每个验证集目录下所有 `problem_type/*.json`（按文件名排序，确定性顺序）
2. 对每个子任务文件固定采样 20 条
3. 随机数由同一个 seed 控制（`random.Random(seed)`），保证可复现

子任务数量是：

- `test-id-subtask`: 11
- `test-ood-task`: 15
- `test-bbh`: 23

所以总数是：

`(11 + 15 + 23) * 20 = 980`

---

## 3. “无策略”在代码里具体怎么生效

`strategy_generation_agent.py` 中新增了 `use_strategy_for_answer` 开关：

- `--no-strategy-for-answer` 会让 `use_strategy_for_answer=False`
- 调答案模型时传入：
  - `strategy=""`（空字符串）
  - 只保留 problem

所以这个实验不是“完全不生成 strategy”（前半段仍会生成并记录），而是“答案阶段不使用 strategy 信息”，用于做 no-strategy baseline。

---

## 4. 单条样本的 correctness 是怎么计算的（`v3`）

每条验证样本流程：

1. 生成 `answer_raw`
2. 用 `reward/v3.py::extract_answer(...)` 抽取最终答案片段（优先 `<answer>...</answer>`，再 fallback 规则）
3. 调用 `compute_answer_correctness(answer, ground_truth, numeric_tolerance, f1_threshold)`
4. 该函数内部调用 `compute_answer_judgement(...)`，返回 detail，其中：
   - `soft_score`：连续值 `[0,1]`
   - `hard_correct`：硬判定 `0/1`（注意：当前训练/验证主流程默认没把它单独写成主指标）
5. 当前主流程把 `correctness = soft_score`

也就是说：**这个脚本下“准确率口径”本质是 soft correctness（连续正确率），不是纯 0/1 exact-match 准确率。**

---

## 5. `v3` 的“新准确率统计”具体判分逻辑

`compute_answer_judgement(...)` 会先路由答案类型，再按类型打分：

- `yes_no`：是/否标签归一化后严格匹配
- `option_letter`：选项字母（支持 `(a)`, `a.` 等形式）
- `numeric`：数值解析 + 容差规则（绝对/相对误差，百分比等子类型）
- `multi_numeric_set` / `multi_text_set`：集合或多重集合比较（含 F1）
- `short_phrase`：短文本精确匹配优先，近似匹配给受限 soft 分
- `medium/long`：字符/词级相似度 + 否定冲突/数字冲突惩罚

输出 `soft_score` 会被裁剪到 `[0,1]`。

---

## 6. 最终“准确率”如何从验证结果汇总

在 `strategy_generation_agent.py` 中，每条验证样本会保存：

- `reward.correctness`（本实验就是 `soft_score`）
- `reward.final`（本实验等于 `correctness`）

汇总时，如果你说“准确率”，当前脚本最一致的定义是：

`Accuracy_soft = mean(reward.correctness)`  

因为本脚本下 `final_reward = correctness`，也等价于：

`Accuracy_soft = mean(reward.final)`

---

## 7. 如果你想同时看“硬准确率（0/1）”

当前主流程默认只用 `compute_answer_correctness(...)` 的 soft 分。  
若要硬准确率，需要在 rollout 中额外调用 `compute_answer_judgement(...)` 并记录 `hard_correct`，再统计：

`Accuracy_hard = mean(hard_correct)`

这会更接近传统 exact-match 准确率。

---

## 8. 一句话总结

使用 `eval_no_strategy_baseline.sh` 得到的“准确率”，在当前实现里是：  
**基于 `reward/v3.py` 的类型路由 soft correctness 的平均值（并且在本脚本配置下与 final reward 均值相同）。**

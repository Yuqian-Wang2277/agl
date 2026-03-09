# strategy_extraction 当前最新 reward 结构分析

## 结论先行

在当前代码与训练脚本配置下，实际使用的“最新 reward”是：

- `reward_version = v2`
- `reward_mode = hybrid_grounded`
- 默认/示例权重：`format_weight=0.1`, `scorer_weight=0.4`, `proxy_weight=0.5`
- 参考：`scripts/train_generation.sh`、`strategy_generation_agent.py`

这意味着最终奖励不是单一正确率，而是 **格式 + OC策略打分 + 多采样落地正确率代理** 的组合。

---

## 1) 目前最新 reward 结构是什么

### 1.1 总体结构（hybrid_grounded）

在 `strategy_generation_agent.py` 的 `hybrid_grounded` 分支中，最终奖励调用：

`compute_hybrid_reward(format_r, oc_scorer_r, grounded_proxy, fw, sw, pw)`

对应公式（权重必须和为 1）：

`final_reward = fw * format_r + sw * oc_scorer_r + pw * grounded_proxy`

其中：

1. `format_r`（格式分）
   - 普通策略提示：`compute_format_reward`（有且仅有一个 `<strategy>...</strategy>` 且非空时为 1，否则 0）
   - 结构化策略提示：`compute_format_reward_structured`（三档：1.0 / 0.3 / 0.0）

2. `oc_scorer_r`（策略质量分，0~1）
   - 来自单独 scorer LLM 的 JSON rubric 评分，经 `parse_oc_scorer_response` 解析并归一化。

3. `grounded_proxy`（落地正确率代理，0~1）
   - 对同一 strategy 采样 `K` 次 answer（默认 K=4）
   - 统计 `score==1.0` 的比例：`exact_success / K`
   - 当组内全 0 时，可回退到 `soft_correctness_mean * 0.3`（soft fallback）

---

## 2) 目前答案提取方式是什么

答案提取由 `v1.extract_answer` 提供，`v2` 复用该逻辑：

- 正则提取首个 `<answer>...</answer>` 块内容
- 提取后 `strip()`
- 若无匹配或空内容，返回 `None`

即：**严格依赖 `<answer>` 标签**，不做自然语言兜底抽取。

---

## 3) 判断答案准确率的方式是什么

准确率函数是 `v1.compute_answer_correctness`（`v2` 同样复用），按以下顺序判定：

1. **Exact Match**（完全匹配，忽略大小写与首尾空格）
   - 命中得分：`1.0`

2. **Numeric Match**（数值近似）
   - 尝试把预测与真值转成 `float`
   - 相对误差 `< numeric_tolerance`（默认 `0.02`）
   - 命中得分：`0.8`

3. **F1 Match**（词集合 F1）
   - 分词后集合求 Precision/Recall/F1
   - `F1 >= f1_threshold`（默认 `0.5`）
   - 命中得分：`0.5`

4. 以上都不满足
   - 得分：`0.0`

在 `hybrid_grounded` 中，这个 correctness 主要用于：

- 形成 `router_scores`（每个 sample 的 correctness）
- 计算 `grounded_proxy`（仅统计 `score==1.0` 的比例）
- 计算 `soft_correctness_mean`（平均 correctness，用于全 0 回退）

---

## 4) 分数标准（评分细则）是什么

### 4.1 Answer correctness 分档标准（用于路由/代理）

- `1.0`：文本完全正确（Exact）
- `0.8`：数值近似正确（相对误差 < 2%）
- `0.5`：语义部分匹配（F1 >= 0.5）
- `0.0`：错误

### 4.2 OC scorer 评分标准（策略质量主信号）

来自 `hybrid_grounded_reward.py` + `prompt/strategy_scoring/v2.toml`：

1. 六维打分（每维 0~5）：
   - `A_outcome_support`
   - `B_executability`
   - `C_example_grounding`
   - `D_problem_coverage`
   - `E_transfer_robustness`
   - `F_clarity_economy`

2. 加权到 100 分制：

`weighted_raw_100 = 20 * (0.30*A + 0.20*B + 0.15*C + 0.15*D + 0.15*E + 0.05*F)`

3. Cap（上限封顶）规则（取最严格上限）：
   - `correct_but_unrelated` -> 上限 55
   - `wrong_but_strategy_ok` -> 上限 75
   - `core_wrong_step` -> 上限 35
   - `generic_template` -> 上限 50
   - `leaked_answer` -> 上限 60

4. Penalty（扣分）规则（可叠加）：
   - `generic_not_operational` -> -15
   - `memorization_leakage` -> -15
   - `contradiction` -> -20
   - `missing_check` -> -10

5. 最终分计算：

`final_100 = clip(weighted_raw_100 - penalty_total, 0, cap_ceiling)`

`final_score_01 = round(final_100 / 100, 4)`

### 4.3 最终训练 reward 标准（当前脚本）

`final_reward = 0.1 * format + 0.4 * oc_scorer + 0.5 * grounded_proxy`

并要求权重和为 1，否则报错。

---

## 5) 与旧版 v1 的区别（便于对比）

- `v1`：`final = fw * format + cw * correctness`（主要靠答案正确性）
- 当前链路（`v2 + hybrid_grounded`）：引入 scorer 直接评策略质量，并用多采样 grounded proxy 提供结果约束，信号更稳定、可解释性更强。


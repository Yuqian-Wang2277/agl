# MIST 各题型准确率

---

## Qwen3-4B-trained / think-open-all

- **数据来源**：`/home/test/test16/chenlu/projects/agent-lightning/checkpoints_eval_no_verl/MIST/Qwen3-4B-trained/think-open-all`（64 个验证 JSON，3420 条 rollout）
- **指标**：`reward.hard_correct`（与 `correctness` 一致）；按 `problem_type` 汇总，**跨 `validation_split` 合并**（同一题型在 ID / OOD / BBH 中均计入）
- **题型数**：47

| problem_type | 准确率 | 正确数/总数 |
| --- | --- | --- |
| reasoning_about_colored_objects | 100.00% | 60/60 |
| temporal_sequences | 100.00% | 60/60 |
| tracking_shuffled_objects | 99.44% | 179/180 |
| boolean_expressions | 98.33% | 59/60 |
| elementary_math_qa | 97.67% | 293/300 |
| logical_deduction | 97.22% | 175/180 |
| formal_fallacies | 96.67% | 58/60 |
| navigate | 96.67% | 58/60 |
| web_of_lies | 96.67% | 58/60 |
| linguistic_mappings | 93.33% | 56/60 |
| arithmetic | 90.00% | 54/60 |
| contextual_parametric_knowledge_conflicts | 90.00% | 54/60 |
| implicatures | 88.33% | 53/60 |
| multistep_arithmetic_two | 88.33% | 53/60 |
| penguins_in_a_table | 88.33% | 53/60 |
| word_sorting | 88.33% | 53/60 |
| hyperbaton | 87.50% | 105/120 |
| object_counting | 81.67% | 49/60 |
| date_understanding | 80.00% | 48/60 |
| modified_arithmetic | 80.00% | 48/60 |
| vitaminc_fact_verification | 78.33% | 47/60 |
| goal_step_wikihow | 76.67% | 46/60 |
| fact_checker | 75.00% | 45/60 |
| nonsense_words_grammar | 73.33% | 44/60 |
| snarks | 73.33% | 88/120 |
| sports_understanding | 68.33% | 41/60 |
| salient_translation_error_detection | 65.00% | 39/60 |
| matrixshapes | 63.33% | 38/60 |
| movie_recommendation | 63.33% | 38/60 |
| disambiguation_qa | 61.67% | 37/60 |
| movie_dialog_same_or_different | 58.33% | 35/60 |
| geometric_shapes | 56.67% | 34/60 |
| causal_judgement | 48.33% | 29/60 |
| unnatural_in_context_learning | 48.33% | 29/60 |
| ruin_names | 45.00% | 27/60 |
| disfl_qa | 35.00% | 21/60 |
| discourse_marker_prediction | 26.67% | 16/60 |
| word_unscrambling | 26.67% | 16/60 |
| language_identification | 23.33% | 14/60 |
| ascii_word_recognition | 21.67% | 13/60 |
| simp_turing_concept | 21.67% | 13/60 |
| dyck_languages | 18.33% | 11/60 |
| intersect_geometry | 13.33% | 8/60 |
| real_or_fake_text | 11.67% | 7/60 |
| cryptonite | 5.00% | 3/60 |
| mnist_ascii | 3.33% | 2/60 |
| chess_state_tracking | 0.00% | 0/60 |

---

## Qwen3-8B-trained

- **数据来源**：`/home/test/test16/chenlu/projects/agent-lightning/checkpoints_eval_no_verl/MIST/Qwen3-8B-trained`（递归扫描全部验证 JSON，当前目录下为 **open-think** 子树；共 64 个 JSON，**3417** 条 rollout，与 4B 侧 3420 条相差 3 条）
- **指标**：`reward.hard_correct`（与 `correctness` 一致）；按 `problem_type` 汇总，**跨 `validation_split` 合并**
- **题型数**：47

| problem_type | 准确率 | 正确数/总数 |
| --- | --- | --- |
| modified_arithmetic | 100.00% | 60/60 |
| arithmetic | 96.67% | 58/60 |
| boolean_expressions | 96.67% | 58/60 |
| linguistic_mappings | 93.33% | 56/60 |
| temporal_sequences | 93.33% | 56/60 |
| web_of_lies | 91.67% | 55/60 |
| implicatures | 90.00% | 54/60 |
| multistep_arithmetic_two | 88.33% | 53/60 |
| reasoning_about_colored_objects | 86.67% | 52/60 |
| contextual_parametric_knowledge_conflicts | 85.00% | 51/60 |
| navigate | 85.00% | 51/60 |
| unnatural_in_context_learning | 85.00% | 51/60 |
| penguins_in_a_table | 83.33% | 50/60 |
| fact_checker | 81.67% | 49/60 |
| formal_fallacies | 81.67% | 49/60 |
| elementary_math_qa | 80.33% | 241/300 |
| tracking_shuffled_objects | 80.00% | 144/180 |
| vitaminc_fact_verification | 75.00% | 45/60 |
| date_understanding | 74.58% | 44/59 |
| sports_understanding | 71.67% | 43/60 |
| object_counting | 68.33% | 41/60 |
| disambiguation_qa | 66.67% | 40/60 |
| goal_step_wikihow | 66.67% | 40/60 |
| hyperbaton | 65.83% | 79/120 |
| real_or_fake_text | 65.00% | 39/60 |
| logical_deduction | 63.33% | 114/180 |
| snarks | 63.33% | 76/120 |
| causal_judgement | 61.67% | 37/60 |
| nonsense_words_grammar | 61.67% | 37/60 |
| movie_dialog_same_or_different | 61.02% | 36/59 |
| movie_recommendation | 56.67% | 34/60 |
| matrixshapes | 51.67% | 31/60 |
| salient_translation_error_detection | 46.67% | 28/60 |
| word_sorting | 43.33% | 26/60 |
| geometric_shapes | 41.67% | 25/60 |
| discourse_marker_prediction | 33.33% | 20/60 |
| disfl_qa | 28.33% | 17/60 |
| ruin_names | 28.33% | 17/60 |
| language_identification | 25.00% | 15/60 |
| ascii_word_recognition | 18.64% | 11/59 |
| word_unscrambling | 16.67% | 10/60 |
| mnist_ascii | 15.00% | 9/60 |
| dyck_languages | 11.67% | 7/60 |
| simp_turing_concept | 10.00% | 6/60 |
| cryptonite | 8.33% | 5/60 |
| intersect_geometry | 5.00% | 3/60 |
| chess_state_tracking | 0.00% | 0/60 |

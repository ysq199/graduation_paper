# 审稿风险改进记录：ILP 基线、终局奖励、遮挡剔除与质量反馈

## 1. ILP+TSP 强基线

已新增传统优化强基线：

```text
src/trajectory_planning/ilp_tsp_baseline.py
scripts/run_ilp_tsp_viewpoint_baseline.py
```

建模方式：

```text
第一阶段：ILP 加权集合覆盖
目标：在最多 max_selected 个候选视点内最大化 weighted_coverage_ratio

第二阶段：TSP/路径排序
对 ILP 选中的视点使用最近邻初始化 + 2-opt 路径短化
```

正式 5k 实验结果：

| 方法 | selected_viewpoints | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt |
| --- | ---: | ---: | ---: | ---: |
| Greedy 5k | 80 | 0.6850 | 0.7036 | 770.9113 |
| PPO local-jump | 80 | 0.6844 | 0.7099 | 1001.2988 |
| GA 5k | 80 | 0.7566 | 0.7804 | 1189.7653 |
| ILP+TSP 5k | 80 | 0.7740 | 0.7982 | 1173.5145 |

阶段结论：ILP+TSP 是当前静态规划中最强覆盖基线，加权覆盖率高于 GA，路径代价略低于 GA，但仍明显高于 Greedy 和 PPO local-jump。这说明论文中不能再把 PPO 描述为静态覆盖最优方法；PPO 的合理定位应转向质量反馈补拍、动态状态更新和增量决策。

## 2. PPO 终局奖励为主的奖励接口

已在 Maskable PPO 环境中新增终局奖励参数：

```text
reward_intermediate_scale
reward_terminal_coverage_scale
reward_terminal_path_scale
reward_terminal_shot_scale
```

对应文件：

```text
src/trajectory_planning/viewpoint_maskable_ppo.py
scripts/train_maskable_ppo_viewpoints.py
scripts/summarize_experiment_results.py
```

启用后，PPO 可以从原来的“每步增量奖励为主”切换为：

```text
每步：小权重 shaping reward
终局：final weighted coverage - normalized total path - selected count penalty
```

已完成 128 timestep smoke 验证，确认训练、评估和 summary 输出正常。正式 50k 实验尚未运行。

推荐正式实验参数起点：

```text
--gamma 0.99
--reward-intermediate-scale 0.10
--reward-final-coverage-scale 0.0
--reward-terminal-coverage-scale 8.0
--reward-terminal-path-scale 2.0
--reward-terminal-shot-scale 0.1
--reward-motion-scale 0.35
--reward-local-jump-threshold 0.55
--reward-local-jump-scale 2.0
```

## 3. 遮挡剔除骨架

已新增候选视点射线遮挡标注模块：

```text
src/trajectory_planning/viewpoint_visibility.py
scripts/annotate_viewpoint_visibility.py
```

当前实现基于 STL 三角面片做相机点到目标点的射线相交检测，给候选视点增加：

```text
visibility_ratio
occlusion_flag
```

smoke 结果：

```text
input_candidate_count = 160
visible_candidate_count = 117
occluded_candidate_count = 43
visible_ratio = 0.7312
```

阶段结论：遮挡剔除已经可以作为候选视点生成后的过滤/降权步骤。下一步需要把它接入正式 candidate CSV，而不是只输出 smoke 文件。

## 4. 质量反馈补拍规则骨架

已新增质量反馈补拍决策模块：

```text
src/trajectory_planning/quality_feedback.py
scripts/plan_quality_feedback.py
```

当前支持四类触发：

```text
反光/过曝：adjust_lighting
虚焦：adjust_distance
遮挡：change_view_angle 或 mark_uninspectable
覆盖不足：generate_gap_viewpoint
```

同时加入补拍上限机制：

```text
max_retake_attempts = 3
超过上限后标记 mark_uninspectable
```

smoke 验证：

```text
observation_count = 4
retake_count = 3
uninspectable_count = 1
```

阶段结论：补拍机制已经从论文描述变成可编码规则，但还没有接入真实图像质量评价或仿真渲染结果。

## 5. 汇总文件

包含 ILP、终局 PPO smoke 和已有 Greedy/PPO/GA 的汇总文件：

```text
outputs/experiment_summary_with_ilp_terminal.csv
outputs/experiment_summary_with_ilp_terminal.md
```

## 6. 下一步建议

1. 将 ILP+TSP 写入论文实验表，作为静态规划强基线。
2. 不再强调 PPO 静态规划优于 ILP，而是强调 PPO 面向动态补拍和状态更新的增量决策优势。
3. 跑一组 50k terminal-objective PPO 正式实验，验证终局奖励是否能缓解增量奖励的局部贪心问题。
4. 将 `occlusion_flag` 接入候选视点过滤或 priority 降权。
5. 用质量反馈模块构造补拍仿真实验：随机设置低质量区域，比较 ILP 重新规划和 PPO 增量补拍。

# Maskable PPO 实验记录

## 1. 本轮目标

本轮目标是将训练步数提升到 50000 timestep，并通过奖励 shaping 让 Maskable PPO 至少在覆盖率、路径代价或补拍策略中的一项相对贪心 baseline 形成优势。

## 2. 奖励函数调整

在 `src/trajectory_planning/viewpoint_maskable_ppo.py` 中新增 `PPORewardConfig`：

```text
reward =
  gain_scale * marginal_weighted_gain
  - motion_scale * normalized_transition_cost
  - step_penalty
```

终止时额外加入：

```text
达到目标覆盖率：+ target_bonus
达到最大视点数但未达标：+ final_coverage_scale * final_weighted_coverage
```

本轮使用参数：

```text
gain_scale = 10.0
motion_scale = 0.35
step_penalty = 0.005
target_bonus = 3.0
final_coverage_scale = 2.0
```

## 3. 训练命令

```cmd
cd /d E:\graduation_paper\trajectory_planning
python .\scripts\train_maskable_ppo_viewpoints.py --viewpoints-csv ".\outputs\candidate_viewpoints\blade_candidate_viewpoints.csv" --point-features-csv ".\outputs\blade_surface_point_features.csv" --model-output ".\outputs\models\maskable_ppo_viewpoints_50k_shaped.zip" --output-csv ".\outputs\selected_viewpath\blade_selected_viewpath_maskable_ppo_50k_shaped.csv" --output-json ".\outputs\selected_viewpath\blade_selected_viewpath_maskable_ppo_50k_shaped_summary.json" --total-timesteps 50000 --n-steps 512 --batch-size 128 --learning-rate 0.0003 --gamma 0.98 --target-coverage 0.85 --max-selected 80 --min-new-coverage 0.0002 --max-surface-points 5000 --two-opt-iterations 20 --reward-gain-scale 10.0 --reward-motion-scale 0.35 --reward-step-penalty 0.005 --reward-target-bonus 3.0 --reward-final-coverage-scale 2.0
```

## 4. 结果对比

同样使用 5000 个表面点、最多 80 个视点、目标加权覆盖率 0.85。

| 方法 | selected_viewpoints | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt |
| --- | ---: | ---: | ---: | ---: |
| Greedy baseline | 80 | 0.6850 | 0.7036 | 770.9113 |
| Maskable PPO 50k shaped | 80 | 0.6934 | 0.7153 | 1038.3672 |

## 5. 阶段性结论

Maskable PPO 在本轮中已经取得覆盖率优势：

```text
coverage_ratio: 0.6850 -> 0.6934
weighted_coverage_ratio: 0.7036 -> 0.7153
```

但路径代价仍高于贪心 baseline：

```text
path_cost_after_2opt: 770.9113 -> 1038.3672
```

说明当前奖励更偏向关键区域覆盖，运动代价约束还不够强。下一轮建议提高 `reward_motion_scale` 或加入局部邻近动作偏好，使 PPO 在保持覆盖优势的同时降低路径长度。

## 6. 路径约束增强实验

为验证运动代价惩罚对策略的影响，继续保持其他参数不变，仅调整 `reward_motion_scale`：

```text
gain_scale = 10.0
step_penalty = 0.005
target_bonus = 3.0
final_coverage_scale = 2.0
```

同样使用 5000 个表面点、最多 80 个视点、50000 timestep。

| 方法 | reward_motion_scale | selected_viewpoints | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy baseline | - | 80 | 0.6850 | 0.7036 | 770.9113 |
| PPO coverage-first | 0.35 | 80 | 0.6934 | 0.7153 | 1038.3672 |
| PPO motion-enhanced | 0.55 | 80 | 0.6514 | 0.6699 | 855.2276 |
| PPO motion-enhanced | 0.75 | 80 | 0.6706 | 0.6929 | 919.3200 |

### 6.1 结论

单纯提高 `reward_motion_scale` 可以降低 PPO 路径代价，但会明显牺牲覆盖率：

```text
0.35 -> 0.55:
weighted_coverage_ratio: 0.7153 -> 0.6699
path_cost_after_2opt: 1038.3672 -> 855.2276

0.35 -> 0.75:
weighted_coverage_ratio: 0.7153 -> 0.6929
path_cost_after_2opt: 1038.3672 -> 919.3200
```

这说明“全局加大运动惩罚”不是最佳平衡方式。后续更适合采用局部邻近动作偏好：保持覆盖收益权重不变，只对过远跳转视点施加额外惩罚，使策略仍优先保证关键区域覆盖，同时减少不必要的大跨度跳转。

## 7. 局部邻近动作偏好实验

在 `PPORewardConfig` 中新增局部远跳惩罚项：

```text
jump_excess = max(0, normalized_transition_cost - local_jump_threshold)
local_jump_penalty = local_jump_scale * jump_excess^2
```

奖励函数变为：

```text
reward =
  gain_scale * marginal_weighted_gain
  - motion_scale * normalized_transition_cost
  - local_jump_penalty
  - step_penalty
```

该策略保持 `motion_scale = 0.35` 的覆盖优先设置，只额外惩罚超过阈值的远距离跳转。

| 方法 | local_jump_threshold | local_jump_scale | selected_viewpoints | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy baseline | - | - | 80 | 0.6850 | 0.7036 | 770.9113 |
| PPO coverage-first | - | 0.0 | 80 | 0.6934 | 0.7153 | 1038.3672 |
| PPO local-jump | 0.35 | 2.0 | 80 | 0.6680 | 0.6902 | 963.8969 |
| PPO local-jump | 0.55 | 2.0 | 80 | 0.6844 | 0.7099 | 1001.2988 |

### 7.1 结论

局部邻近动作偏好比全局加大 `reward_motion_scale` 更合理。阈值过低时，普通跳转也被惩罚，覆盖率下降明显；提高阈值到 `0.55` 后，可以保住相对贪心 baseline 的加权覆盖优势，同时比覆盖优先 PPO 缩短路径：

```text
相对 Greedy:
weighted_coverage_ratio: 0.7036 -> 0.7099
path_cost_after_2opt: 770.9113 -> 1001.2988

相对 PPO coverage-first:
weighted_coverage_ratio: 0.7153 -> 0.7099
path_cost_after_2opt: 1038.3672 -> 1001.2988
```

当前最可写入论文的表述是：局部邻近偏好在保持覆盖优势的同时缓解了强化学习策略的远距离跳转问题，但路径长度仍未达到贪心基线水平。后续可继续调高 `local_jump_threshold` 或降低 `local_jump_scale`，寻找覆盖率与路径代价的更优折中。

## 8. 遗传算法对比基线

为补全论文对比实验，新增遗传算法 baseline。该方法直接在候选视点序列空间中进行全局搜索，适应度函数为：

```text
fitness = weighted_coverage + target_coverage_bonus - path_weight * normalized_path_cost
```

正式实验同样使用 5000 个表面点、最多 80 个视点、目标加权覆盖率 0.85：

```cmd
cd /d E:\graduation_paper\trajectory_planning
python .\scripts\run_genetic_viewpoint_baseline.py --viewpoints-csv ".\outputs\candidate_viewpoints\blade_candidate_viewpoints.csv" --point-features-csv ".\outputs\blade_surface_point_features.csv" --output-csv ".\outputs\selected_viewpath\blade_selected_viewpath_ga_5k.csv" --output-json ".\outputs\selected_viewpath\blade_selected_viewpath_ga_5k_summary.json" --target-coverage 0.85 --max-selected 80 --min-new-coverage 0.0002 --max-surface-points 5000 --two-opt-iterations 20 --population-size 80 --generations 120 --path-weight 0.10 --target-coverage-bonus 0.15 --ga-seed 23
```

| 方法 | selected_viewpoints | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt |
| --- | ---: | ---: | ---: | ---: |
| Greedy baseline | 80 | 0.6850 | 0.7036 | 770.9113 |
| PPO coverage-first | 80 | 0.6934 | 0.7153 | 1038.3672 |
| PPO local-jump | 80 | 0.6844 | 0.7099 | 1001.2988 |
| Genetic algorithm | 80 | 0.7566 | 0.7804 | 1189.7653 |

阶段结论：遗传算法目前覆盖率最高，说明全局搜索对覆盖目标有优势；但其路径代价也最高，仍需要通过更强的路径项、多目标排序或后处理进一步压缩。因此当前论文叙事可以写成：GA 作为覆盖优先的全局优化对比基线，PPO 的价值主要体现在学习覆盖收益与运动代价之间的策略折中，而局部邻近动作偏好能缓解 PPO 的远跳问题。

## 9. 路径结构约束增强实验

为进一步完善 PPO 奖励函数和动作特征表示，本轮在原有局部邻近动作偏好的基础上，增加路径结构相关信息：

```text
候选动作特征新增：
nearby_score：候选视点相对当前位置的邻近程度
same_region：候选视点是否与当前视点属于同一 region_type
direction_alignment：从上一视点到当前视点、再到候选视点的运动方向一致性
```

奖励函数新增两项：

```text
smoothness_penalty = reward_smoothness_scale * turn_penalty
region_switch_penalty = reward_region_switch_scale * region_switch_penalty_raw
```

其中，`turn_penalty = 1 - direction_alignment`，用于惩罚尖锐转向；`region_switch_penalty_raw` 用于轻度惩罚频繁跨区域跳转。本轮实验保持原 local-jump 配置：

```text
reward_motion_scale = 0.35
reward_local_jump_threshold = 0.55
reward_local_jump_scale = 2.0
reward_smoothness_scale = 0.10
reward_region_switch_scale = 0.02
```

正式实验同样使用 5000 个表面点、最多 80 个视点、50000 timestep：

| 方法 | selected_viewpoints | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt |
| --- | ---: | ---: | ---: | ---: |
| Greedy baseline | 80 | 0.6850 | 0.7036 | 770.9113 |
| PPO local-jump | 80 | 0.6844 | 0.7099 | 1001.2988 |
| PPO structure s0.10 r0.02 | 80 | 0.6800 | 0.7077 | 1080.2185 |

### 9.1 结论

该组结构约束实验没有优于当前最好的 local-jump 配置。相对 Greedy baseline，结构约束 PPO 的加权覆盖率仍略高：

```text
weighted_coverage_ratio: 0.7036 -> 0.7077
```

但相对 PPO local-jump，覆盖率略降且路径代价升高：

```text
weighted_coverage_ratio: 0.7099 -> 0.7077
path_cost_after_2opt: 1001.2988 -> 1080.2185
```

这说明当前 `smoothness_scale = 0.10` 与 `region_switch_scale = 0.02` 的组合并没有稳定压缩最终 2-opt 后路径，反而可能改变了策略选点分布，使高收益视点之间的空间组织变差。后续不建议继续增大该结构惩罚，而应采用更温和的设置，例如：

```text
reward_smoothness_scale = 0.03 ~ 0.05
reward_region_switch_scale = 0.00 ~ 0.01
```

当前论文叙事可写为：路径结构特征已接入 PPO 状态和奖励，但第一组结构惩罚参数未取得优于 local-jump 的结果，说明路径连续性约束需要更精细调参，不能简单叠加较强惩罚项。

## 10. 终局奖励为主的 PPO 实验

为验证“增量奖励容易导致局部贪心”的风险，本轮将 PPO 奖励改为终局目标为主、中间 shaping 为辅。训练参数为：

```text
gamma = 0.99
reward_intermediate_scale = 0.10
reward_final_coverage_scale = 0.0
reward_terminal_coverage_scale = 8.0
reward_terminal_path_scale = 2.0
reward_terminal_shot_scale = 0.1
reward_motion_scale = 0.35
reward_local_jump_threshold = 0.55
reward_local_jump_scale = 2.0
```

正式实验同样使用 5000 个表面点、最多 80 个视点、50000 timestep：

| 方法 | selected_viewpoints | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt |
| --- | ---: | ---: | ---: | ---: |
| Greedy baseline | 80 | 0.6850 | 0.7036 | 770.9113 |
| PPO coverage-first | 80 | 0.6934 | 0.7153 | 1038.3672 |
| PPO local-jump | 80 | 0.6844 | 0.7099 | 1001.2988 |
| PPO terminal c8 p2 | 80 | 0.6336 | 0.6703 | 1153.5403 |

### 10.1 结论

该组终局奖励参数没有取得预期效果。相对 local-jump 配置，终局奖励 PPO 的加权覆盖率明显下降，路径代价反而升高：

```text
weighted_coverage_ratio: 0.7099 -> 0.6703
path_cost_after_2opt: 1001.2988 -> 1153.5403
```

这说明当前终局奖励设置中 `terminal_path_scale = 2.0` 与较低的 `intermediate_reward_scale = 0.10` 可能削弱了覆盖收益的逐步学习信号，使策略未能稳定学习高覆盖视点组合。后续若继续尝试终局奖励，应采用更温和的参数，例如：

```text
reward_intermediate_scale = 0.30 ~ 0.50
reward_terminal_coverage_scale = 4.0 ~ 6.0
reward_terminal_path_scale = 0.5 ~ 1.0
```

当前可写入论文的结论是：单纯将奖励改为终局目标为主并不能自动改善 PPO 的全局规划能力，终局奖励与中间覆盖 shaping 之间需要保持平衡；在当前数据上，coverage-first 和 local-jump 仍是更稳定的 PPO 版本。

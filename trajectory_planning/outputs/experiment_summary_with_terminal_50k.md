# Experiment Summary

| name | baseline | selected_view_count | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt | reward_motion_scale | reward_local_jump_threshold | reward_local_jump_scale | reward_smoothness_scale | reward_region_switch_scale | reward_intermediate_scale | reward_terminal_coverage_scale | reward_terminal_path_scale | ga_generations | ga_path_weight | ilp_time_limit | ilp_mip_gap | ilp_solver_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blade_selected_viewpath_ga_5k | genetic_algorithm | 80 | 0.7566 | 0.7804 | 1189.7653 |  |  |  |  |  |  |  |  | 120 | 0.1000 |  |  |  |
| blade_selected_viewpath_ga_smoke | genetic_algorithm | 20 | 0.2590 | 0.3220 | 223.1795 |  |  |  |  |  |  |  |  | 4 | 0.1000 |  |  |  |
| blade_selected_viewpath_greedy_5k | greedy_env | 80 | 0.6850 | 0.7036 | 770.9113 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_rl_baseline | greedy_env | 80 | 0.6220 | 0.6479 | 755.4387 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_ilp_tsp_5k | ilp_tsp | 80 | 0.7740 | 0.7982 | 1173.5145 |  |  |  |  |  |  |  |  |  |  | 300.0000 | 0.0050 | True |
| blade_selected_viewpath_maskable_ppo_50k_shaped | maskable_ppo | 80 | 0.6934 | 0.7153 | 1038.3672 | 0.3500 |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_50k_localjump2_thr055 | maskable_ppo | 80 | 0.6844 | 0.7099 | 1001.2988 | 0.3500 | 0.5500 | 2.0000 |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_50k_structure_s010_r002 | maskable_ppo | 80 | 0.6800 | 0.7077 | 1080.2185 | 0.3500 | 0.5500 | 2.0000 | 0.1000 | 0.0200 |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_50k_motion075 | maskable_ppo | 80 | 0.6706 | 0.6929 | 919.3200 | 0.7500 |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_50k_localjump2 | maskable_ppo | 80 | 0.6680 | 0.6902 | 963.8969 | 0.3500 | 0.3500 | 2.0000 |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_50k_terminal_c8_p2 | maskable_ppo | 80 | 0.6336 | 0.6703 | 1153.5403 | 0.3500 | 0.5500 | 2.0000 | 0.0000 | 0.0000 | 0.1000 | 8.0000 | 2.0000 |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_50k_motion055 | maskable_ppo | 80 | 0.6514 | 0.6699 | 855.2276 | 0.5500 |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_1k | maskable_ppo | 80 | 0.5858 | 0.6244 | 1105.3948 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_terminal_smoke | maskable_ppo | 10 | 0.1260 | 0.1793 | 254.9470 | 0.3500 | 0.5500 | 2.0000 | 0.0000 | 0.0000 | 0.1000 | 8.0000 | 2.0000 |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_localjump_smoke | maskable_ppo | 10 | 0.1063 | 0.1178 | 279.4389 | 0.3500 | 0.3500 | 2.0000 |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_reward_smoke | maskable_ppo | 10 | 0.1063 | 0.1178 | 279.4389 | 0.3500 |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_smoke | maskable_ppo | 10 | 0.1072 | 0.1145 | 328.6512 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_full | unknown | 134 | 0.8421 | 0.8619 | 1551.2670 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_full_fast | unknown | 134 | 0.8421 | 0.8619 | 1798.6696 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath | unknown | 80 | 0.6220 | 0.6479 | 755.4387 |  |  |  |  |  |  |  |  |  |  |  |  |  |

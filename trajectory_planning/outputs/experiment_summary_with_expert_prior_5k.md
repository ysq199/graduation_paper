# Experiment Summary

| name | baseline | selected_view_count | coverage_ratio | weighted_coverage_ratio | path_cost_after_2opt | reward_motion_scale | reward_local_jump_threshold | reward_local_jump_scale | reward_smoothness_scale | reward_region_switch_scale | reward_intermediate_scale | reward_terminal_coverage_scale | reward_terminal_path_scale | reward_terminal_shot_scale | reward_expert_prior_scale | reward_expert_next_scale | expert_route_count | ga_generations | ga_path_weight | ilp_time_limit | ilp_mip_gap | ilp_solver_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blade_selected_viewpath_ilp_tsp_5k | ilp_tsp | 80 | 0.7740 | 0.7982 | 1173.5145 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 300.0000 | 0.0050 | True |
| blade_selected_viewpath_ilp_tsp_smoke | ilp_tsp | 20 | 0.3520 | 0.4264 | 547.5836 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 30.0000 | 0.0200 | True |
| blade_selected_viewpath_maskable_ppo_50k_structure_s010_r002 | maskable_ppo | 80 | 0.6800 | 0.7077 | 1080.2185 | 0.3500 | 0.5500 | 2.0000 | 0.1000 | 0.0200 |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_expert_prior_5k | maskable_ppo | 80 | 0.6649 | 0.6980 | 1176.4333 | 0.3500 | 0.3500 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 8.0000 | 2.0000 | 0.0000 | 0.0500 | 0.0500 | 80 |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_50k_terminal_c8_p2 | maskable_ppo | 80 | 0.6336 | 0.6703 | 1153.5403 | 0.3500 | 0.5500 | 2.0000 | 0.0000 | 0.0000 | 0.1000 | 8.0000 | 2.0000 | 0.1000 |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_expert_prior_smoke | maskable_ppo | 80 | 0.6296 | 0.6640 | 1169.8532 | 0.3500 | 0.3500 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 8.0000 | 2.0000 | 0.0000 | 0.0500 | 0.0500 | 80 |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_structure_smoke | maskable_ppo | 10 | 0.1240 | 0.1809 | 230.8823 | 0.3500 | 0.5500 | 2.0000 | 0.1500 | 0.0200 |  |  |  |  |  |  |  |  |  |  |  |  |
| blade_selected_viewpath_maskable_ppo_terminal_smoke | maskable_ppo | 10 | 0.1260 | 0.1793 | 254.9470 | 0.3500 | 0.5500 | 2.0000 | 0.0000 | 0.0000 | 0.1000 | 8.0000 | 2.0000 | 0.1000 |  |  |  |  |  |  |  |  |

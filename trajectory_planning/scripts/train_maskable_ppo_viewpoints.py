from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.viewpoint_selection import (  # noqa: E402
    SelectionConfig,
    load_point_feature_csv,
    load_viewpoints_csv,
    write_selected_viewpoints_csv,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Maskable PPO for viewpoint selection.")
    parser.add_argument("--viewpoints-csv", type=Path, required=True, help="Candidate viewpoint CSV.")
    parser.add_argument("--point-features-csv", type=Path, required=True, help="Surface point feature CSV.")
    parser.add_argument("--model-output", type=Path, required=True, help="Output path for the trained model zip.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Evaluated ordered viewpoint path CSV.")
    parser.add_argument("--output-json", type=Path, help="Optional evaluation summary JSON.")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Training timesteps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size.")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor.")
    parser.add_argument("--reward-gain-scale", type=float, default=10.0, help="Scale for marginal weighted coverage gain.")
    parser.add_argument("--reward-motion-scale", type=float, default=0.35, help="Scale for normalized transition-cost penalty.")
    parser.add_argument("--reward-step-penalty", type=float, default=0.005, help="Per-shot penalty.")
    parser.add_argument("--reward-target-bonus", type=float, default=3.0, help="Terminal bonus when target coverage is reached.")
    parser.add_argument("--reward-final-coverage-scale", type=float, default=2.0, help="Truncated-episode bonus from final weighted coverage.")
    parser.add_argument("--reward-local-jump-threshold", type=float, default=0.35, help="No extra local-jump penalty below this normalized transition cost.")
    parser.add_argument("--reward-local-jump-scale", type=float, default=0.0, help="Quadratic penalty scale for normalized transition cost above local-jump threshold.")
    parser.add_argument("--reward-smoothness-scale", type=float, default=0.0, help="Penalty scale for sharp turn changes between consecutive moves.")
    parser.add_argument("--reward-region-switch-scale", type=float, default=0.0, help="Penalty scale for switching candidate region types between consecutive moves.")
    parser.add_argument("--reward-intermediate-scale", type=float, default=1.0, help="Scale for per-step shaping reward; use a small value for terminal-objective-first PPO.")
    parser.add_argument("--reward-terminal-coverage-scale", type=float, default=0.0, help="Episode-end reward scale for final weighted coverage.")
    parser.add_argument("--reward-terminal-path-scale", type=float, default=0.0, help="Episode-end penalty scale for normalized raw path cost.")
    parser.add_argument("--reward-terminal-shot-scale", type=float, default=0.0, help="Episode-end penalty scale for normalized selected viewpoint count.")
    parser.add_argument("--expert-route-csv", type=Path, help="Optional selected-viewpath CSV used as a structural expert route prior.")
    parser.add_argument("--reward-expert-prior-scale", type=float, default=0.0, help="Small reward for selecting viewpoints that appear in the expert route.")
    parser.add_argument("--reward-expert-next-scale", type=float, default=0.0, help="Small reward for following the next unvisited segment of the expert route.")
    parser.add_argument("--target-coverage", type=float, default=0.85, help="Target weighted surface coverage ratio.")
    parser.add_argument("--max-selected", type=int, default=80, help="Maximum selected viewpoints.")
    parser.add_argument("--min-new-coverage", type=float, default=0.002, help="Stop if best marginal coverage is below this.")
    parser.add_argument("--motion-weight", type=float, default=0.12, help="Penalty weight for transition motion cost.")
    parser.add_argument("--orientation-weight", type=float, default=2.0, help="Orientation-change cost in route optimization.")
    parser.add_argument("--curvature-weight", type=float, default=1.0, help="Surface-point weight for high-curvature areas.")
    parser.add_argument("--boundary-weight", type=float, default=1.0, help="Surface-point weight for ROI boundary areas.")
    parser.add_argument("--roughness-weight", type=float, default=0.20, help="Surface-point weight for rough areas.")
    parser.add_argument("--max-surface-points", type=int, default=50000, help="Max surface points used for coverage modeling.")
    parser.add_argument("--two-opt-iterations", type=int, default=50, help="2-opt path optimization iterations.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--no-two-opt", action="store_true", help="Keep raw policy rollout order without 2-opt shortening.")
    return parser.parse_args()


def load_expert_order(path: Path | None, candidates: list[dict[str, object]]) -> list[int]:
    if path is None:
        return []
    view_id_to_index = {int(candidate["view_id"]): idx for idx, candidate in enumerate(candidates)}
    expert_order: list[int] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Expert route CSV has no header: {path}")
        id_field = "candidate_view_id" if "candidate_view_id" in reader.fieldnames else "view_id"
        if id_field not in reader.fieldnames:
            raise ValueError(f"Expert route CSV must contain candidate_view_id or view_id: {path}")
        for row in reader:
            raw_view_id = row.get(id_field, "")
            if raw_view_id == "":
                continue
            view_id = int(float(raw_view_id))
            candidate_index = view_id_to_index.get(view_id)
            if candidate_index is None or candidate_index in seen:
                continue
            expert_order.append(candidate_index)
            seen.add(candidate_index)
    return expert_order


def main() -> int:
    args = parse_args()
    try:
        from sb3_contrib import MaskablePPO
        from trajectory_planning.viewpoint_maskable_ppo import (
            MaskablePPOViewpointEnv,
            PPORewardConfig,
            evaluate_maskable_model,
        )
    except ImportError as exc:
        print("Missing optional Maskable PPO dependencies.")
        print("Install them before training:")
        print("python -m pip install gymnasium stable-baselines3 sb3-contrib")
        print(f"Import error: {exc}")
        return 2

    candidates = load_viewpoints_csv(args.viewpoints_csv)
    _, point_features = load_point_feature_csv(args.point_features_csv)
    config = SelectionConfig(
        target_coverage=args.target_coverage,
        max_selected=args.max_selected,
        min_new_coverage=args.min_new_coverage,
        motion_weight=args.motion_weight,
        orientation_weight=args.orientation_weight,
        curvature_weight=args.curvature_weight,
        boundary_weight=args.boundary_weight,
        roughness_weight=args.roughness_weight,
        max_surface_points=args.max_surface_points,
        two_opt_iterations=args.two_opt_iterations,
        seed=args.seed,
    )
    reward_config = PPORewardConfig(
        gain_scale=args.reward_gain_scale,
        motion_scale=args.reward_motion_scale,
        step_penalty=args.reward_step_penalty,
        target_bonus=args.reward_target_bonus,
        final_coverage_scale=args.reward_final_coverage_scale,
        local_jump_threshold=args.reward_local_jump_threshold,
        local_jump_scale=args.reward_local_jump_scale,
        smoothness_scale=args.reward_smoothness_scale,
        region_switch_scale=args.reward_region_switch_scale,
        intermediate_reward_scale=args.reward_intermediate_scale,
        terminal_coverage_scale=args.reward_terminal_coverage_scale,
        terminal_path_scale=args.reward_terminal_path_scale,
        terminal_shot_scale=args.reward_terminal_shot_scale,
        expert_prior_scale=args.reward_expert_prior_scale,
        expert_next_scale=args.reward_expert_next_scale,
    )
    expert_order = load_expert_order(args.expert_route_csv, candidates)
    env = MaskablePPOViewpointEnv(candidates, point_features, config, reward_config, expert_order=expert_order)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)

    selected_rows, summary = evaluate_maskable_model(model, env, optimize_route=not args.no_two_opt)
    summary["expert_route_csv"] = "" if args.expert_route_csv is None else str(args.expert_route_csv)
    write_selected_viewpoints_csv(args.output_csv, selected_rows)
    if args.output_json:
        write_summary_json(args.output_json, summary)

    print(f"baseline={summary['baseline']}")
    print(f"expert_route_count={summary['expert_route_count']}")
    print(f"model_output={args.model_output}")
    print(f"selected_viewpoints={summary['selected_view_count']}")
    print(f"coverage_ratio={summary['coverage_ratio']:.4f}")
    print(f"weighted_coverage_ratio={summary['weighted_coverage_ratio']:.4f}")
    print(f"total_reward={summary['total_reward']:.4f}")
    print(f"terminated={summary['terminated']}")
    print(f"truncated={summary['truncated']}")
    print(f"path_cost_before_2opt={summary['path_cost_before_2opt']:.4f}")
    print(f"path_cost_after_2opt={summary['path_cost_after_2opt']:.4f}")
    print(f"output_csv={args.output_csv}")
    if args.output_json:
        print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

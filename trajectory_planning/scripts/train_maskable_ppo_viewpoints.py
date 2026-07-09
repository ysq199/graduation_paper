from __future__ import annotations

import argparse
import csv
import shutil
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
    parser.add_argument("--reward-expert-edge-scale", type=float, default=0.0, help="Small reward for taking a candidate edge close to the expert route's next local edge.")
    parser.add_argument("--reward-expert-edge-jump-scale", type=float, default=0.0, help="Penalty scale for large forward or backward jumps along the expert route order.")
    parser.add_argument("--reward-expert-edge-bandwidth", type=float, default=3.0, help="Rank-distance bandwidth used by expert edge locality features.")
    parser.add_argument("--expert-edge-mask-window", type=int, default=0, help="Enable structured action masking by allowing this many expert-route successors near the next local edge.")
    parser.add_argument("--expert-edge-mask-min-candidates", type=int, default=4, help="Minimum number of actions kept by structured expert-edge masking.")
    parser.add_argument("--expert-edge-mask-escape-ratio", type=float, default=0.0, help="Allow nonlocal actions whose marginal gain is at least this ratio of the current best marginal gain.")
    parser.add_argument("--expert-edge-mask-escape-motion-weight", type=float, default=0.0, help="Motion-cost weight used when scoring nonlocal coverage escape actions.")
    parser.add_argument("--expert-edge-mask-escape-max-motion", type=float, default=1.0, help="Maximum normalized transition cost allowed for nonlocal coverage escape actions.")
    parser.add_argument("--expert-edge-mask-start-step", type=int, default=0, help="Episode step from which structured expert-edge masking becomes active.")
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
    parser.add_argument("--checkpoint-eval-interval", type=int, default=0, help="Train/evaluate checkpoints every N timesteps; disabled when 0.")
    parser.add_argument("--checkpoint-output-dir", type=Path, help="Directory for checkpoint models, CSVs and JSON summaries.")
    parser.add_argument("--checkpoint-prefix", type=str, help="Filename prefix for checkpoint artifacts.")
    parser.add_argument("--checkpoint-score-path-weight", type=float, default=2.0, help="Weight for normalized 2-opt path cost in checkpoint score.")
    parser.add_argument("--checkpoint-index-csv", type=Path, help="Optional CSV index of checkpoint evaluation results.")
    parser.add_argument("--best-model-output", type=Path, help="Optional output path for the best checkpoint model zip.")
    parser.add_argument("--best-output-csv", type=Path, help="Optional output CSV for the best checkpoint route.")
    parser.add_argument("--best-output-json", type=Path, help="Optional output JSON for the best checkpoint summary.")
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


def add_checkpoint_score(
    summary: dict[str, object],
    env: object,
    checkpoint_timesteps: int,
    path_weight: float,
    selected: bool = False,
) -> None:
    max_route = env.core.bbox_diag * max(float(env.config.max_selected - 1), 1.0)
    normalized_path = float(summary["path_cost_after_2opt"]) / max(float(max_route), 1e-12)
    score = float(summary["weighted_coverage_ratio"]) - float(path_weight) * normalized_path
    summary["checkpoint_timesteps"] = int(checkpoint_timesteps)
    summary["checkpoint_score_path_weight"] = float(path_weight)
    summary["checkpoint_normalized_path_after_2opt"] = float(normalized_path)
    summary["checkpoint_score"] = float(score)
    summary["checkpoint_selected"] = bool(selected)


def checkpoint_prefix(args: argparse.Namespace) -> str:
    return args.checkpoint_prefix or args.model_output.stem


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return args.checkpoint_output_dir or args.model_output.parent / f"{args.model_output.stem}_checkpoints"


def checkpoint_paths(args: argparse.Namespace, timesteps: int) -> tuple[Path, Path, Path]:
    prefix = checkpoint_prefix(args)
    base_dir = checkpoint_dir(args)
    return (
        base_dir / "models" / f"{prefix}_{timesteps}.zip",
        base_dir / "selected_viewpath" / f"{prefix}_{timesteps}.csv",
        base_dir / "selected_viewpath" / f"{prefix}_{timesteps}_summary.json",
    )


def write_checkpoint_index(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint_timesteps",
        "checkpoint_score",
        "weighted_coverage_ratio",
        "path_cost_after_2opt",
        "checkpoint_normalized_path_after_2opt",
        "checkpoint_selected",
        "checkpoint_model_path",
        "checkpoint_summary_json",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})


def train_with_checkpoint_evaluation(
    model: object,
    env: object,
    eval_env: object,
    args: argparse.Namespace,
    expert_route_csv: Path | None,
    optimize_route: bool,
    evaluate_maskable_model: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    interval = int(args.checkpoint_eval_interval)
    if interval <= 0:
        model.learn(total_timesteps=args.total_timesteps)
        return evaluate_maskable_model(model, eval_env, optimize_route=optimize_route)

    summaries: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] | None = None
    best_summary: dict[str, object] | None = None
    best_model_path: Path | None = None

    while int(model.num_timesteps) < int(args.total_timesteps):
        remaining = max(int(args.total_timesteps) - int(model.num_timesteps), 1)
        chunk_steps = min(interval, remaining)
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=(int(model.num_timesteps) == 0))
        current_timesteps = int(model.num_timesteps)

        model_path, csv_path, json_path = checkpoint_paths(args, current_timesteps)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        selected_rows, summary = evaluate_maskable_model(model, eval_env, optimize_route=optimize_route)
        summary["expert_route_csv"] = "" if expert_route_csv is None else str(expert_route_csv)
        add_checkpoint_score(summary, eval_env, current_timesteps, args.checkpoint_score_path_weight)
        summary["checkpoint_model_path"] = str(model_path)
        summary["checkpoint_summary_json"] = str(json_path)
        write_selected_viewpoints_csv(csv_path, selected_rows)
        write_summary_json(json_path, summary)
        summaries.append(summary)

        if best_summary is None or float(summary["checkpoint_score"]) > float(best_summary["checkpoint_score"]):
            best_rows = selected_rows
            best_summary = dict(summary)
            best_model_path = model_path

        print(
            "checkpoint="
            f"{current_timesteps} weighted_coverage={summary['weighted_coverage_ratio']:.4f} "
            f"path_after_2opt={summary['path_cost_after_2opt']:.4f} "
            f"score={summary['checkpoint_score']:.4f}"
        )

    if best_rows is None or best_summary is None or best_model_path is None:
        raise RuntimeError("No checkpoint was evaluated.")

    best_summary["checkpoint_selected"] = True
    if args.best_model_output:
        args.best_model_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_model_path, args.best_model_output)
        best_summary["best_model_output"] = str(args.best_model_output)
    if args.best_output_csv:
        write_selected_viewpoints_csv(args.best_output_csv, best_rows)
        best_summary["best_output_csv"] = str(args.best_output_csv)
    if args.best_output_json:
        best_summary["best_output_json"] = str(args.best_output_json)
        write_summary_json(args.best_output_json, best_summary)

    if args.checkpoint_index_csv:
        for summary in summaries:
            summary["checkpoint_selected"] = int(summary["checkpoint_timesteps"]) == int(best_summary["checkpoint_timesteps"])
        write_checkpoint_index(args.checkpoint_index_csv, summaries)

    return best_rows, best_summary


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
        expert_edge_scale=args.reward_expert_edge_scale,
        expert_edge_jump_scale=args.reward_expert_edge_jump_scale,
        expert_edge_bandwidth=args.reward_expert_edge_bandwidth,
        expert_edge_mask_window=args.expert_edge_mask_window,
        expert_edge_mask_min_candidates=args.expert_edge_mask_min_candidates,
        expert_edge_mask_escape_ratio=args.expert_edge_mask_escape_ratio,
        expert_edge_mask_escape_motion_weight=args.expert_edge_mask_escape_motion_weight,
        expert_edge_mask_escape_max_motion=args.expert_edge_mask_escape_max_motion,
        expert_edge_mask_start_step=args.expert_edge_mask_start_step,
    )
    expert_order = load_expert_order(args.expert_route_csv, candidates)
    env = MaskablePPOViewpointEnv(candidates, point_features, config, reward_config, expert_order=expert_order)
    eval_env = MaskablePPOViewpointEnv(candidates, point_features, config, reward_config, expert_order=expert_order)
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
    selected_rows, summary = train_with_checkpoint_evaluation(
        model,
        env,
        eval_env,
        args,
        args.expert_route_csv,
        optimize_route=not args.no_two_opt,
        evaluate_maskable_model=evaluate_maskable_model,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)

    summary["expert_route_csv"] = "" if args.expert_route_csv is None else str(args.expert_route_csv)
    if "checkpoint_timesteps" not in summary:
        add_checkpoint_score(summary, eval_env, int(model.num_timesteps), args.checkpoint_score_path_weight)
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

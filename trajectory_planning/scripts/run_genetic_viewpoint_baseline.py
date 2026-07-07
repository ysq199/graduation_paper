from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.genetic_viewpoint_baseline import GeneticConfig, run_genetic_baseline  # noqa: E402
from trajectory_planning.viewpoint_selection import (  # noqa: E402
    SelectionConfig,
    load_point_feature_csv,
    load_viewpoints_csv,
    write_selected_viewpoints_csv,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a genetic-algorithm baseline for ordered viewpoint selection.")
    parser.add_argument("--viewpoints-csv", type=Path, required=True, help="Candidate viewpoint CSV.")
    parser.add_argument("--point-features-csv", type=Path, required=True, help="Surface point feature CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Selected ordered viewpoint path CSV.")
    parser.add_argument("--output-json", type=Path, help="Optional baseline summary JSON.")
    parser.add_argument("--target-coverage", type=float, default=0.85, help="Target weighted surface coverage ratio.")
    parser.add_argument("--max-selected", type=int, default=80, help="Chromosome length / maximum selected viewpoints.")
    parser.add_argument("--min-new-coverage", type=float, default=0.0002, help="Greedy seed stop threshold.")
    parser.add_argument("--motion-weight", type=float, default=0.12, help="Greedy seed motion weight.")
    parser.add_argument("--orientation-weight", type=float, default=2.0, help="Orientation-change cost in route optimization.")
    parser.add_argument("--curvature-weight", type=float, default=1.0, help="Surface-point weight for high-curvature areas.")
    parser.add_argument("--boundary-weight", type=float, default=1.0, help="Surface-point weight for ROI boundary areas.")
    parser.add_argument("--roughness-weight", type=float, default=0.20, help="Surface-point weight for rough areas.")
    parser.add_argument("--max-surface-points", type=int, default=5000, help="Max surface points used for coverage modeling.")
    parser.add_argument("--two-opt-iterations", type=int, default=20, help="2-opt path optimization iterations.")
    parser.add_argument("--seed", type=int, default=17, help="Selection/downsampling random seed.")
    parser.add_argument("--population-size", type=int, default=80, help="GA population size.")
    parser.add_argument("--generations", type=int, default=120, help="GA generations.")
    parser.add_argument("--tournament-size", type=int, default=4, help="Tournament selection size.")
    parser.add_argument("--elite-count", type=int, default=4, help="Number of elite chromosomes kept each generation.")
    parser.add_argument("--crossover-rate", type=float, default=0.85, help="Ordered crossover probability.")
    parser.add_argument("--mutation-rate", type=float, default=0.25, help="Mutation probability.")
    parser.add_argument("--path-weight", type=float, default=0.10, help="Fitness penalty weight for normalized path cost.")
    parser.add_argument("--target-coverage-bonus", type=float, default=0.15, help="Fitness bonus when target coverage is reached.")
    parser.add_argument("--ga-seed", type=int, default=23, help="GA random seed.")
    parser.add_argument("--no-two-opt", action="store_true", help="Keep raw GA order without 2-opt shortening.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = load_viewpoints_csv(args.viewpoints_csv)
    _, point_features = load_point_feature_csv(args.point_features_csv)
    selection_config = SelectionConfig(
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
    genetic_config = GeneticConfig(
        population_size=args.population_size,
        generations=args.generations,
        tournament_size=args.tournament_size,
        elite_count=args.elite_count,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        path_weight=args.path_weight,
        target_coverage_bonus=args.target_coverage_bonus,
        seed=args.ga_seed,
    )
    selected_rows, summary = run_genetic_baseline(
        candidates,
        point_features,
        selection_config,
        genetic_config,
        optimize_route=not args.no_two_opt,
    )
    write_selected_viewpoints_csv(args.output_csv, selected_rows)
    if args.output_json:
        write_summary_json(args.output_json, summary)

    print(f"baseline={summary['baseline']}")
    print(f"selected_viewpoints={summary['selected_view_count']}")
    print(f"coverage_ratio={summary['coverage_ratio']:.4f}")
    print(f"weighted_coverage_ratio={summary['weighted_coverage_ratio']:.4f}")
    print(f"fitness={summary['fitness']:.4f}")
    print(f"path_cost_before_2opt={summary['path_cost_before_2opt']:.4f}")
    print(f"path_cost_after_2opt={summary['path_cost_after_2opt']:.4f}")
    print(f"output_csv={args.output_csv}")
    if args.output_json:
        print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible viewpoint-planning experiment pipeline.")
    parser.add_argument("--stl", type=Path, default=ROOT / "model" / "CRB.STL", help="Input blade STL model.")
    parser.add_argument("--points", type=int, default=50000, help="Number of STL surface points to sample.")
    parser.add_argument("--candidate-views", type=int, default=160, help="Number of generated candidate viewpoints.")
    parser.add_argument("--working-distance", type=float, default=20.0, help="Camera/endoscope working distance.")
    parser.add_argument("--hfov-deg", type=float, default=55.0, help="Horizontal FOV.")
    parser.add_argument("--vfov-deg", type=float, default=40.0, help="Vertical FOV.")
    parser.add_argument("--overlap", type=float, default=0.35, help="Expected image overlap.")
    parser.add_argument("--max-surface-points", type=int, default=50000, help="Max points used in path selection.")
    parser.add_argument("--run-ilp", action="store_true", help="Also run the ILP weighted set cover + TSP comparison baseline.")
    parser.add_argument("--ilp-output-name", default="blade_selected_viewpath_ilp_tsp_5k", help="Output basename for ILP CSV/JSON files.")
    parser.add_argument("--ilp-max-surface-points", type=int, default=5000, help="Max points used by the ILP coverage model.")
    parser.add_argument("--ilp-time-limit", type=float, default=300.0, help="ILP solver time limit in seconds.")
    parser.add_argument("--ilp-mip-gap", type=float, default=0.005, help="ILP relative MIP gap tolerance.")
    parser.add_argument("--run-ga", action="store_true", help="Also run the genetic-algorithm comparison baseline.")
    parser.add_argument("--ga-output-name", default="blade_selected_viewpath_ga_5k", help="Output basename for GA CSV/JSON files.")
    parser.add_argument("--ga-max-surface-points", type=int, default=5000, help="Max points used by the GA coverage model.")
    parser.add_argument("--ga-population-size", type=int, default=80, help="GA population size.")
    parser.add_argument("--ga-generations", type=int, default=120, help="GA generations.")
    parser.add_argument("--ga-path-weight", type=float, default=0.10, help="GA normalized path-cost penalty weight.")
    parser.add_argument("--ga-seed", type=int, default=23, help="GA random seed.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a step when all declared output files already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def run_step(
    name: str,
    command: list[str],
    outputs: list[Path],
    skip_existing: bool,
    dry_run: bool,
) -> None:
    if skip_existing and outputs and all(path.exists() for path in outputs):
        print(f"[skip] {name}")
        return

    print(f"[run] {name}")
    print(" ".join(str(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    py = sys.executable

    surface_points = ROOT / "outputs" / "blade_surface_points.csv"
    surface_features = ROOT / "outputs" / "blade_surface_features.json"
    point_features = ROOT / "outputs" / "blade_surface_point_features.csv"
    candidate_csv = ROOT / "outputs" / "candidate_viewpoints" / "blade_candidate_viewpoints.csv"
    candidate_json = ROOT / "outputs" / "candidate_viewpoints" / "blade_candidate_viewpoints_summary.json"
    selected_csv = ROOT / "outputs" / "selected_viewpath" / "blade_selected_viewpath.csv"
    selected_json = ROOT / "outputs" / "selected_viewpath" / "blade_selected_viewpath_summary.json"
    selected_full_csv = ROOT / "outputs" / "selected_viewpath" / "blade_selected_viewpath_full.csv"
    selected_full_json = ROOT / "outputs" / "selected_viewpath" / "blade_selected_viewpath_full_summary.json"
    ga_csv = ROOT / "outputs" / "selected_viewpath" / f"{args.ga_output_name}.csv"
    ga_json = ROOT / "outputs" / "selected_viewpath" / f"{args.ga_output_name}_summary.json"
    ga_visual_dir = ROOT / "outputs" / "selected_viewpath" / f"visualization_{args.ga_output_name.removeprefix('blade_selected_viewpath_')}"
    ilp_csv = ROOT / "outputs" / "selected_viewpath" / f"{args.ilp_output_name}.csv"
    ilp_json = ROOT / "outputs" / "selected_viewpath" / f"{args.ilp_output_name}_summary.json"
    ilp_visual_dir = ROOT / "outputs" / "selected_viewpath" / f"visualization_{args.ilp_output_name.removeprefix('blade_selected_viewpath_')}"

    steps: list[tuple[str, list[str], list[Path]]] = [
        (
            "sample STL surface",
            [
                py,
                str(ROOT / "scripts" / "sample_stl_point_cloud.py"),
                "--stl",
                str(args.stl),
                "--output",
                str(surface_points),
                "--points",
                str(args.points),
            ],
            [surface_points],
        ),
        (
            "extract surface features",
            [
                py,
                str(ROOT / "scripts" / "extract_roi_features.py"),
                "--point-cloud",
                str(surface_points),
                "--output-json",
                str(surface_features),
                "--output-point-features-csv",
                str(point_features),
            ],
            [surface_features, point_features],
        ),
        (
            "visualize surface features",
            [
                py,
                str(ROOT / "scripts" / "visualize_roi_features.py"),
                "--features-json",
                str(surface_features),
                "--point-features-csv",
                str(point_features),
                "--output-dir",
                str(ROOT / "outputs" / "blade_visualization"),
                "--max-points",
                "20000",
            ],
            [
                ROOT / "outputs" / "blade_visualization" / "surface_3d_curvature.png",
                ROOT / "outputs" / "blade_visualization" / "feature_summary.txt",
            ],
        ),
        (
            "generate candidate viewpoints",
            [
                py,
                str(ROOT / "scripts" / "generate_candidate_viewpoints.py"),
                "--point-features-csv",
                str(point_features),
                "--output-csv",
                str(candidate_csv),
                "--output-json",
                str(candidate_json),
                "--working-distance",
                str(args.working_distance),
                "--hfov-deg",
                str(args.hfov_deg),
                "--vfov-deg",
                str(args.vfov_deg),
                "--overlap",
                str(args.overlap),
                "--max-views",
                str(args.candidate_views),
            ],
            [candidate_csv, candidate_json],
        ),
        (
            "visualize candidate viewpoints",
            [
                py,
                str(ROOT / "scripts" / "visualize_candidate_viewpoints.py"),
                "--viewpoints-csv",
                str(candidate_csv),
                "--point-features-csv",
                str(point_features),
                "--output-dir",
                str(ROOT / "outputs" / "candidate_viewpoints" / "visualization"),
                "--max-background-points",
                "20000",
                "--max-viewpoints",
                str(args.candidate_views),
            ],
            [
                ROOT / "outputs" / "candidate_viewpoints" / "visualization" / "candidate_viewpoints_3d.png",
                ROOT / "outputs" / "candidate_viewpoints" / "visualization" / "candidate_viewpoints_top_projection.png",
            ],
        ),
        (
            "select greedy efficient path",
            [
                py,
                str(ROOT / "scripts" / "select_viewpoint_path.py"),
                "--viewpoints-csv",
                str(candidate_csv),
                "--point-features-csv",
                str(point_features),
                "--output-csv",
                str(selected_csv),
                "--output-json",
                str(selected_json),
                "--target-coverage",
                "0.85",
                "--max-selected",
                "80",
                "--min-new-coverage",
                "0.0002",
                "--max-surface-points",
                str(args.max_surface_points),
            ],
            [selected_csv, selected_json],
        ),
        (
            "select greedy high-coverage path",
            [
                py,
                str(ROOT / "scripts" / "select_viewpoint_path.py"),
                "--viewpoints-csv",
                str(candidate_csv),
                "--point-features-csv",
                str(point_features),
                "--output-csv",
                str(selected_full_csv),
                "--output-json",
                str(selected_full_json),
                "--target-coverage",
                "0.95",
                "--max-selected",
                "160",
                "--min-new-coverage",
                "0.00005",
                "--max-surface-points",
                str(args.max_surface_points),
                "--two-opt-iterations",
                "20",
            ],
            [selected_full_csv, selected_full_json],
        ),
        (
            "visualize greedy efficient path",
            [
                py,
                str(ROOT / "scripts" / "visualize_selected_viewpath.py"),
                "--selected-csv",
                str(selected_csv),
                "--candidate-csv",
                str(candidate_csv),
                "--point-features-csv",
                str(point_features),
                "--output-dir",
                str(ROOT / "outputs" / "selected_viewpath" / "visualization_80"),
                "--max-background-points",
                "20000",
            ],
            [
                ROOT / "outputs" / "selected_viewpath" / "visualization_80" / "selected_viewpath_3d.png",
                ROOT / "outputs" / "selected_viewpath" / "visualization_80" / "selected_viewpath_top_projection.png",
            ],
        ),
        (
            "visualize greedy high-coverage path",
            [
                py,
                str(ROOT / "scripts" / "visualize_selected_viewpath.py"),
                "--selected-csv",
                str(selected_full_csv),
                "--candidate-csv",
                str(candidate_csv),
                "--point-features-csv",
                str(point_features),
                "--output-dir",
                str(ROOT / "outputs" / "selected_viewpath" / "visualization_full"),
                "--max-background-points",
                "20000",
            ],
            [
                ROOT / "outputs" / "selected_viewpath" / "visualization_full" / "selected_viewpath_3d.png",
                ROOT / "outputs" / "selected_viewpath" / "visualization_full" / "selected_viewpath_top_projection.png",
            ],
        ),
    ]

    if args.run_ga:
        steps.extend(
            [
                (
                    "select genetic algorithm baseline",
                    [
                        py,
                        str(ROOT / "scripts" / "run_genetic_viewpoint_baseline.py"),
                        "--viewpoints-csv",
                        str(candidate_csv),
                        "--point-features-csv",
                        str(point_features),
                        "--output-csv",
                        str(ga_csv),
                        "--output-json",
                        str(ga_json),
                        "--target-coverage",
                        "0.85",
                        "--max-selected",
                        "80",
                        "--min-new-coverage",
                        "0.0002",
                        "--max-surface-points",
                        str(args.ga_max_surface_points),
                        "--two-opt-iterations",
                        "20",
                        "--population-size",
                        str(args.ga_population_size),
                        "--generations",
                        str(args.ga_generations),
                        "--path-weight",
                        str(args.ga_path_weight),
                        "--ga-seed",
                        str(args.ga_seed),
                    ],
                    [ga_csv, ga_json],
                ),
                (
                    "visualize genetic algorithm baseline",
                    [
                        py,
                        str(ROOT / "scripts" / "visualize_selected_viewpath.py"),
                        "--selected-csv",
                        str(ga_csv),
                        "--candidate-csv",
                        str(candidate_csv),
                        "--point-features-csv",
                        str(point_features),
                        "--output-dir",
                        str(ga_visual_dir),
                        "--max-background-points",
                        "20000",
                    ],
                    [
                        ga_visual_dir / "selected_viewpath_3d.png",
                        ga_visual_dir / "selected_viewpath_top_projection.png",
                    ],
                ),
            ]
        )

    if args.run_ilp:
        steps.extend(
            [
                (
                    "select ILP+TSP baseline",
                    [
                        py,
                        str(ROOT / "scripts" / "run_ilp_tsp_viewpoint_baseline.py"),
                        "--viewpoints-csv",
                        str(candidate_csv),
                        "--point-features-csv",
                        str(point_features),
                        "--output-csv",
                        str(ilp_csv),
                        "--output-json",
                        str(ilp_json),
                        "--target-coverage",
                        "0.85",
                        "--max-selected",
                        "80",
                        "--min-new-coverage",
                        "0.0002",
                        "--max-surface-points",
                        str(args.ilp_max_surface_points),
                        "--two-opt-iterations",
                        "20",
                        "--ilp-time-limit",
                        str(args.ilp_time_limit),
                        "--ilp-mip-gap",
                        str(args.ilp_mip_gap),
                    ],
                    [ilp_csv, ilp_json],
                ),
                (
                    "visualize ILP+TSP baseline",
                    [
                        py,
                        str(ROOT / "scripts" / "visualize_selected_viewpath.py"),
                        "--selected-csv",
                        str(ilp_csv),
                        "--candidate-csv",
                        str(candidate_csv),
                        "--point-features-csv",
                        str(point_features),
                        "--output-dir",
                        str(ilp_visual_dir),
                        "--max-background-points",
                        "20000",
                    ],
                    [
                        ilp_visual_dir / "selected_viewpath_3d.png",
                        ilp_visual_dir / "selected_viewpath_top_projection.png",
                    ],
                ),
            ]
        )

    steps.append(
        (
            "summarize experiment results",
            [
                py,
                str(ROOT / "scripts" / "summarize_experiment_results.py"),
                "--summary-dir",
                str(ROOT / "outputs" / "selected_viewpath"),
                "--output-csv",
                str(ROOT / "outputs" / "experiment_summary.csv"),
                "--output-md",
                str(ROOT / "outputs" / "experiment_summary.md"),
            ],
            [ROOT / "outputs" / "experiment_summary.csv", ROOT / "outputs" / "experiment_summary.md"],
        )
    )

    for name, command, outputs in steps:
        rerun_summary = (args.run_ga or args.run_ilp) and name == "summarize experiment results"
        skip_existing = args.skip_existing and not rerun_summary
        run_step(name, command, outputs, skip_existing=skip_existing, dry_run=args.dry_run)

    print("[done] pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

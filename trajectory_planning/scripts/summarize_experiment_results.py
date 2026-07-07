from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


SUMMARY_FIELDS = [
    "name",
    "baseline",
    "selected_view_count",
    "coverage_ratio",
    "weighted_coverage_ratio",
    "path_cost_before_2opt",
    "path_cost_after_2opt",
    "path_cost_reduction",
    "total_reward",
    "terminated",
    "truncated",
    "surface_point_count",
    "input_candidate_count",
    "target_coverage",
    "max_selected",
    "reward_motion_scale",
    "reward_local_jump_threshold",
    "reward_local_jump_scale",
    "reward_smoothness_scale",
    "reward_region_switch_scale",
    "reward_intermediate_scale",
    "reward_terminal_coverage_scale",
    "reward_terminal_path_scale",
    "reward_terminal_shot_scale",
    "reward_expert_prior_scale",
    "reward_expert_next_scale",
    "expert_route_count",
    "expert_route_csv",
    "ga_population_size",
    "ga_generations",
    "ga_path_weight",
    "ilp_time_limit",
    "ilp_mip_gap",
    "ilp_solver_success",
    "ilp_solver_status",
    "summary_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize viewpoint-planning experiment summary JSON files.")
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=ROOT / "outputs" / "selected_viewpath",
        help="Directory containing *_summary.json files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "outputs" / "experiment_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "outputs" / "experiment_summary.md",
        help="Output Markdown table path.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    reward_config = payload.get("reward_config", {})
    genetic_config = payload.get("genetic_config", {})
    ilp_config = payload.get("ilp_config", {})
    ilp_summary = payload.get("ilp_summary", {})
    return {
        "name": path.stem.removesuffix("_summary"),
        "baseline": payload.get("baseline", "unknown"),
        "selected_view_count": payload.get("selected_view_count", ""),
        "coverage_ratio": payload.get("coverage_ratio", ""),
        "weighted_coverage_ratio": payload.get("weighted_coverage_ratio", ""),
        "path_cost_before_2opt": payload.get("path_cost_before_2opt", ""),
        "path_cost_after_2opt": payload.get("path_cost_after_2opt", ""),
        "path_cost_reduction": payload.get("path_cost_reduction", ""),
        "total_reward": payload.get("total_reward", ""),
        "terminated": payload.get("terminated", ""),
        "truncated": payload.get("truncated", ""),
        "surface_point_count": payload.get("surface_point_count", ""),
        "input_candidate_count": payload.get("input_candidate_count", ""),
        "target_coverage": payload.get("target_coverage", ""),
        "max_selected": config.get("max_selected", ""),
        "reward_motion_scale": reward_config.get("motion_scale", ""),
        "reward_local_jump_threshold": reward_config.get("local_jump_threshold", ""),
        "reward_local_jump_scale": reward_config.get("local_jump_scale", ""),
        "reward_smoothness_scale": reward_config.get("smoothness_scale", ""),
        "reward_region_switch_scale": reward_config.get("region_switch_scale", ""),
        "reward_intermediate_scale": reward_config.get("intermediate_reward_scale", ""),
        "reward_terminal_coverage_scale": reward_config.get("terminal_coverage_scale", ""),
        "reward_terminal_path_scale": reward_config.get("terminal_path_scale", ""),
        "reward_terminal_shot_scale": reward_config.get("terminal_shot_scale", ""),
        "reward_expert_prior_scale": reward_config.get("expert_prior_scale", ""),
        "reward_expert_next_scale": reward_config.get("expert_next_scale", ""),
        "expert_route_count": payload.get("expert_route_count", ""),
        "expert_route_csv": payload.get("expert_route_csv", ""),
        "ga_population_size": genetic_config.get("population_size", ""),
        "ga_generations": genetic_config.get("generations", ""),
        "ga_path_weight": genetic_config.get("path_weight", ""),
        "ilp_time_limit": ilp_config.get("time_limit", ""),
        "ilp_mip_gap": ilp_config.get("mip_rel_gap", ""),
        "ilp_solver_success": ilp_summary.get("solver_success", ""),
        "ilp_solver_status": ilp_summary.get("solver_status", ""),
        "summary_json": str(path),
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display_fields = [
        "name",
        "baseline",
        "selected_view_count",
        "coverage_ratio",
        "weighted_coverage_ratio",
        "path_cost_after_2opt",
        "reward_motion_scale",
        "reward_local_jump_threshold",
        "reward_local_jump_scale",
        "reward_smoothness_scale",
        "reward_region_switch_scale",
        "reward_intermediate_scale",
        "reward_terminal_coverage_scale",
        "reward_terminal_path_scale",
        "reward_terminal_shot_scale",
        "reward_expert_prior_scale",
        "reward_expert_next_scale",
        "expert_route_count",
        "ga_generations",
        "ga_path_weight",
        "ilp_time_limit",
        "ilp_mip_gap",
        "ilp_solver_success",
    ]
    lines = [
        "# Experiment Summary",
        "",
        "| " + " | ".join(display_fields) + " |",
        "| " + " | ".join(["---"] * len(display_fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in display_fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.summary_dir.exists():
        print(f"Summary directory not found: {args.summary_dir}", file=sys.stderr)
        return 2

    paths = sorted(args.summary_dir.rglob("*summary.json"))
    rows = [load_summary(path) for path in paths]
    rows.sort(
        key=lambda row: (
            str(row.get("baseline", "")),
            -float(row.get("weighted_coverage_ratio") or 0.0),
            float(row.get("path_cost_after_2opt") or 0.0),
            str(row.get("name", "")),
        )
    )

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"summary_count={len(rows)}")
    print(f"output_csv={args.output_csv}")
    print(f"output_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

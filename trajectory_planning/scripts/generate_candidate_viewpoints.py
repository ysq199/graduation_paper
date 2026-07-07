from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.viewpoint_generation import (  # noqa: E402
    ViewpointConfig,
    generate_candidate_viewpoints,
    load_point_feature_csv,
    write_summary_json,
    write_viewpoints_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate camera/endoscope viewpoints from ROI surface features."
    )
    parser.add_argument("--point-features-csv", type=Path, required=True, help="CSV from extract_roi_features.py.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output candidate viewpoint CSV.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON summary output.")
    parser.add_argument("--working-distance", type=float, default=20.0, help="Standoff distance from surface point.")
    parser.add_argument("--hfov-deg", type=float, default=55.0, help="Horizontal camera/endoscope FOV in degrees.")
    parser.add_argument("--vfov-deg", type=float, default=40.0, help="Vertical camera/endoscope FOV in degrees.")
    parser.add_argument("--overlap", type=float, default=0.35, help="Expected overlap ratio between neighbor views.")
    parser.add_argument("--max-views", type=int, default=120, help="Maximum selected candidate viewpoints.")
    parser.add_argument("--curvature-percentile", type=float, default=75.0, help="Percentile threshold for high-curvature ROI.")
    parser.add_argument("--min-spacing-scale", type=float, default=0.45, help="Spacing scale for high-priority points.")
    parser.add_argument("--max-spacing-scale", type=float, default=1.15, help="Spacing scale for low-priority points.")
    parser.add_argument("--curvature-weight", type=float, default=0.45, help="Priority weight for local curvature.")
    parser.add_argument("--boundary-weight", type=float, default=0.35, help="Priority weight for ROI boundary score.")
    parser.add_argument("--sparsity-weight", type=float, default=0.20, help="Priority weight for local roughness/sparsity.")
    parser.add_argument(
        "--normal-orientation",
        choices=["as_is", "away_from_centroid", "toward_centroid"],
        default="away_from_centroid",
        help="How to resolve PCA normal sign ambiguity.",
    )
    parser.add_argument(
        "--candidate-filter",
        choices=["all", "boundary", "high_curvature", "boundary_or_high_curvature", "boundary_and_high_curvature"],
        default="all",
        help="Optional ROI subset used before greedy viewpoint selection.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["stratified", "priority"],
        default="stratified",
        help="stratified keeps coverage for surface/boundary/curvature regions; priority selects only highest-score views.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, point_features = load_point_feature_csv(args.point_features_csv)
    config = ViewpointConfig(
        working_distance=args.working_distance,
        hfov_deg=args.hfov_deg,
        vfov_deg=args.vfov_deg,
        overlap_ratio=args.overlap,
        max_views=args.max_views,
        curvature_percentile=args.curvature_percentile,
        min_spacing_scale=args.min_spacing_scale,
        max_spacing_scale=args.max_spacing_scale,
        curvature_weight=args.curvature_weight,
        boundary_weight=args.boundary_weight,
        sparsity_weight=args.sparsity_weight,
        normal_orientation=args.normal_orientation,
        candidate_filter=args.candidate_filter,
        sampling_strategy=args.sampling_strategy,
    )
    viewpoints, summary = generate_candidate_viewpoints(point_features, config)
    write_viewpoints_csv(args.output_csv, viewpoints)
    if args.output_json:
        write_summary_json(args.output_json, summary)

    print(f"input_points={summary['input_point_count']}")
    print(f"candidate_points={summary['candidate_mask_count']}")
    print(f"selected_viewpoints={summary['selected_view_count']}")
    print(f"output_csv={args.output_csv}")
    if args.output_json:
        print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

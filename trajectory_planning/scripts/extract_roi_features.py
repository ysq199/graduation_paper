from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.roi_feature_extraction import (  # noqa: E402
    contour_payload,
    estimate_surface_features,
    extract_contour_features,
    load_mask,
    load_point_cloud,
    surface_payload,
    write_json,
    write_points_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 2D ROI contour features and 3D surface features.")
    parser.add_argument("--image-mask", type=Path, help="Path to a grayscale/binary ROI mask image.")
    parser.add_argument("--point-cloud", type=Path, help="Path to an XYZ/TXT/CSV point cloud file.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON summary path.")
    parser.add_argument("--output-contour-csv", type=Path, help="Optional sampled contour CSV output.")
    parser.add_argument("--output-point-features-csv", type=Path, help="Optional per-point surface feature CSV output.")
    parser.add_argument("--threshold", type=int, default=None, help="Optional binary threshold for image masks.")
    parser.add_argument("--contour-samples", type=int, default=256, help="Number of sampled contour points.")
    parser.add_argument("--k-neighbors", type=int, default=24, help="K for point-cloud local PCA.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: dict[str, object] = {"inputs": {}, "results": []}

    if not args.image_mask and not args.point_cloud:
        print("Provide --image-mask, --point-cloud, or both.", file=sys.stderr)
        return 2

    if args.image_mask:
        mask = load_mask(args.image_mask, threshold=args.threshold)
        contour_features, sampled = extract_contour_features(mask, contour_sample_count=args.contour_samples)
        payload["inputs"]["image_mask"] = str(args.image_mask)
        payload["results"].append(contour_payload(contour_features))
        if args.output_contour_csv:
            write_points_csv(args.output_contour_csv, ["x", "y"], sampled)

    if args.point_cloud:
        points = load_point_cloud(args.point_cloud)
        surface_features, per_point = estimate_surface_features(points, k_neighbors=args.k_neighbors)
        payload["inputs"]["point_cloud"] = str(args.point_cloud)
        payload["results"].append(surface_payload(surface_features))
        if args.output_point_features_csv:
            write_points_csv(
                args.output_point_features_csv,
                ["x", "y", "z", "nx", "ny", "nz", "curvature", "roughness", "boundary_score", "boundary_flag"],
                per_point,
            )

    write_json(args.output_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


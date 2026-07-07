from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.viewpoint_visibility import (  # noqa: E402
    VisibilityConfig,
    annotate_viewpoint_visibility,
    load_candidate_csv,
    load_stl_mesh,
    write_candidate_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate candidate viewpoints with simple STL ray-visibility flags.")
    parser.add_argument("--stl", type=Path, required=True, help="Input STL mesh.")
    parser.add_argument("--viewpoints-csv", type=Path, required=True, help="Input candidate viewpoint CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output annotated candidate viewpoint CSV.")
    parser.add_argument("--output-json", type=Path, help="Optional visibility summary JSON.")
    parser.add_argument("--target-clearance-ratio", type=float, default=1e-4, help="Ignore hits very close to the target endpoint.")
    parser.add_argument("--visible-threshold", type=float, default=0.5, help="Visibility ratio threshold below which a viewpoint is flagged occluded.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fieldnames, candidates = load_candidate_csv(args.viewpoints_csv)
    mesh = load_stl_mesh(args.stl)
    config = VisibilityConfig(
        target_clearance_ratio=args.target_clearance_ratio,
        visible_threshold=args.visible_threshold,
    )
    rows, summary = annotate_viewpoint_visibility(candidates, mesh, config)
    write_candidate_csv(args.output_csv, fieldnames, rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"input_candidate_count={summary['input_candidate_count']}")
    print(f"visible_candidate_count={summary['visible_candidate_count']}")
    print(f"occluded_candidate_count={summary['occluded_candidate_count']}")
    print(f"visible_ratio={summary['visible_ratio']:.4f}")
    print(f"output_csv={args.output_csv}")
    if args.output_json:
        print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

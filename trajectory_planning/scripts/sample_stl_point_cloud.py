from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.stl_sampling import load_stl, sample_stl_surface, triangle_areas, write_xyz_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample an STL mesh surface into an XYZ CSV point cloud.")
    parser.add_argument("--stl", type=Path, required=True, help="Input STL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV/XYZ path.")
    parser.add_argument("--points", type=int, default=20000, help="Number of sampled surface points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh = load_stl(args.stl)
    points = sample_stl_surface(mesh, point_count=args.points, seed=args.seed)
    write_xyz_csv(args.output, points)
    print(f"triangles={len(mesh.faces)}")
    print(f"surface_area={triangle_areas(mesh).sum():.6f}")
    print(f"points={len(points)}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

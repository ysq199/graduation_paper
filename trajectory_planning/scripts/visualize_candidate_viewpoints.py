from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.viewpoint_generation import POINT_FEATURE_HEADER  # noqa: E402


REGION_COLORS = {
    "surface": "#4c78a8",
    "high_curvature": "#f58518",
    "boundary": "#e45756",
    "boundary_high_curvature": "#b279a2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize generated candidate camera/endoscope viewpoints.")
    parser.add_argument("--viewpoints-csv", type=Path, required=True, help="CSV from generate_candidate_viewpoints.py.")
    parser.add_argument("--point-features-csv", type=Path, help="Optional surface point features for background point cloud.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated figures.")
    parser.add_argument("--max-background-points", type=int, default=20000, help="Maximum background surface points to draw.")
    parser.add_argument("--max-viewpoints", type=int, default=300, help="Maximum viewpoints to draw.")
    return parser.parse_args()


def read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    return header, np.asarray(rows, dtype=float)


def read_viewpoints(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_downsample(points: np.ndarray, max_points: int, seed: int = 31) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def set_axes_equal(ax) -> None:
    ranges = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = ranges.mean(axis=1)
    radius = 0.5 * np.max(ranges[:, 1] - ranges[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def viewpoints_to_arrays(rows: list[dict[str, str]], max_viewpoints: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if len(rows) > max_viewpoints:
        rows = rows[:max_viewpoints]
    cameras = []
    targets = []
    regions = []
    priorities = []
    for row in rows:
        cameras.append([float(row["camera_x"]), float(row["camera_y"]), float(row["camera_z"])])
        targets.append([float(row["target_x"]), float(row["target_y"]), float(row["target_z"])])
        regions.append(row.get("region_type", "surface"))
        priorities.append(float(row.get("priority_score", 0.0)))
    return np.asarray(cameras, dtype=float), np.asarray(targets, dtype=float), regions, np.asarray(priorities, dtype=float)


def draw_viewpoints_3d(
    surface_points: np.ndarray | None,
    cameras: np.ndarray,
    targets: np.ndarray,
    regions: list[str],
    priorities: np.ndarray,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    if surface_points is not None and len(surface_points):
        ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1.2, c="#9aa0a6", alpha=0.23)

    for region, color in REGION_COLORS.items():
        mask = np.array([item == region for item in regions], dtype=bool)
        if mask.any():
            ax.scatter(
                cameras[mask, 0],
                cameras[mask, 1],
                cameras[mask, 2],
                s=18 + 18 * priorities[mask],
                c=color,
                alpha=0.95,
                label=region,
                edgecolors="none",
            )

    step = max(len(cameras) // 120, 1)
    for camera, target in zip(cameras[::step], targets[::step]):
        ax.plot([camera[0], target[0]], [camera[1], target[1]], [camera[2], target[2]], c="#333333", alpha=0.25, linewidth=0.55)

    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], s=4, c="#1b1b1b", alpha=0.6, label="surface targets")
    ax.view_init(elev=25, azim=-58)
    ax.set_title("Candidate camera/endoscope viewpoints")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best", fontsize=8)
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_viewpoints_projection(
    surface_points: np.ndarray | None,
    cameras: np.ndarray,
    targets: np.ndarray,
    regions: list[str],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    if surface_points is not None and len(surface_points):
        ax.scatter(surface_points[:, 0], surface_points[:, 1], s=1.2, c="#b8bec6", alpha=0.35, linewidths=0)

    for region, color in REGION_COLORS.items():
        mask = np.array([item == region for item in regions], dtype=bool)
        if mask.any():
            ax.scatter(cameras[mask, 0], cameras[mask, 1], s=12, c=color, label=region, alpha=0.9, linewidths=0)

    step = max(len(cameras) // 160, 1)
    for camera, target in zip(cameras[::step], targets[::step]):
        ax.plot([camera[0], target[0]], [camera[1], target[1]], c="#333333", alpha=0.2, linewidth=0.45)

    ax.set_title("Candidate viewpoints top projection")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_viewpoints(args.viewpoints_csv)
    if not rows:
        print("No viewpoints found in CSV.", file=sys.stderr)
        return 2
    cameras, targets, regions, priorities = viewpoints_to_arrays(rows, args.max_viewpoints)

    surface_points = None
    if args.point_features_csv:
        header, point_features = read_numeric_csv(args.point_features_csv)
        if header[: len(POINT_FEATURE_HEADER)] != POINT_FEATURE_HEADER:
            print(f"Unexpected point feature CSV header: {header}", file=sys.stderr)
            return 2
        surface_points = maybe_downsample(point_features[:, :3], args.max_background_points)

    outputs = [
        args.output_dir / "candidate_viewpoints_3d.png",
        args.output_dir / "candidate_viewpoints_top_projection.png",
    ]
    draw_viewpoints_3d(surface_points, cameras, targets, regions, priorities, outputs[0])
    draw_viewpoints_projection(surface_points, cameras, targets, regions, outputs[1])

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize selected and ordered camera/endoscope view path.")
    parser.add_argument("--selected-csv", type=Path, required=True, help="CSV from select_viewpoint_path.py.")
    parser.add_argument("--candidate-csv", type=Path, help="Optional candidate viewpoint CSV for background.")
    parser.add_argument("--point-features-csv", type=Path, help="Optional surface point feature CSV for background.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated figures.")
    parser.add_argument("--max-background-points", type=int, default=20000, help="Maximum surface points to draw.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    return header, np.asarray(rows, dtype=float)


def maybe_downsample(points: np.ndarray, max_points: int, seed: int = 41) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def rows_to_camera_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row["camera_x"]), float(row["camera_y"]), float(row["camera_z"])] for row in rows], dtype=float)


def rows_to_target_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row["target_x"]), float(row["target_y"]), float(row["target_z"])] for row in rows], dtype=float)


def set_axes_equal(ax) -> None:
    ranges = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = ranges.mean(axis=1)
    radius = 0.5 * np.max(ranges[:, 1] - ranges[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def draw_path_3d(
    surface_points: np.ndarray | None,
    candidate_cameras: np.ndarray | None,
    selected_cameras: np.ndarray,
    selected_targets: np.ndarray,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    if surface_points is not None and len(surface_points):
        ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1.0, c="#9aa0a6", alpha=0.20)
    if candidate_cameras is not None and len(candidate_cameras):
        ax.scatter(candidate_cameras[:, 0], candidate_cameras[:, 1], candidate_cameras[:, 2], s=8, c="#bbbbbb", alpha=0.28, label="candidate views")

    ax.plot(
        selected_cameras[:, 0],
        selected_cameras[:, 1],
        selected_cameras[:, 2],
        c="#d62728",
        linewidth=1.4,
        alpha=0.95,
        label="selected route",
    )
    ax.scatter(selected_cameras[:, 0], selected_cameras[:, 1], selected_cameras[:, 2], s=20, c="#d62728", alpha=0.95)
    ax.scatter(selected_cameras[:1, 0], selected_cameras[:1, 1], selected_cameras[:1, 2], s=58, c="#2ca02c", label="start")
    ax.scatter(selected_cameras[-1:, 0], selected_cameras[-1:, 1], selected_cameras[-1:, 2], s=58, c="#1f77b4", label="end")

    step = max(len(selected_cameras) // 120, 1)
    for camera, target in zip(selected_cameras[::step], selected_targets[::step]):
        ax.plot([camera[0], target[0]], [camera[1], target[1]], [camera[2], target[2]], c="#333333", alpha=0.18, linewidth=0.45)

    ax.view_init(elev=25, azim=-58)
    ax.set_title("Selected viewpoint path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best", fontsize=8)
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_path_projection(
    surface_points: np.ndarray | None,
    candidate_cameras: np.ndarray | None,
    selected_cameras: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    if surface_points is not None and len(surface_points):
        ax.scatter(surface_points[:, 0], surface_points[:, 1], s=1.0, c="#b8bec6", alpha=0.30, linewidths=0)
    if candidate_cameras is not None and len(candidate_cameras):
        ax.scatter(candidate_cameras[:, 0], candidate_cameras[:, 1], s=8, c="#bbbbbb", alpha=0.35, label="candidate views", linewidths=0)
    ax.plot(selected_cameras[:, 0], selected_cameras[:, 1], c="#d62728", linewidth=1.3, label="selected route")
    ax.scatter(selected_cameras[:, 0], selected_cameras[:, 1], s=18, c="#d62728", alpha=0.95, linewidths=0)
    ax.scatter(selected_cameras[:1, 0], selected_cameras[:1, 1], s=60, c="#2ca02c", label="start", linewidths=0)
    ax.scatter(selected_cameras[-1:, 0], selected_cameras[-1:, 1], s=60, c="#1f77b4", label="end", linewidths=0)
    ax.set_title("Selected viewpoint path top projection")
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

    selected_rows = read_rows(args.selected_csv)
    if not selected_rows:
        print("No selected viewpoints found.", file=sys.stderr)
        return 2
    selected_cameras = rows_to_camera_array(selected_rows)
    selected_targets = rows_to_target_array(selected_rows)

    candidate_cameras = None
    if args.candidate_csv:
        candidate_rows = read_rows(args.candidate_csv)
        candidate_cameras = rows_to_camera_array(candidate_rows)

    surface_points = None
    if args.point_features_csv:
        header, point_features = read_numeric_csv(args.point_features_csv)
        if header[: len(POINT_FEATURE_HEADER)] != POINT_FEATURE_HEADER:
            print(f"Unexpected point feature CSV header: {header}", file=sys.stderr)
            return 2
        surface_points = maybe_downsample(point_features[:, :3], args.max_background_points)

    outputs = [
        args.output_dir / "selected_viewpath_3d.png",
        args.output_dir / "selected_viewpath_top_projection.png",
    ]
    draw_path_3d(surface_points, candidate_cameras, selected_cameras, selected_targets, outputs[0])
    draw_path_projection(surface_points, candidate_cameras, selected_cameras, outputs[1])

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

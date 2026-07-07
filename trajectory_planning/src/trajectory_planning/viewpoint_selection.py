from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from trajectory_planning.viewpoint_generation import POINT_FEATURE_HEADER


SELECTED_VIEWPOINT_HEADER = [
    "route_order",
    "candidate_view_id",
    "camera_x",
    "camera_y",
    "camera_z",
    "target_x",
    "target_y",
    "target_z",
    "view_dir_x",
    "view_dir_y",
    "view_dir_z",
    "qw",
    "qx",
    "qy",
    "qz",
    "region_type",
    "priority_score",
    "newly_covered_points",
    "cumulative_coverage_ratio",
    "transition_distance",
]


@dataclass
class SelectionConfig:
    target_coverage: float = 0.85
    max_selected: int = 80
    min_new_coverage: float = 0.002
    motion_weight: float = 0.12
    orientation_weight: float = 2.0
    curvature_weight: float = 1.0
    boundary_weight: float = 1.0
    roughness_weight: float = 0.20
    max_surface_points: int = 50000
    two_opt_iterations: int = 50
    seed: int = 17


def load_viewpoints_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No candidate viewpoints found in {path}")

    numeric_fields = [
        "view_id",
        "camera_x",
        "camera_y",
        "camera_z",
        "target_x",
        "target_y",
        "target_z",
        "view_dir_x",
        "view_dir_y",
        "view_dir_z",
        "qw",
        "qx",
        "qy",
        "qz",
        "effective_radius",
        "priority_score",
    ]
    for row in rows:
        for field in numeric_fields:
            if field in row and row[field] != "":
                row[field] = float(row[field])
    return rows


def load_point_feature_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    if header[: len(POINT_FEATURE_HEADER)] != POINT_FEATURE_HEADER:
        raise ValueError(f"Unexpected point feature CSV header: {header}")
    return header, np.asarray(rows, dtype=float)[:, : len(POINT_FEATURE_HEADER)]


def select_viewpoints(
    candidates: list[dict[str, object]],
    point_features: np.ndarray,
    config: SelectionConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    points, point_weights = prepare_surface_points(point_features, config)
    cover_sets = build_coverage_sets(candidates, points)
    selected_order, selection_trace = greedy_weighted_set_cover(candidates, cover_sets, point_weights, config)

    path_before = route_cost(selected_order, candidates, config)
    optimized_order = two_opt_route(selected_order, candidates, config)
    path_after = route_cost(optimized_order, candidates, config)

    selected_rows, final_coverage = build_selected_rows(optimized_order, candidates, cover_sets, len(points))
    summary = {
        "input_candidate_count": int(len(candidates)),
        "surface_point_count": int(len(points)),
        "selected_view_count": int(len(selected_rows)),
        "coverage_ratio": float(final_coverage),
        "weighted_coverage_ratio": float(selection_trace["weighted_coverage_ratio"]),
        "target_coverage": float(config.target_coverage),
        "path_cost_before_2opt": float(path_before),
        "path_cost_after_2opt": float(path_after),
        "path_cost_reduction": float(path_before - path_after),
        "region_counts": count_regions(selected_rows),
        "config": asdict(config),
    }
    return selected_rows, summary


def prepare_surface_points(point_features: np.ndarray, config: SelectionConfig) -> tuple[np.ndarray, np.ndarray]:
    if len(point_features) > config.max_surface_points:
        rng = np.random.default_rng(config.seed)
        idx = rng.choice(len(point_features), size=config.max_surface_points, replace=False)
        point_features = point_features[idx]

    points = point_features[:, 0:3]
    curvature = normalize01(point_features[:, 6])
    roughness = normalize01(point_features[:, 7])
    boundary = point_features[:, 9] > 0.5
    weights = (
        1.0
        + config.curvature_weight * curvature
        + config.boundary_weight * boundary.astype(float)
        + config.roughness_weight * roughness
    )
    return points, weights.astype(float)


def build_coverage_sets(candidates: list[dict[str, object]], points: np.ndarray) -> list[np.ndarray]:
    targets = np.asarray([[row["target_x"], row["target_y"], row["target_z"]] for row in candidates], dtype=float)
    radii = np.asarray([row.get("effective_radius", 0.0) for row in candidates], dtype=float)
    max_radius = max(float(radii.max()), 1e-9)

    nn = NearestNeighbors(radius=max_radius)
    nn.fit(points)
    distances, indices = nn.radius_neighbors(targets, radius=max_radius, return_distance=True)

    cover_sets: list[np.ndarray] = []
    for idx, dist in zip(indices, distances):
        radius = max(float(radii[len(cover_sets)]), 1e-9)
        cover_sets.append(np.asarray(idx, dtype=int)[np.asarray(dist) <= radius])
    return cover_sets


def greedy_weighted_set_cover(
    candidates: list[dict[str, object]],
    cover_sets: list[np.ndarray],
    point_weights: np.ndarray,
    config: SelectionConfig,
) -> tuple[list[int], dict[str, float]]:
    total_weight = max(float(point_weights.sum()), 1e-12)
    bbox_diag = camera_bbox_diag(candidates)
    covered = np.zeros(len(point_weights), dtype=bool)
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < config.max_selected:
        best_idx = None
        best_score = -float("inf")
        best_gain = 0.0

        for idx in list(remaining):
            cover_idx = cover_sets[idx]
            if len(cover_idx) == 0:
                continue
            new_mask = ~covered[cover_idx]
            gain = float(point_weights[cover_idx[new_mask]].sum()) / total_weight
            if gain <= 0.0:
                continue

            motion_penalty = 0.0
            if selected:
                motion_penalty = transition_cost(candidates[selected[-1]], candidates[idx], config) / bbox_diag
            score = gain - config.motion_weight * motion_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
                best_gain = gain

        if best_idx is None or best_gain < config.min_new_coverage:
            break

        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
        covered[cover_sets[best_idx]] = True

        weighted_coverage = float(point_weights[covered].sum()) / total_weight
        if weighted_coverage >= config.target_coverage:
            break

    weighted_coverage = float(point_weights[covered].sum()) / total_weight
    return selected, {"weighted_coverage_ratio": weighted_coverage}


def two_opt_route(order: list[int], candidates: list[dict[str, object]], config: SelectionConfig) -> list[int]:
    if len(order) < 4 or config.two_opt_iterations <= 0:
        return order[:]

    best = order[:]
    best_cost = route_cost(best, candidates, config)
    for _ in range(config.two_opt_iterations):
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cost = route_cost(candidate, candidates, config)
                if cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = cost
                    improved = True
        if not improved:
            break
    return best


def build_selected_rows(
    order: list[int],
    candidates: list[dict[str, object]],
    cover_sets: list[np.ndarray],
    surface_point_count: int,
) -> tuple[list[dict[str, object]], float]:
    covered = np.zeros(surface_point_count, dtype=bool)
    rows: list[dict[str, object]] = []
    previous = None

    for route_order, candidate_idx in enumerate(order):
        candidate = candidates[candidate_idx]
        cover_idx = cover_sets[candidate_idx]
        new_cover = int((~covered[cover_idx]).sum())
        covered[cover_idx] = True
        transition = 0.0 if previous is None else euclidean_camera_distance(previous, candidate)
        previous = candidate
        rows.append(
            {
                "route_order": int(route_order),
                "candidate_view_id": int(candidate["view_id"]),
                "camera_x": float(candidate["camera_x"]),
                "camera_y": float(candidate["camera_y"]),
                "camera_z": float(candidate["camera_z"]),
                "target_x": float(candidate["target_x"]),
                "target_y": float(candidate["target_y"]),
                "target_z": float(candidate["target_z"]),
                "view_dir_x": float(candidate["view_dir_x"]),
                "view_dir_y": float(candidate["view_dir_y"]),
                "view_dir_z": float(candidate["view_dir_z"]),
                "qw": float(candidate["qw"]),
                "qx": float(candidate["qx"]),
                "qy": float(candidate["qy"]),
                "qz": float(candidate["qz"]),
                "region_type": str(candidate.get("region_type", "surface")),
                "priority_score": float(candidate.get("priority_score", 0.0)),
                "newly_covered_points": new_cover,
                "cumulative_coverage_ratio": float(covered.mean()),
                "transition_distance": float(transition),
            }
        )
    return rows, float(covered.mean())


def route_cost(order: list[int], candidates: list[dict[str, object]], config: SelectionConfig) -> float:
    if len(order) < 2:
        return 0.0
    return float(
        sum(
            transition_cost(candidates[a], candidates[b], config)
            for a, b in zip(order[:-1], order[1:])
        )
    )


def transition_cost(a: dict[str, object], b: dict[str, object], config: SelectionConfig) -> float:
    distance = euclidean_camera_distance(a, b)
    direction_a = np.array([a["view_dir_x"], a["view_dir_y"], a["view_dir_z"]], dtype=float)
    direction_b = np.array([b["view_dir_x"], b["view_dir_y"], b["view_dir_z"]], dtype=float)
    dot = float(np.dot(normalize(direction_a), normalize(direction_b)))
    angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
    return float(distance + config.orientation_weight * angle)


def euclidean_camera_distance(a: dict[str, object], b: dict[str, object]) -> float:
    pa = np.array([a["camera_x"], a["camera_y"], a["camera_z"]], dtype=float)
    pb = np.array([b["camera_x"], b["camera_y"], b["camera_z"]], dtype=float)
    return float(np.linalg.norm(pa - pb))


def camera_bbox_diag(candidates: list[dict[str, object]]) -> float:
    cameras = np.asarray([[row["camera_x"], row["camera_y"], row["camera_z"]] for row in candidates], dtype=float)
    diag = float(np.linalg.norm(cameras.max(axis=0) - cameras.min(axis=0)))
    return max(diag, 1e-9)


def normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        return values
    return values / norm


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin <= 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def count_regions(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("region_type", "surface"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_selected_viewpoints_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_VIEWPOINT_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SELECTED_VIEWPOINT_HEADER})


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

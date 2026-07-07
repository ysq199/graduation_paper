from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


POINT_FEATURE_HEADER = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "curvature",
    "roughness",
    "boundary_score",
    "boundary_flag",
]


VIEWPOINT_HEADER = [
    "view_id",
    "surface_x",
    "surface_y",
    "surface_z",
    "camera_x",
    "camera_y",
    "camera_z",
    "target_x",
    "target_y",
    "target_z",
    "view_dir_x",
    "view_dir_y",
    "view_dir_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "up_x",
    "up_y",
    "up_z",
    "qw",
    "qx",
    "qy",
    "qz",
    "footprint_width",
    "footprint_height",
    "effective_radius",
    "curvature",
    "roughness",
    "boundary_score",
    "boundary_flag",
    "priority_score",
    "spacing_radius",
    "region_type",
]


@dataclass
class ViewpointConfig:
    working_distance: float = 20.0
    hfov_deg: float = 55.0
    vfov_deg: float = 40.0
    overlap_ratio: float = 0.35
    max_views: int = 120
    curvature_percentile: float = 75.0
    min_spacing_scale: float = 0.45
    max_spacing_scale: float = 1.15
    curvature_weight: float = 0.45
    boundary_weight: float = 0.35
    sparsity_weight: float = 0.20
    normal_orientation: str = "away_from_centroid"
    candidate_filter: str = "all"
    sampling_strategy: str = "stratified"


def load_point_feature_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    data = np.asarray(rows, dtype=float)
    if header[: len(POINT_FEATURE_HEADER)] != POINT_FEATURE_HEADER:
        raise ValueError(f"Unexpected point feature CSV header: {header}")
    if data.ndim != 2 or data.shape[1] < len(POINT_FEATURE_HEADER):
        raise ValueError("point feature CSV must contain at least 10 numeric columns")
    return header, data[:, : len(POINT_FEATURE_HEADER)]


def fov_footprint(working_distance: float, hfov_deg: float, vfov_deg: float, overlap_ratio: float) -> tuple[float, float, float]:
    if working_distance <= 0:
        raise ValueError("working_distance must be positive")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")
    if hfov_deg <= 0 or vfov_deg <= 0:
        raise ValueError("FOV angles must be positive")

    width = 2.0 * working_distance * math.tan(math.radians(hfov_deg) * 0.5)
    height = 2.0 * working_distance * math.tan(math.radians(vfov_deg) * 0.5)
    effective_radius = 0.5 * min(width, height) * (1.0 - overlap_ratio)
    return float(width), float(height), float(effective_radius)


def generate_candidate_viewpoints(point_features: np.ndarray, config: ViewpointConfig) -> tuple[list[dict[str, object]], dict[str, object]]:
    points = point_features[:, 0:3]
    normals = point_features[:, 3:6]
    curvature = point_features[:, 6]
    roughness = point_features[:, 7]
    boundary_score = point_features[:, 8]
    boundary_flag = point_features[:, 9] > 0.5

    normals = orient_normals(points, normals, config.normal_orientation)
    footprint_width, footprint_height, effective_radius = fov_footprint(
        config.working_distance,
        config.hfov_deg,
        config.vfov_deg,
        config.overlap_ratio,
    )

    curvature_norm = normalize01(curvature)
    boundary_norm = np.maximum(normalize01(boundary_score), boundary_flag.astype(float))
    sparsity_norm = normalize01(roughness)
    weight_sum = max(config.curvature_weight + config.boundary_weight + config.sparsity_weight, 1e-12)
    priority = (
        config.curvature_weight * curvature_norm
        + config.boundary_weight * boundary_norm
        + config.sparsity_weight * sparsity_norm
    ) / weight_sum
    priority = normalize01(priority)

    high_curvature = curvature >= np.percentile(curvature, config.curvature_percentile)
    valid_mask = build_candidate_mask(config.candidate_filter, boundary_flag, high_curvature)
    if not valid_mask.any():
        valid_mask = np.ones(len(points), dtype=bool)

    local_spacing = effective_radius * (
        config.max_spacing_scale - (config.max_spacing_scale - config.min_spacing_scale) * priority
    )
    labels = classify_regions(boundary_flag, high_curvature)
    if config.sampling_strategy == "priority":
        selected = greedy_select(points, priority, local_spacing, valid_mask, config.max_views)
    elif config.sampling_strategy == "stratified":
        selected = stratified_greedy_select(points, priority, local_spacing, valid_mask, labels, config.max_views)
    else:
        raise ValueError(f"Unsupported sampling_strategy: {config.sampling_strategy}")

    viewpoints: list[dict[str, object]] = []
    for view_id, idx in enumerate(selected):
        surface = points[idx]
        normal = safe_normalize(normals[idx])
        camera = surface + config.working_distance * normal
        view_dir = safe_normalize(surface - camera)
        up, quat = camera_up_and_quaternion(view_dir)
        region_type = str(labels[idx])
        viewpoints.append(
            {
                "view_id": int(view_id),
                "surface_x": float(surface[0]),
                "surface_y": float(surface[1]),
                "surface_z": float(surface[2]),
                "camera_x": float(camera[0]),
                "camera_y": float(camera[1]),
                "camera_z": float(camera[2]),
                "target_x": float(surface[0]),
                "target_y": float(surface[1]),
                "target_z": float(surface[2]),
                "view_dir_x": float(view_dir[0]),
                "view_dir_y": float(view_dir[1]),
                "view_dir_z": float(view_dir[2]),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "up_x": float(up[0]),
                "up_y": float(up[1]),
                "up_z": float(up[2]),
                "qw": float(quat[0]),
                "qx": float(quat[1]),
                "qy": float(quat[2]),
                "qz": float(quat[3]),
                "footprint_width": float(footprint_width),
                "footprint_height": float(footprint_height),
                "effective_radius": float(effective_radius),
                "curvature": float(curvature[idx]),
                "roughness": float(roughness[idx]),
                "boundary_score": float(boundary_score[idx]),
                "boundary_flag": int(boundary_flag[idx]),
                "priority_score": float(priority[idx]),
                "spacing_radius": float(local_spacing[idx]),
                "region_type": region_type,
            }
        )

    summary = {
        "input_point_count": int(len(points)),
        "candidate_mask_count": int(valid_mask.sum()),
        "selected_view_count": int(len(viewpoints)),
        "region_counts": count_regions(viewpoints),
        "fov": {
            "footprint_width": float(footprint_width),
            "footprint_height": float(footprint_height),
            "effective_radius": float(effective_radius),
        },
        "config": asdict(config),
    }
    return viewpoints, summary


def build_candidate_mask(candidate_filter: str, boundary_flag: np.ndarray, high_curvature: np.ndarray) -> np.ndarray:
    if candidate_filter == "all":
        return np.ones(len(boundary_flag), dtype=bool)
    if candidate_filter == "boundary":
        return boundary_flag.copy()
    if candidate_filter == "high_curvature":
        return high_curvature.copy()
    if candidate_filter == "boundary_or_high_curvature":
        return boundary_flag | high_curvature
    if candidate_filter == "boundary_and_high_curvature":
        return boundary_flag & high_curvature
    raise ValueError(f"Unsupported candidate_filter: {candidate_filter}")


def orient_normals(points: np.ndarray, normals: np.ndarray, mode: str) -> np.ndarray:
    oriented = np.asarray(normals, dtype=float).copy()
    if mode == "as_is":
        return normalize_rows(oriented)

    centroid = points.mean(axis=0)
    radial = points - centroid
    dots = np.sum(oriented * radial, axis=1)
    if mode == "away_from_centroid":
        oriented[dots < 0] *= -1.0
    elif mode == "toward_centroid":
        oriented[dots > 0] *= -1.0
    else:
        raise ValueError(f"Unsupported normal_orientation: {mode}")
    return normalize_rows(oriented)


def greedy_select(points: np.ndarray, priority: np.ndarray, spacing: np.ndarray, valid_mask: np.ndarray, max_views: int) -> list[int]:
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0 or max_views <= 0:
        return []

    order = valid_indices[np.argsort(priority[valid_indices])[::-1]]
    selected: list[int] = []
    selected_points: list[np.ndarray] = []
    selected_spacing: list[float] = []

    for idx in order:
        if len(selected) >= max_views:
            break
        if not selected:
            selected.append(int(idx))
            selected_points.append(points[idx])
            selected_spacing.append(float(spacing[idx]))
            continue

        selected_array = np.vstack(selected_points)
        distances = np.linalg.norm(selected_array - points[idx], axis=1)
        allowed = np.minimum(np.asarray(selected_spacing), spacing[idx])
        if np.all(distances >= allowed):
            selected.append(int(idx))
            selected_points.append(points[idx])
            selected_spacing.append(float(spacing[idx]))

    return selected


def stratified_greedy_select(
    points: np.ndarray,
    priority: np.ndarray,
    spacing: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    max_views: int,
) -> list[int]:
    region_quotas = {
        "boundary_high_curvature": 0.30,
        "boundary": 0.20,
        "high_curvature": 0.20,
        "surface": 0.30,
    }
    selected: list[int] = []
    selected_points: list[np.ndarray] = []
    selected_spacing: list[float] = []

    for region, ratio in region_quotas.items():
        region_mask = valid_mask & (labels == region)
        if not region_mask.any():
            continue
        quota = max(1, int(round(max_views * ratio)))
        _append_greedy(
            points,
            priority,
            spacing,
            region_mask,
            quota,
            selected,
            selected_points,
            selected_spacing,
        )

    if len(selected) < max_views:
        _append_greedy(
            points,
            priority,
            spacing,
            valid_mask,
            max_views - len(selected),
            selected,
            selected_points,
            selected_spacing,
        )

    return selected[:max_views]


def _append_greedy(
    points: np.ndarray,
    priority: np.ndarray,
    spacing: np.ndarray,
    valid_mask: np.ndarray,
    quota: int,
    selected: list[int],
    selected_points: list[np.ndarray],
    selected_spacing: list[float],
) -> None:
    if quota <= 0:
        return
    selected_set = set(selected)
    valid_indices = np.flatnonzero(valid_mask)
    order = valid_indices[np.argsort(priority[valid_indices])[::-1]]
    added = 0

    for idx in order:
        if added >= quota:
            break
        if int(idx) in selected_set:
            continue
        if _can_add_point(points[idx], spacing[idx], selected_points, selected_spacing):
            selected.append(int(idx))
            selected_points.append(points[idx])
            selected_spacing.append(float(spacing[idx]))
            selected_set.add(int(idx))
            added += 1


def _can_add_point(
    point: np.ndarray,
    point_spacing: float,
    selected_points: list[np.ndarray],
    selected_spacing: list[float],
) -> bool:
    if not selected_points:
        return True
    selected_array = np.vstack(selected_points)
    distances = np.linalg.norm(selected_array - point, axis=1)
    allowed = np.minimum(np.asarray(selected_spacing), point_spacing)
    return bool(np.all(distances >= allowed))


def camera_up_and_quaternion(view_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = safe_normalize(view_dir)
    preferred_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, preferred_up))) > 0.92:
        preferred_up = np.array([0.0, 1.0, 0.0], dtype=float)

    x_axis = safe_normalize(np.cross(preferred_up, z_axis))
    y_axis = safe_normalize(np.cross(z_axis, x_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return y_axis, rotation_matrix_to_quaternion(rotation)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    m = rotation
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=float)
    return safe_normalize(quat)


def classify_region(boundary: bool, high_curvature: bool) -> str:
    if boundary and high_curvature:
        return "boundary_high_curvature"
    if boundary:
        return "boundary"
    if high_curvature:
        return "high_curvature"
    return "surface"


def classify_regions(boundary_flag: np.ndarray, high_curvature: np.ndarray) -> np.ndarray:
    labels = np.full(len(boundary_flag), "surface", dtype=object)
    labels[high_curvature] = "high_curvature"
    labels[boundary_flag] = "boundary"
    labels[boundary_flag & high_curvature] = "boundary_high_curvature"
    return labels


def count_regions(viewpoints: Iterable[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for viewpoint in viewpoints:
        key = str(viewpoint["region_type"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin <= 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def safe_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        fallback = np.zeros_like(vector, dtype=float)
        fallback[0] = 1.0
        return fallback
    return vector / norm


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return values / norms


def write_viewpoints_csv(path: Path, viewpoints: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIEWPOINT_HEADER)
        writer.writeheader()
        for viewpoint in viewpoints:
            writer.writerow({key: viewpoint.get(key, "") for key in VIEWPOINT_HEADER})


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

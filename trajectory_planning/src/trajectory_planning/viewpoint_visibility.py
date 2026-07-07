from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_planning.stl_sampling import StlMesh, load_stl


@dataclass
class VisibilityConfig:
    target_clearance_ratio: float = 1e-4
    ray_epsilon: float = 1e-9
    visible_threshold: float = 0.5


def annotate_viewpoint_visibility(
    candidates: list[dict[str, object]],
    mesh: StlMesh,
    config: VisibilityConfig | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = config or VisibilityConfig()
    triangles = mesh.vertices[mesh.faces]
    annotated: list[dict[str, object]] = []
    visible_count = 0

    for candidate in candidates:
        camera = np.asarray([candidate["camera_x"], candidate["camera_y"], candidate["camera_z"]], dtype=float)
        target = np.asarray([candidate["target_x"], candidate["target_y"], candidate["target_z"]], dtype=float)
        visibility = target_visibility(camera, target, triangles, config)
        row = dict(candidate)
        row["visibility_ratio"] = float(visibility)
        row["occlusion_flag"] = int(visibility < config.visible_threshold)
        annotated.append(row)
        visible_count += int(visibility >= config.visible_threshold)

    summary = {
        "input_candidate_count": int(len(candidates)),
        "visible_candidate_count": int(visible_count),
        "occluded_candidate_count": int(len(candidates) - visible_count),
        "visible_ratio": float(visible_count / max(len(candidates), 1)),
        "config": asdict(config),
    }
    return annotated, summary


def target_visibility(
    camera: np.ndarray,
    target: np.ndarray,
    triangles: np.ndarray,
    config: VisibilityConfig,
) -> float:
    ray = target - camera
    ray_length = float(np.linalg.norm(ray))
    if ray_length <= config.ray_epsilon:
        return 0.0
    direction = ray / ray_length
    hit_distance = first_intersection_distance(camera, direction, triangles, config.ray_epsilon)
    if hit_distance is None:
        return 1.0
    clearance = max(ray_length * config.target_clearance_ratio, config.ray_epsilon)
    return 0.0 if hit_distance < ray_length - clearance else 1.0


def first_intersection_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    epsilon: float,
) -> float | None:
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(a) > epsilon
    if not np.any(valid):
        return None

    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = f * np.einsum("j,ij->i", direction, q)
    t = f * np.einsum("ij,ij->i", edge2, q)
    hit_mask = valid & (u >= -epsilon) & (v >= -epsilon) & (u + v <= 1.0 + epsilon) & (t > epsilon)
    if not np.any(hit_mask):
        return None
    return float(np.min(t[hit_mask]))


def load_candidate_csv(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    numeric_fields = {
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
    }
    for row in rows:
        for field in numeric_fields:
            if field in row and row[field] != "":
                row[field] = float(row[field])
    return fieldnames, rows


def write_candidate_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    output_fields = fieldnames[:]
    for field in ["visibility_ratio", "occlusion_flag"]:
        if field not in output_fields:
            output_fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in output_fields})


def load_stl_mesh(path: Path) -> StlMesh:
    return load_stl(path)

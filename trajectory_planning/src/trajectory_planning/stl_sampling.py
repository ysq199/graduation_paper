from __future__ import annotations

import csv
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StlMesh:
    vertices: np.ndarray
    faces: np.ndarray


def load_stl(path: Path) -> StlMesh:
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        return _load_binary_stl(data)
    return _load_ascii_stl(data.decode("utf-8", errors="ignore"))


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    return expected == len(data)


def _load_binary_stl(data: bytes) -> StlMesh:
    tri_count = struct.unpack_from("<I", data, 80)[0]
    triangles = np.zeros((tri_count, 3, 3), dtype=float)
    offset = 84
    for i in range(tri_count):
        # normal: 12 bytes, vertices: 36 bytes, attribute: 2 bytes
        values = struct.unpack_from("<12fH", data, offset)
        triangles[i] = np.asarray(values[3:12], dtype=float).reshape(3, 3)
        offset += 50
    return _triangles_to_mesh(triangles)


def _load_ascii_stl(text: str) -> StlMesh:
    vertices: list[list[float]] = []
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) % 3 != 0:
        raise ValueError("ASCII STL vertex count is not divisible by 3")
    triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
    return _triangles_to_mesh(triangles)


def _triangles_to_mesh(triangles: np.ndarray) -> StlMesh:
    if len(triangles) == 0:
        raise ValueError("STL mesh has no triangles")
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=int).reshape(-1, 3)
    return StlMesh(vertices=vertices, faces=faces)


def triangle_areas(mesh: StlMesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def sample_stl_surface(mesh: StlMesh, point_count: int, seed: int = 42) -> np.ndarray:
    if point_count <= 0:
        raise ValueError("point_count must be positive")

    areas = triangle_areas(mesh)
    total_area = float(areas.sum())
    if total_area <= 1e-12:
        raise ValueError("STL mesh has zero surface area")

    rng = np.random.default_rng(seed)
    probabilities = areas / total_area
    face_ids = rng.choice(len(mesh.faces), size=point_count, p=probabilities)
    tri = mesh.vertices[mesh.faces[face_ids]]

    r1 = rng.random(point_count)
    r2 = rng.random(point_count)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    return tri[:, 0] * w0[:, None] + tri[:, 1] * w1[:, None] + tri[:, 2] * w2[:, None]


def write_xyz_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(points.tolist())


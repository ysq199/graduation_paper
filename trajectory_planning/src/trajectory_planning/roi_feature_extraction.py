from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors

try:
    from skimage import measure
except Exception:  # pragma: no cover - optional dependency fallback
    measure = None


@dataclass
class ContourFeatures:
    area_px: int
    perimeter_px: float
    centroid_xy: list[float]
    bbox_xyxy: list[int]
    compactness: float
    aspect_ratio: float
    orientation_deg: float
    major_axis_px: float
    minor_axis_px: float
    mean_curvature: float
    max_curvature: float
    contour_points: int


@dataclass
class SurfaceFeatures:
    point_count: int
    bbox_min_xyz: list[float]
    bbox_max_xyz: list[float]
    centroid_xyz: list[float]
    principal_axes: list[list[float]]
    axis_lengths: list[float]
    mean_curvature: float
    max_curvature: float
    mean_roughness: float
    boundary_candidate_count: int
    k_neighbors: int


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 0
    best_variance = -1.0

    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def load_mask(path: Path, threshold: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    gray = np.asarray(image)
    if threshold is None:
        threshold = _otsu_threshold(gray)
    mask = gray > threshold
    return clean_mask(mask)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    cleaned = ndimage.binary_opening(mask, structure=structure)
    cleaned = ndimage.binary_closing(cleaned, structure=structure)
    cleaned = ndimage.binary_fill_holes(cleaned)

    labels, count = ndimage.label(cleaned)
    if count == 0:
        return cleaned.astype(bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labels == largest


def ordered_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty((0, 2), dtype=float)

    if measure is not None:
        contours = measure.find_contours(mask.astype(float), level=0.5)
        if contours:
            contour_rc = max(contours, key=len)
            return np.column_stack([contour_rc[:, 1], contour_rc[:, 0]]).astype(float)

    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary_rc = np.argwhere(mask & ~eroded)
    if len(boundary_rc) == 0:
        return np.empty((0, 2), dtype=float)

    # Approximate a stable contour order by polar angle around the ROI centroid.
    # This is sufficient for compact blade-region masks and keeps dependencies light.
    centroid_rc = boundary_rc.mean(axis=0)
    dy = boundary_rc[:, 0] - centroid_rc[0]
    dx = boundary_rc[:, 1] - centroid_rc[1]
    order = np.argsort(np.arctan2(dy, dx))
    boundary_rc = boundary_rc[order]
    return np.column_stack([boundary_rc[:, 1], boundary_rc[:, 0]]).astype(float)


def _polyline_perimeter(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    closed = np.vstack([points_xy, points_xy[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def _curvature_stats(points_xy: np.ndarray) -> tuple[float, float]:
    if len(points_xy) < 5:
        return 0.0, 0.0
    prev_pts = np.roll(points_xy, 1, axis=0)
    next_pts = np.roll(points_xy, -1, axis=0)
    v1 = points_xy - prev_pts
    v2 = next_pts - points_xy
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-9) & (n2 > 1e-9)
    cosang = np.zeros(len(points_xy), dtype=float)
    cosang[valid] = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid] * n2[valid])
    angles = np.arccos(np.clip(cosang[valid], -1.0, 1.0))
    curvature = angles / np.maximum((n1[valid] + n2[valid]) * 0.5, 1e-9)
    if len(curvature) == 0:
        return 0.0, 0.0
    return float(curvature.mean()), float(curvature.max())


def sample_polyline(points_xy: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or len(points_xy) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points_xy) == 1:
        return np.repeat(points_xy, count, axis=0)

    closed = np.vstack([points_xy, points_xy[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(seg_lengths)]
    total = cumulative[-1]
    if total <= 1e-9:
        return np.repeat(points_xy[:1], count, axis=0)

    targets = np.linspace(0.0, total, count, endpoint=False)
    sampled = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = min(idx, len(seg_lengths) - 1)
        local = (target - cumulative[idx]) / max(seg_lengths[idx], 1e-9)
        sampled.append(closed[idx] * (1.0 - local) + closed[idx + 1] * local)
    return np.asarray(sampled)


def extract_contour_features(mask: np.ndarray, contour_sample_count: int = 256) -> tuple[ContourFeatures, np.ndarray]:
    coords_rc = np.argwhere(mask)
    if len(coords_rc) == 0:
        raise ValueError("ROI mask has no foreground pixels")

    coords_xy = np.column_stack([coords_rc[:, 1], coords_rc[:, 0]]).astype(float)
    centroid = coords_xy.mean(axis=0)
    xmin, ymin = coords_xy.min(axis=0)
    xmax, ymax = coords_xy.max(axis=0)

    centered = coords_xy - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = 4.0 * math.sqrt(max(float(eigvals[0]), 0.0))
    minor = 4.0 * math.sqrt(max(float(eigvals[1]), 0.0)) if len(eigvals) > 1 else 0.0
    orientation = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    contour = ordered_boundary(mask)
    sampled = sample_polyline(contour, contour_sample_count)
    perimeter = _polyline_perimeter(contour)
    area = int(mask.sum())
    compactness = float(4.0 * math.pi * area / (perimeter**2)) if perimeter > 1e-9 else 0.0
    width = max(float(xmax - xmin + 1), 1.0)
    height = max(float(ymax - ymin + 1), 1.0)
    mean_curv, max_curv = _curvature_stats(sampled)

    features = ContourFeatures(
        area_px=area,
        perimeter_px=perimeter,
        centroid_xy=[float(centroid[0]), float(centroid[1])],
        bbox_xyxy=[int(xmin), int(ymin), int(xmax), int(ymax)],
        compactness=compactness,
        aspect_ratio=width / height,
        orientation_deg=float(orientation),
        major_axis_px=float(major),
        minor_axis_px=float(minor),
        mean_curvature=mean_curv,
        max_curvature=max_curv,
        contour_points=int(len(contour)),
    )
    return features, sampled


def load_point_cloud(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No XYZ points found in {path}")
    return np.asarray(rows, dtype=float)


def estimate_surface_features(points: np.ndarray, k_neighbors: int = 24, boundary_percentile: float = 90.0) -> tuple[SurfaceFeatures, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must be an Nx3 array")
    if len(points) < max(4, k_neighbors):
        raise ValueError("point cloud does not have enough points for local PCA")

    k = min(k_neighbors, len(points))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    normals = np.zeros_like(points)
    curvature = np.zeros(len(points), dtype=float)
    roughness = np.zeros(len(points), dtype=float)

    for i, neigh_idx in enumerate(indices):
        local = points[neigh_idx]
        centered = local - local.mean(axis=0)
        cov = centered.T @ centered / max(len(local) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        normal = eigvecs[:, 0]
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal
        denom = max(float(eigvals.sum()), 1e-12)
        curvature[i] = float(eigvals[0] / denom)
        roughness[i] = math.sqrt(max(float(eigvals[0]), 0.0))

    kth_distance = distances[:, -1]
    curv_norm = _normalize01(curvature)
    sparse_norm = _normalize01(kth_distance)
    boundary_score = 0.65 * curv_norm + 0.35 * sparse_norm
    threshold = np.percentile(boundary_score, boundary_percentile)
    boundary_flag = boundary_score >= threshold

    centered_all = points - points.mean(axis=0)
    cov_all = np.cov(centered_all.T)
    eigvals_all, eigvecs_all = np.linalg.eigh(cov_all)
    order_all = np.argsort(eigvals_all)[::-1]
    eigvals_all = eigvals_all[order_all]
    eigvecs_all = eigvecs_all[:, order_all]

    features = SurfaceFeatures(
        point_count=int(len(points)),
        bbox_min_xyz=[float(v) for v in points.min(axis=0)],
        bbox_max_xyz=[float(v) for v in points.max(axis=0)],
        centroid_xyz=[float(v) for v in points.mean(axis=0)],
        principal_axes=eigvecs_all.T.astype(float).tolist(),
        axis_lengths=[float(4.0 * math.sqrt(max(v, 0.0))) for v in eigvals_all],
        mean_curvature=float(curvature.mean()),
        max_curvature=float(curvature.max()),
        mean_roughness=float(roughness.mean()),
        boundary_candidate_count=int(boundary_flag.sum()),
        k_neighbors=int(k),
    )

    per_point = np.column_stack([points, normals, curvature, roughness, boundary_score, boundary_flag.astype(int)])
    return features, per_point


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_points_csv(path: Path, header: Iterable[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows.tolist())


def contour_payload(features: ContourFeatures) -> dict:
    return {"type": "contour_roi", "features": asdict(features)}


def surface_payload(features: SurfaceFeatures) -> dict:
    return {"type": "surface_roi", "features": asdict(features)}

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from trajectory_planning.viewpoint_selection import (
    SelectionConfig,
    build_coverage_sets,
    build_selected_rows,
    camera_bbox_diag,
    count_regions,
    prepare_surface_points,
    route_cost,
    transition_cost,
    two_opt_route,
)


@dataclass
class ILPTSPConfig:
    time_limit: float = 180.0
    mip_rel_gap: float = 0.01
    shot_penalty: float = 1e-6
    seed: int = 31


def run_ilp_tsp_baseline(
    candidates: list[dict[str, object]],
    point_features: np.ndarray,
    selection_config: SelectionConfig,
    ilp_config: ILPTSPConfig,
    optimize_route: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    points, point_weights = prepare_surface_points(point_features, selection_config)
    cover_sets = build_coverage_sets(candidates, points)
    selected_order, ilp_summary = solve_weighted_set_cover_ilp(
        cover_sets,
        point_weights,
        selection_config,
        ilp_config,
    )

    route_seed = nearest_neighbor_route(selected_order, candidates, selection_config)
    path_before = route_cost(route_seed, candidates, selection_config)
    optimized_order = two_opt_route(route_seed, candidates, selection_config) if optimize_route else route_seed
    path_after = route_cost(optimized_order, candidates, selection_config)
    selected_rows, final_coverage = build_selected_rows(optimized_order, candidates, cover_sets, len(points))
    weighted_coverage = weighted_coverage_ratio(optimized_order, cover_sets, point_weights)

    summary: dict[str, Any] = {
        "baseline": "ilp_tsp",
        "input_candidate_count": int(len(candidates)),
        "surface_point_count": int(len(points)),
        "selected_view_count": int(len(selected_rows)),
        "coverage_ratio": float(final_coverage),
        "weighted_coverage_ratio": float(weighted_coverage),
        "target_coverage": float(selection_config.target_coverage),
        "path_cost_before_2opt": float(path_before),
        "path_cost_after_2opt": float(path_after),
        "path_cost_reduction": float(path_before - path_after),
        "region_counts": count_regions(selected_rows),
        "config": asdict(selection_config),
        "ilp_config": asdict(ilp_config),
        "ilp_summary": ilp_summary,
    }
    return selected_rows, summary


def solve_weighted_set_cover_ilp(
    cover_sets: list[np.ndarray],
    point_weights: np.ndarray,
    selection_config: SelectionConfig,
    ilp_config: ILPTSPConfig,
) -> tuple[list[int], dict[str, Any]]:
    candidate_count = len(cover_sets)
    point_count = len(point_weights)
    variable_count = candidate_count + point_count
    total_weight = max(float(point_weights.sum()), 1e-12)

    objective = np.zeros(variable_count, dtype=float)
    objective[:candidate_count] = ilp_config.shot_penalty
    objective[candidate_count:] = -point_weights / total_weight

    coverage_constraint = build_coverage_constraint(cover_sets, point_count, variable_count)
    budget_row = coo_matrix(
        (
            np.ones(candidate_count, dtype=float),
            (np.zeros(candidate_count, dtype=int), np.arange(candidate_count, dtype=int)),
        ),
        shape=(1, variable_count),
    ).tocsr()
    constraints = [
        LinearConstraint(coverage_constraint, -np.inf, np.zeros(point_count, dtype=float)),
        LinearConstraint(budget_row, -np.inf, float(selection_config.max_selected)),
    ]
    integrality = np.ones(variable_count, dtype=int)
    bounds = Bounds(np.zeros(variable_count, dtype=float), np.ones(variable_count, dtype=float))

    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(ilp_config.time_limit),
            "mip_rel_gap": float(ilp_config.mip_rel_gap),
            "disp": False,
        },
    )
    if result.x is None:
        raise RuntimeError(f"ILP solver did not return a solution: status={result.status}, message={result.message}")

    x = np.asarray(result.x[:candidate_count], dtype=float)
    selected = [int(idx) for idx in np.flatnonzero(x >= 0.5)]
    ilp_summary = {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun) if result.fun is not None else None,
        "selected_by_ilp": int(len(selected)),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", -1)),
    }
    return selected, ilp_summary


def build_coverage_constraint(
    cover_sets: list[np.ndarray],
    point_count: int,
    variable_count: int,
) -> coo_matrix:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    y_rows = np.arange(point_count, dtype=int)
    rows.append(y_rows)
    cols.append(np.arange(len(cover_sets), len(cover_sets) + point_count, dtype=int))
    data.append(np.ones(point_count, dtype=float))

    for candidate_idx, cover_idx in enumerate(cover_sets):
        if len(cover_idx) == 0:
            continue
        rows.append(np.asarray(cover_idx, dtype=int))
        cols.append(np.full(len(cover_idx), candidate_idx, dtype=int))
        data.append(np.full(len(cover_idx), -1.0, dtype=float))

    return coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(point_count, variable_count),
    ).tocsr()


def nearest_neighbor_route(
    selected: list[int],
    candidates: list[dict[str, object]],
    config: SelectionConfig,
) -> list[int]:
    if len(selected) <= 2:
        return selected[:]

    remaining = set(int(idx) for idx in selected)
    current = max(remaining, key=lambda idx: float(candidates[idx].get("priority_score", 0.0)))
    route = [current]
    remaining.remove(current)
    while remaining:
        next_idx = min(
            remaining,
            key=lambda idx: transition_cost(candidates[current], candidates[idx], config),
        )
        route.append(int(next_idx))
        remaining.remove(int(next_idx))
        current = int(next_idx)
    return route


def weighted_coverage_ratio(
    order: list[int],
    cover_sets: list[np.ndarray],
    point_weights: np.ndarray,
) -> float:
    if len(point_weights) == 0:
        return 0.0
    covered = np.zeros(len(point_weights), dtype=bool)
    for candidate_idx in order:
        covered[cover_sets[int(candidate_idx)]] = True
    return float(point_weights[covered].sum()) / max(float(point_weights.sum()), 1e-12)

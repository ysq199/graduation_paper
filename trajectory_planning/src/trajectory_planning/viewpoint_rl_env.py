from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

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


class ViewpointPlanningEnv:
    """Small Gym-like environment for discrete camera viewpoint selection."""

    def __init__(
        self,
        candidates: list[dict[str, object]],
        point_features: np.ndarray,
        config: SelectionConfig | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("At least one candidate viewpoint is required.")
        self.candidates = candidates
        self.config = config or SelectionConfig()
        self.points, self.point_weights = prepare_surface_points(point_features, self.config)
        self.cover_sets = build_coverage_sets(candidates, self.points)
        self.total_weight = max(float(self.point_weights.sum()), 1e-12)
        self.bbox_diag = camera_bbox_diag(candidates)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.covered = np.zeros(len(self.points), dtype=bool)
        self.selected_order: list[int] = []
        self.remaining = set(range(len(self.candidates)))
        self.current_index: int | None = None
        self.step_count = 0
        return self.observation()

    def observation(self) -> dict[str, Any]:
        return {
            "current_index": self.current_index,
            "step_count": int(self.step_count),
            "coverage_ratio": self.coverage_ratio,
            "weighted_coverage_ratio": self.weighted_coverage_ratio,
            "remaining_count": int(len(self.remaining)),
            "action_mask": self.action_mask(),
        }

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.candidates), dtype=bool)
        if self.done:
            return mask
        for idx in self.remaining:
            if self.marginal_weighted_gain(idx) >= self.config.min_new_coverage:
                mask[idx] = True
        return mask

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= len(self.candidates):
            raise IndexError(f"Action index out of range: {action}")
        if action not in self.remaining:
            raise ValueError(f"Action {action} has already been selected.")

        gain = self.marginal_weighted_gain(action)
        if gain < self.config.min_new_coverage:
            raise ValueError(
                f"Action {action} marginal gain {gain:.6f} is below "
                f"min_new_coverage={self.config.min_new_coverage:.6f}."
            )

        previous_index = self.current_index
        motion_penalty = self.normalized_transition_cost(previous_index, action)
        reward = gain - self.config.motion_weight * motion_penalty

        cover_idx = self.cover_sets[action]
        newly_covered = int((~self.covered[cover_idx]).sum())
        self.covered[cover_idx] = True
        self.selected_order.append(action)
        self.remaining.remove(action)
        self.current_index = action
        self.step_count += 1

        terminated = self.weighted_coverage_ratio >= self.config.target_coverage
        truncated = self.step_count >= self.config.max_selected or not self.action_mask().any()
        info = {
            "action": action,
            "candidate_view_id": int(self.candidates[action]["view_id"]),
            "newly_covered_points": newly_covered,
            "marginal_weighted_gain": float(gain),
            "motion_penalty": float(motion_penalty),
            "selected_count": int(len(self.selected_order)),
            "coverage_ratio": self.coverage_ratio,
            "weighted_coverage_ratio": self.weighted_coverage_ratio,
        }
        return self.observation(), float(reward), bool(terminated), bool(truncated), info

    @property
    def done(self) -> bool:
        return self.weighted_coverage_ratio >= self.config.target_coverage or len(self.selected_order) >= self.config.max_selected

    @property
    def coverage_ratio(self) -> float:
        if len(self.covered) == 0:
            return 0.0
        return float(self.covered.mean())

    @property
    def weighted_coverage_ratio(self) -> float:
        return float(self.point_weights[self.covered].sum()) / self.total_weight

    def marginal_weighted_gain(self, candidate_index: int) -> float:
        cover_idx = self.cover_sets[candidate_index]
        if len(cover_idx) == 0:
            return 0.0
        new_mask = ~self.covered[cover_idx]
        return float(self.point_weights[cover_idx[new_mask]].sum()) / self.total_weight

    def normalized_transition_cost(self, previous_index: int | None, next_index: int) -> float:
        if previous_index is None:
            return 0.0
        cost = transition_cost(self.candidates[previous_index], self.candidates[next_index], self.config)
        return float(cost) / self.bbox_diag


class GreedyBaselinePolicy:
    """Baseline policy that uses the current greedy coverage-minus-motion score."""

    def select_action(self, env: ViewpointPlanningEnv) -> int | None:
        mask = env.action_mask()
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            return None

        best_action: int | None = None
        best_score = -float("inf")
        for action in valid_actions:
            gain = env.marginal_weighted_gain(int(action))
            motion_penalty = env.normalized_transition_cost(env.current_index, int(action))
            score = gain - env.config.motion_weight * motion_penalty
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action


def run_greedy_baseline(
    candidates: list[dict[str, object]],
    point_features: np.ndarray,
    config: SelectionConfig,
    optimize_route: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    env = ViewpointPlanningEnv(candidates, point_features, config)
    policy = GreedyBaselinePolicy()
    env.reset()

    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.select_action(env)
        if action is None:
            truncated = True
            break
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    selected_order = env.selected_order[:]
    path_before = route_cost(selected_order, candidates, config)
    optimized_order = two_opt_route(selected_order, candidates, config) if optimize_route else selected_order
    path_after = route_cost(optimized_order, candidates, config)
    selected_rows, final_coverage = build_selected_rows(optimized_order, candidates, env.cover_sets, len(env.points))
    summary = {
        "baseline": "greedy_env",
        "input_candidate_count": int(len(candidates)),
        "surface_point_count": int(len(env.points)),
        "selected_view_count": int(len(selected_rows)),
        "coverage_ratio": float(final_coverage),
        "weighted_coverage_ratio": float(env.weighted_coverage_ratio),
        "target_coverage": float(config.target_coverage),
        "total_reward": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "path_cost_before_2opt": float(path_before),
        "path_cost_after_2opt": float(path_after),
        "path_cost_reduction": float(path_before - path_after),
        "region_counts": count_regions(selected_rows),
        "config": asdict(config),
    }
    return selected_rows, summary

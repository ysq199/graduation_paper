from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None

from trajectory_planning.viewpoint_rl_env import ViewpointPlanningEnv
from trajectory_planning.viewpoint_selection import (
    SelectionConfig,
    build_selected_rows,
    count_regions,
    route_cost,
    two_opt_route,
)


REGION_CODE = {
    "surface": 0.0,
    "boundary": 1.0 / 3.0,
    "high_curvature": 2.0 / 3.0,
    "boundary_high_curvature": 1.0,
}


@dataclass
class PPORewardConfig:
    gain_scale: float = 10.0
    motion_scale: float = 0.35
    step_penalty: float = 0.005
    target_bonus: float = 3.0
    final_coverage_scale: float = 2.0
    local_jump_threshold: float = 0.35
    local_jump_scale: float = 0.0
    smoothness_scale: float = 0.0
    region_switch_scale: float = 0.0
    intermediate_reward_scale: float = 1.0
    terminal_coverage_scale: float = 0.0
    terminal_path_scale: float = 0.0
    terminal_shot_scale: float = 0.0
    expert_prior_scale: float = 0.0
    expert_next_scale: float = 0.0
    expert_edge_scale: float = 0.0
    expert_edge_jump_scale: float = 0.0
    expert_edge_bandwidth: float = 3.0


class MaskablePPOViewpointEnv(gym.Env if gym is not None else object):
    """Gymnasium wrapper with action_masks() for sb3-contrib MaskablePPO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        candidates: list[dict[str, object]],
        point_features: np.ndarray,
        config: SelectionConfig | None = None,
        reward_config: PPORewardConfig | None = None,
        expert_order: list[int] | None = None,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError(
                "MaskablePPOViewpointEnv requires gymnasium. Install gymnasium, "
                "stable-baselines3 and sb3-contrib before training."
            )
        super().__init__()
        self.core = ViewpointPlanningEnv(candidates, point_features, config)
        self.candidates = candidates
        self.config = self.core.config
        self.reward_config = reward_config or PPORewardConfig()
        self.expert_order = self._build_expert_order(expert_order)
        self.expert_rank = np.full(len(candidates), -1, dtype=np.int32)
        for rank, candidate_index in enumerate(self.expert_order):
            self.expert_rank[candidate_index] = rank
        self.action_space = spaces.Discrete(len(candidates))
        self.features_per_candidate = 14
        self.global_feature_count = 7
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.global_feature_count + len(candidates) * self.features_per_candidate,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.core.reset()
        return self._flat_observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        turn_penalty = self._turn_penalty(int(action))
        region_switch_penalty = self._region_switch_penalty(int(action))
        expert_membership = self._expert_membership_feature(int(action))
        expert_next_score = self._expert_next_score(int(action))
        expert_edge_score = self._expert_edge_score(int(action))
        expert_edge_jump = self._expert_edge_jump_feature(int(action))
        _, _, terminated, truncated, info = self.core.step(int(action))
        info["turn_penalty"] = float(turn_penalty)
        info["region_switch_penalty_raw"] = float(region_switch_penalty)
        info["expert_membership"] = float(expert_membership)
        info["expert_next_score"] = float(expert_next_score)
        info["expert_edge_score"] = float(expert_edge_score)
        info["expert_edge_jump"] = float(expert_edge_jump)
        reward = self._shaped_reward(info, terminated, truncated)
        info["shaped_reward"] = float(reward)
        return self._flat_observation(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.core.action_mask()

    @property
    def selected_order(self) -> list[int]:
        return self.core.selected_order

    @property
    def cover_sets(self) -> list[np.ndarray]:
        return self.core.cover_sets

    @property
    def surface_point_count(self) -> int:
        return len(self.core.points)

    def _flat_observation(self) -> np.ndarray:
        max_steps = max(float(self.config.max_selected), 1.0)
        max_candidates = max(float(len(self.candidates)), 1.0)
        current_index = -1 if self.core.current_index is None else self.core.current_index
        last_motion_cost = self._last_motion_cost()
        selected_region_ratio = self._selected_region_ratio()
        global_features = [
            0.0 if current_index < 0 else (float(current_index) + 1.0) / max_candidates,
            min(float(self.core.step_count) / max_steps, 1.0),
            float(self.core.coverage_ratio),
            float(self.core.weighted_coverage_ratio),
            float(len(self.core.remaining)) / max_candidates,
            last_motion_cost,
            selected_region_ratio,
        ]

        candidate_features: list[float] = []
        mask = self.core.action_mask()
        for idx, candidate in enumerate(self.candidates):
            is_available = 1.0 if mask[idx] else 0.0
            is_selected = 0.0 if idx in self.core.remaining else 1.0
            marginal_gain = min(max(self.core.marginal_weighted_gain(idx), 0.0), 1.0)
            motion_cost = min(max(self.core.normalized_transition_cost(self.core.current_index, idx), 0.0), 1.0)
            priority = min(max(float(candidate.get("priority_score", 0.0)), 0.0), 1.0)
            region = REGION_CODE.get(str(candidate.get("region_type", "surface")), 0.0)
            nearby_score = 1.0 - motion_cost
            same_region = self._same_region_feature(idx)
            direction_alignment = self._direction_alignment_feature(idx)
            expert_membership = self._expert_membership_feature(idx)
            expert_order_progress = self._expert_order_progress_feature(idx)
            expert_next_score = self._expert_next_score(idx)
            expert_edge_score = self._expert_edge_score(idx)
            expert_edge_jump = self._expert_edge_jump_feature(idx)
            candidate_features.extend(
                [
                    is_available,
                    is_selected,
                    marginal_gain,
                    motion_cost,
                    priority,
                    region,
                    nearby_score,
                    same_region,
                    direction_alignment,
                    expert_membership,
                    expert_order_progress,
                    expert_next_score,
                    expert_edge_score,
                    expert_edge_jump,
                ]
            )

        return np.asarray(global_features + candidate_features, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "coverage_ratio": self.core.coverage_ratio,
            "weighted_coverage_ratio": self.core.weighted_coverage_ratio,
            "selected_count": len(self.core.selected_order),
        }

    def _shaped_reward(self, info: dict[str, Any], terminated: bool, truncated: bool) -> float:
        motion_penalty = float(info["motion_penalty"])
        jump_excess = max(0.0, motion_penalty - self.reward_config.local_jump_threshold)
        local_jump_penalty = self.reward_config.local_jump_scale * jump_excess * jump_excess
        smoothness_penalty = self.reward_config.smoothness_scale * float(info["turn_penalty"])
        region_switch_penalty = self.reward_config.region_switch_scale * float(info["region_switch_penalty_raw"])
        expert_prior_reward = self.reward_config.expert_prior_scale * float(info["expert_membership"])
        expert_next_reward = self.reward_config.expert_next_scale * float(info["expert_next_score"])
        expert_edge_reward = self.reward_config.expert_edge_scale * float(info["expert_edge_score"])
        expert_edge_jump_penalty = self.reward_config.expert_edge_jump_scale * float(info["expert_edge_jump"])
        intermediate_reward = (
            self.reward_config.gain_scale * float(info["marginal_weighted_gain"])
            - self.reward_config.motion_scale * motion_penalty
            - local_jump_penalty
            - smoothness_penalty
            - region_switch_penalty
            - self.reward_config.step_penalty
            + expert_prior_reward
            + expert_next_reward
            + expert_edge_reward
            - expert_edge_jump_penalty
        )
        reward = self.reward_config.intermediate_reward_scale * intermediate_reward
        info["local_jump_penalty"] = float(local_jump_penalty)
        info["smoothness_penalty"] = float(smoothness_penalty)
        info["region_switch_penalty"] = float(region_switch_penalty)
        info["expert_prior_reward"] = float(expert_prior_reward)
        info["expert_next_reward"] = float(expert_next_reward)
        info["expert_edge_reward"] = float(expert_edge_reward)
        info["expert_edge_jump_penalty"] = float(expert_edge_jump_penalty)
        terminal_objective = 0.0
        if terminated or truncated:
            terminal_objective = self._terminal_objective()
            reward += terminal_objective
        info["terminal_objective_reward"] = float(terminal_objective)
        if terminated:
            reward += self.reward_config.target_bonus
        elif truncated:
            reward += self.reward_config.final_coverage_scale * self.core.weighted_coverage_ratio
        return float(reward)

    def _terminal_objective(self) -> float:
        if (
            self.reward_config.terminal_coverage_scale == 0.0
            and self.reward_config.terminal_path_scale == 0.0
            and self.reward_config.terminal_shot_scale == 0.0
        ):
            return 0.0
        max_route = self.core.bbox_diag * max(float(self.config.max_selected - 1), 1.0)
        raw_path_cost = route_cost(self.core.selected_order, self.candidates, self.config)
        normalized_path = float(raw_path_cost) / max(max_route, 1e-12)
        normalized_shots = float(len(self.core.selected_order)) / max(float(self.config.max_selected), 1.0)
        return float(
            self.reward_config.terminal_coverage_scale * self.core.weighted_coverage_ratio
            - self.reward_config.terminal_path_scale * normalized_path
            - self.reward_config.terminal_shot_scale * normalized_shots
        )

    def _last_motion_cost(self) -> float:
        order = self.core.selected_order
        if len(order) < 2:
            return 0.0
        return min(max(self.core.normalized_transition_cost(order[-2], order[-1]), 0.0), 1.0)

    def _selected_region_ratio(self) -> float:
        all_regions = {str(candidate.get("region_type", "surface")) for candidate in self.candidates}
        if not all_regions:
            return 0.0
        selected_regions = {
            str(self.candidates[idx].get("region_type", "surface"))
            for idx in self.core.selected_order
        }
        return float(len(selected_regions)) / float(len(all_regions))

    def _same_region_feature(self, candidate_index: int) -> float:
        if self.core.current_index is None:
            return 1.0
        return 1.0 if self._candidate_region(candidate_index) == self._candidate_region(self.core.current_index) else 0.0

    def _direction_alignment_feature(self, candidate_index: int) -> float:
        order = self.core.selected_order
        if len(order) < 2 or self.core.current_index is None:
            return 1.0
        prev_position = self._camera_position(order[-2])
        current_position = self._camera_position(self.core.current_index)
        next_position = self._camera_position(candidate_index)
        incoming = current_position - prev_position
        outgoing = next_position - current_position
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= 1e-12 or outgoing_norm <= 1e-12:
            return 1.0
        cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
        return float((np.clip(cosine, -1.0, 1.0) + 1.0) * 0.5)

    def _turn_penalty(self, candidate_index: int) -> float:
        return 1.0 - self._direction_alignment_feature(candidate_index)

    def _region_switch_penalty(self, candidate_index: int) -> float:
        if self.core.current_index is None:
            return 0.0
        return 0.0 if self._candidate_region(candidate_index) == self._candidate_region(self.core.current_index) else 1.0

    def _candidate_region(self, candidate_index: int) -> str:
        return str(self.candidates[candidate_index].get("region_type", "surface"))

    def _camera_position(self, candidate_index: int) -> np.ndarray:
        candidate = self.candidates[candidate_index]
        return np.asarray(
            [candidate["camera_x"], candidate["camera_y"], candidate["camera_z"]],
            dtype=float,
        )

    def _build_expert_order(self, expert_order: list[int] | None) -> list[int]:
        if expert_order is None:
            return []
        ordered: list[int] = []
        seen: set[int] = set()
        for candidate_index in expert_order:
            idx = int(candidate_index)
            if idx < 0 or idx >= len(self.candidates) or idx in seen:
                continue
            ordered.append(idx)
            seen.add(idx)
        return ordered

    def _expert_membership_feature(self, candidate_index: int) -> float:
        if len(self.expert_order) == 0:
            return 0.0
        return 1.0 if int(self.expert_rank[candidate_index]) >= 0 else 0.0

    def _expert_order_progress_feature(self, candidate_index: int) -> float:
        if len(self.expert_order) == 0:
            return 0.0
        rank = int(self.expert_rank[candidate_index])
        if rank < 0:
            return 0.0
        return float(rank + 1) / float(len(self.expert_order))

    def _expert_next_score(self, candidate_index: int) -> float:
        if len(self.expert_order) == 0:
            return 0.0
        rank = int(self.expert_rank[candidate_index])
        if rank < 0:
            return 0.0
        next_rank = self._next_unselected_expert_rank()
        if next_rank is None:
            return 0.0
        rank_distance = abs(float(rank - next_rank))
        return float(np.exp(-rank_distance / 3.0))

    def _expert_edge_score(self, candidate_index: int) -> float:
        if len(self.expert_order) == 0:
            return 0.0
        candidate_rank = int(self.expert_rank[candidate_index])
        if candidate_rank < 0:
            return 0.0
        current_rank = self._current_expert_rank()
        if current_rank is None:
            return self._expert_next_score(candidate_index)
        next_rank = self._next_unselected_expert_rank_after(current_rank)
        if next_rank is None or candidate_rank <= current_rank:
            return 0.0
        bandwidth = max(float(self.reward_config.expert_edge_bandwidth), 1e-6)
        rank_distance = abs(float(candidate_rank - next_rank))
        return float(np.exp(-rank_distance / bandwidth))

    def _expert_edge_jump_feature(self, candidate_index: int) -> float:
        if len(self.expert_order) == 0:
            return 0.0
        candidate_rank = int(self.expert_rank[candidate_index])
        if candidate_rank < 0:
            return 0.0
        current_rank = self._current_expert_rank()
        if current_rank is None:
            return 0.0
        next_rank = self._next_unselected_expert_rank_after(current_rank)
        if next_rank is None:
            return 0.0
        if candidate_rank <= current_rank:
            return 1.0
        bandwidth = max(float(self.reward_config.expert_edge_bandwidth), 1e-6)
        return float(min(abs(float(candidate_rank - next_rank)) / bandwidth, 1.0))

    def _current_expert_rank(self) -> int | None:
        if self.core.current_index is None:
            return None
        rank = int(self.expert_rank[self.core.current_index])
        if rank < 0:
            return None
        return rank

    def _next_unselected_expert_rank_after(self, current_rank: int) -> int | None:
        for rank in range(current_rank + 1, len(self.expert_order)):
            if self.expert_order[rank] in self.core.remaining:
                return rank
        return None

    def _next_unselected_expert_rank(self) -> int | None:
        for rank, candidate_index in enumerate(self.expert_order):
            if candidate_index in self.core.remaining:
                return rank
        return None


def evaluate_maskable_model(
    model: Any,
    env: MaskablePPOViewpointEnv,
    optimize_route: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    obs, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)

    order = env.selected_order[:]
    path_before = route_cost(order, env.candidates, env.config)
    optimized_order = two_opt_route(order, env.candidates, env.config) if optimize_route else order
    path_after = route_cost(optimized_order, env.candidates, env.config)
    selected_rows, final_coverage = build_selected_rows(
        optimized_order,
        env.candidates,
        env.cover_sets,
        env.surface_point_count,
    )
    summary = {
        "baseline": "maskable_ppo",
        "input_candidate_count": int(len(env.candidates)),
        "surface_point_count": int(env.surface_point_count),
        "selected_view_count": int(len(selected_rows)),
        "coverage_ratio": float(final_coverage),
        "weighted_coverage_ratio": float(env.core.weighted_coverage_ratio),
        "target_coverage": float(env.config.target_coverage),
        "total_reward": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "path_cost_before_2opt": float(path_before),
        "path_cost_after_2opt": float(path_after),
        "path_cost_reduction": float(path_before - path_after),
        "region_counts": count_regions(selected_rows),
        "config": asdict(env.config),
        "reward_config": asdict(env.reward_config),
        "expert_route_count": int(len(env.expert_order)),
        "expert_edge_summary": summarize_expert_edges(order, env),
    }
    return selected_rows, summary


def summarize_expert_edges(order: list[int], env: MaskablePPOViewpointEnv) -> dict[str, float | int]:
    expert_selected_count = sum(1 for idx in order if int(env.expert_rank[idx]) >= 0)
    comparable_edges = 0
    forward_edges = 0
    local_forward_edges = 0
    backward_edges = 0
    rank_jumps: list[float] = []
    bandwidth = max(float(env.reward_config.expert_edge_bandwidth), 1.0)
    for previous, current in zip(order, order[1:]):
        previous_rank = int(env.expert_rank[previous])
        current_rank = int(env.expert_rank[current])
        if previous_rank < 0 or current_rank < 0:
            continue
        comparable_edges += 1
        rank_jump = float(current_rank - previous_rank)
        rank_jumps.append(abs(rank_jump))
        if rank_jump > 0:
            forward_edges += 1
            if rank_jump <= bandwidth:
                local_forward_edges += 1
        else:
            backward_edges += 1
    return {
        "expert_selected_count": int(expert_selected_count),
        "expert_selected_ratio": float(expert_selected_count / max(len(order), 1)),
        "expert_comparable_edge_count": int(comparable_edges),
        "expert_forward_edge_ratio": float(forward_edges / max(comparable_edges, 1)),
        "expert_local_forward_edge_ratio": float(local_forward_edges / max(comparable_edges, 1)),
        "expert_backward_edge_count": int(backward_edges),
        "expert_mean_rank_jump": float(np.mean(rank_jumps)) if rank_jumps else 0.0,
    }

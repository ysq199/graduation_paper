from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from trajectory_planning.viewpoint_selection import (
    SelectionConfig,
    build_coverage_sets,
    build_selected_rows,
    camera_bbox_diag,
    count_regions,
    greedy_weighted_set_cover,
    prepare_surface_points,
    route_cost,
    two_opt_route,
)


@dataclass
class GeneticConfig:
    population_size: int = 80
    generations: int = 120
    tournament_size: int = 4
    elite_count: int = 4
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    path_weight: float = 0.10
    target_coverage_bonus: float = 0.15
    seed: int = 23


def run_genetic_baseline(
    candidates: list[dict[str, object]],
    point_features: np.ndarray,
    selection_config: SelectionConfig,
    genetic_config: GeneticConfig,
    optimize_route: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    points, point_weights = prepare_surface_points(point_features, selection_config)
    cover_sets = build_coverage_sets(candidates, points)
    total_weight = max(float(point_weights.sum()), 1e-12)
    route_norm = camera_bbox_diag(candidates) * max(selection_config.max_selected - 1, 1)
    rng = np.random.default_rng(genetic_config.seed)
    chromosome_len = min(selection_config.max_selected, len(candidates))

    greedy_order, _ = greedy_weighted_set_cover(candidates, cover_sets, point_weights, selection_config)
    population = initial_population(candidates, greedy_order, chromosome_len, genetic_config.population_size, rng)

    best_chromosome = population[0]
    best_eval = evaluate_chromosome(best_chromosome, candidates, cover_sets, point_weights, total_weight, route_norm, selection_config, genetic_config)
    trace: list[dict[str, float]] = []

    for generation in range(genetic_config.generations):
        evaluated = [
            (chromosome, evaluate_chromosome(chromosome, candidates, cover_sets, point_weights, total_weight, route_norm, selection_config, genetic_config))
            for chromosome in population
        ]
        evaluated.sort(key=lambda item: item[1]["fitness"], reverse=True)
        if evaluated[0][1]["fitness"] > best_eval["fitness"]:
            best_chromosome = evaluated[0][0].copy()
            best_eval = evaluated[0][1]
        trace.append(
            {
                "generation": float(generation),
                "best_fitness": float(best_eval["fitness"]),
                "best_weighted_coverage": float(best_eval["weighted_coverage_ratio"]),
                "best_path_cost": float(best_eval["path_cost"]),
            }
        )

        next_population = [item[0].copy() for item in evaluated[: genetic_config.elite_count]]
        while len(next_population) < genetic_config.population_size:
            parent_a = tournament_select(evaluated, genetic_config.tournament_size, rng)
            parent_b = tournament_select(evaluated, genetic_config.tournament_size, rng)
            if rng.random() < genetic_config.crossover_rate:
                child = ordered_subset_crossover(parent_a, parent_b, len(candidates), rng)
            else:
                child = parent_a.copy()
            mutate(child, len(candidates), genetic_config.mutation_rate, rng)
            next_population.append(child)
        population = next_population

    selected_order = [int(value) for value in best_chromosome]
    path_before = route_cost(selected_order, candidates, selection_config)
    optimized_order = two_opt_route(selected_order, candidates, selection_config) if optimize_route else selected_order
    path_after = route_cost(optimized_order, candidates, selection_config)
    selected_rows, final_coverage = build_selected_rows(optimized_order, candidates, cover_sets, len(points))
    final_eval = evaluate_chromosome(np.asarray(optimized_order, dtype=int), candidates, cover_sets, point_weights, total_weight, route_norm, selection_config, genetic_config)

    summary = {
        "baseline": "genetic_algorithm",
        "input_candidate_count": int(len(candidates)),
        "surface_point_count": int(len(points)),
        "selected_view_count": int(len(selected_rows)),
        "coverage_ratio": float(final_coverage),
        "weighted_coverage_ratio": float(final_eval["weighted_coverage_ratio"]),
        "target_coverage": float(selection_config.target_coverage),
        "fitness": float(final_eval["fitness"]),
        "path_cost_before_2opt": float(path_before),
        "path_cost_after_2opt": float(path_after),
        "path_cost_reduction": float(path_before - path_after),
        "region_counts": count_regions(selected_rows),
        "config": asdict(selection_config),
        "genetic_config": asdict(genetic_config),
        "trace_tail": trace[-10:],
    }
    return selected_rows, summary


def initial_population(
    candidates: list[dict[str, object]],
    greedy_order: list[int],
    chromosome_len: int,
    population_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    candidate_count = len(candidates)
    population: list[np.ndarray] = []
    if greedy_order:
        population.append(pad_unique(greedy_order, candidate_count, chromosome_len, rng))

    priority_order = sorted(
        range(candidate_count),
        key=lambda idx: float(candidates[idx].get("priority_score", 0.0)),
        reverse=True,
    )
    population.append(pad_unique(priority_order[:chromosome_len], candidate_count, chromosome_len, rng))

    while len(population) < population_size:
        population.append(rng.choice(candidate_count, size=chromosome_len, replace=False).astype(int))
    return population


def pad_unique(order: list[int], candidate_count: int, chromosome_len: int, rng: np.random.Generator) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for value in order:
        idx = int(value)
        if idx not in seen and 0 <= idx < candidate_count:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= chromosome_len:
            return np.asarray(selected, dtype=int)

    remaining = [idx for idx in range(candidate_count) if idx not in seen]
    rng.shuffle(remaining)
    selected.extend(remaining[: chromosome_len - len(selected)])
    return np.asarray(selected, dtype=int)


def evaluate_chromosome(
    chromosome: np.ndarray,
    candidates: list[dict[str, object]],
    cover_sets: list[np.ndarray],
    point_weights: np.ndarray,
    total_weight: float,
    route_norm: float,
    selection_config: SelectionConfig,
    genetic_config: GeneticConfig,
) -> dict[str, float]:
    covered = np.zeros(len(point_weights), dtype=bool)
    for candidate_idx in chromosome:
        covered[cover_sets[int(candidate_idx)]] = True

    weighted_coverage = float(point_weights[covered].sum()) / total_weight
    coverage = float(covered.mean()) if len(covered) else 0.0
    path_cost = route_cost([int(value) for value in chromosome], candidates, selection_config)
    normalized_path = path_cost / max(route_norm, 1e-12)
    target_bonus = genetic_config.target_coverage_bonus if weighted_coverage >= selection_config.target_coverage else 0.0
    fitness = weighted_coverage + target_bonus - genetic_config.path_weight * normalized_path
    return {
        "fitness": float(fitness),
        "coverage_ratio": float(coverage),
        "weighted_coverage_ratio": float(weighted_coverage),
        "path_cost": float(path_cost),
        "normalized_path_cost": float(normalized_path),
    }


def tournament_select(
    evaluated: list[tuple[np.ndarray, dict[str, float]]],
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    size = min(max(tournament_size, 1), len(evaluated))
    indices = rng.choice(len(evaluated), size=size, replace=False)
    best = max((evaluated[int(idx)] for idx in indices), key=lambda item: item[1]["fitness"])
    return best[0].copy()


def ordered_subset_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    candidate_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    length = len(parent_a)
    if length < 2:
        return parent_a.copy()
    left, right = sorted(rng.choice(length, size=2, replace=False))
    child = np.full(length, -1, dtype=int)
    child[left : right + 1] = parent_a[left : right + 1]
    used = {int(value) for value in child if value >= 0}
    fill_values = [int(value) for value in parent_b if int(value) not in used]
    fill_values.extend(idx for idx in range(candidate_count) if idx not in used and idx not in fill_values)
    fill_iter = iter(fill_values)
    for idx in range(length):
        if child[idx] < 0:
            child[idx] = next(fill_iter)
    return child


def mutate(chromosome: np.ndarray, candidate_count: int, mutation_rate: float, rng: np.random.Generator) -> None:
    if rng.random() > mutation_rate:
        return

    if len(chromosome) >= 2 and rng.random() < 0.55:
        a, b = rng.choice(len(chromosome), size=2, replace=False)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        return

    replace_idx = int(rng.integers(0, len(chromosome)))
    used = set(int(value) for value in chromosome)
    available = [idx for idx in range(candidate_count) if idx not in used]
    if available:
        chromosome[replace_idx] = int(rng.choice(available))

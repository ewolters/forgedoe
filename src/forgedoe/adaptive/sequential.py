"""Sequential DOE — the orchestrator for multi-batch experimentation.

"I have a budget of 20 runs. Design an initial batch of 8.
After results, suggest the next 4. After those, suggest the
final 8 — but only in the promising region."

This is Bayesian adaptive DOE applied as a workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bayesian_doe import (
    AdaptiveExperiment,
    add_observation,
    generate_initial_points,
    get_current_best,
    should_stop,
    suggest_next_point,
)


@dataclass
class SequentialPlan:
    """A sequential DOE plan with batched execution."""

    experiment: AdaptiveExperiment
    total_budget: int
    batch_sizes: list[int]  # [8, 4, 8] = three batches
    current_batch: int = 0
    batches: list[dict] = field(default_factory=list)


def create_sequential_plan(
    factor_names: list[str],
    factor_bounds: list[tuple[float, float]],
    total_budget: int = 20,
    n_batches: int = 3,
    initial_fraction: float = 0.4,
    acquisition: str = "expected_improvement",
) -> SequentialPlan:
    """Create a sequential DOE plan.

    Args:
        total_budget: maximum total runs
        n_batches: number of batches to split into
        initial_fraction: fraction of budget for initial exploration
    """
    initial_size = max(4, int(total_budget * initial_fraction))
    remaining = total_budget - initial_size
    subsequent_size = remaining // max(1, n_batches - 1)
    last_batch = remaining - subsequent_size * (n_batches - 2) if n_batches > 2 else remaining

    batch_sizes = [initial_size]
    for i in range(n_batches - 2):
        batch_sizes.append(subsequent_size)
    if n_batches > 1:
        batch_sizes.append(last_batch)

    experiment = AdaptiveExperiment(
        factor_names=factor_names,
        factor_bounds=factor_bounds,
        acquisition=acquisition,
    )

    return SequentialPlan(
        experiment=experiment,
        total_budget=total_budget,
        batch_sizes=batch_sizes,
    )


def get_next_batch(plan: SequentialPlan) -> list[list[float]]:
    """Get the next batch of experimental points.

    First batch: space-filling (LHS).
    Subsequent batches: Bayesian adaptive (focused on promising regions).
    """
    if plan.current_batch >= len(plan.batch_sizes):
        return []  # budget exhausted

    batch_size = plan.batch_sizes[plan.current_batch]

    if plan.current_batch == 0:
        # Initial exploration
        points = generate_initial_points(plan.experiment, n_points=batch_size)
    else:
        # Adaptive — suggest points one at a time
        points = []
        for _ in range(batch_size):
            point = suggest_next_point(plan.experiment)
            points.append(point)
            # Temporarily add as observation with predicted value to diversify
            # (will be replaced with actual observation later)

    plan.batches.append({
        "batch": plan.current_batch + 1,
        "size": batch_size,
        "points": points,
        "results": None,
    })

    return points


def record_batch_results(
    plan: SequentialPlan,
    results: list[float],
) -> dict:
    """Record results for the current batch and advance.

    Args:
        results: response values for each point in the batch

    Returns:
        Summary of batch + stopping check
    """
    if not plan.batches:
        return {"error": "No batch to record"}

    batch = plan.batches[-1]
    points = batch["points"]

    if len(results) != len(points):
        return {"error": f"Expected {len(points)} results, got {len(results)}"}

    for point, response in zip(points, results):
        add_observation(plan.experiment, point, response)

    batch["results"] = results
    plan.current_batch += 1

    # Check stopping
    stop = should_stop(
        plan.experiment,
        budget=plan.total_budget,
    )

    best = get_current_best(plan.experiment)

    return {
        "batch": plan.current_batch,
        "points_added": len(results),
        "total_observations": len(plan.experiment.y_observed),
        "best": best,
        "should_stop": stop["should_stop"],
        "stop_reason": stop["reason"],
        "remaining_budget": plan.total_budget - len(plan.experiment.y_observed),
        "next_batch_size": plan.batch_sizes[plan.current_batch] if plan.current_batch < len(plan.batch_sizes) else 0,
    }

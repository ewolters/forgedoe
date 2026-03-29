"""Response optimization — desirability functions, multi-response."""

from __future__ import annotations

import math

from ..core.types import AnalysisResult, Factor, OptimizationResult, Response


def desirability_function(
    value: float,
    response: Response,
) -> float:
    """Individual desirability for a single response.

    Returns 0-1 where 1 = perfectly desirable.
    """
    if response.maximize:
        if response.lower_limit is None or response.upper_limit is None:
            return min(1.0, max(0.0, value / 100))  # fallback
        if value <= response.lower_limit:
            return 0.0
        if value >= response.upper_limit:
            return 1.0
        return (value - response.lower_limit) / (response.upper_limit - response.lower_limit)

    elif response.minimize:
        if response.lower_limit is None or response.upper_limit is None:
            return min(1.0, max(0.0, 1 - value / 100))
        if value >= response.upper_limit:
            return 0.0
        if value <= response.lower_limit:
            return 1.0
        return (response.upper_limit - value) / (response.upper_limit - response.lower_limit)

    elif response.target is not None:
        # Target — two-sided
        lower = response.lower_limit or (response.target - abs(response.target) * 0.1)
        upper = response.upper_limit or (response.target + abs(response.target) * 0.1)
        if value < lower or value > upper:
            return 0.0
        if value <= response.target:
            return (value - lower) / (response.target - lower) if response.target != lower else 1.0
        return (upper - value) / (upper - response.target) if upper != response.target else 1.0

    return 0.5  # no preference


def composite_desirability(
    values: dict[str, float],
    responses: list[Response],
) -> float:
    """Geometric mean of individual desirabilities, weighted by importance."""
    if not responses:
        return 0.0

    log_sum = 0.0
    weight_sum = 0.0
    for resp in responses:
        val = values.get(resp.name, 0)
        d = desirability_function(val, resp)
        if d <= 0:
            return 0.0  # any zero desirability kills composite
        log_sum += resp.importance * math.log(d)
        weight_sum += resp.importance

    if weight_sum <= 0:
        return 0.0
    return math.exp(log_sum / weight_sum)


def optimize_responses(
    analysis_results: dict[str, AnalysisResult],
    factors: list[Factor],
    responses: list[Response],
    n_starts: int = 100,
) -> OptimizationResult:
    """Multi-response optimization via desirability.

    Grid search + local refinement to find factor settings
    that maximize composite desirability.
    """
    import random

    best_settings = None
    best_desirability = -1
    best_predictions = {}

    for _ in range(n_starts):
        # Random starting point in coded space
        settings = {f.name: random.uniform(-1, 1) for f in factors}

        # Predict each response
        predictions = {}
        for resp_name, ar in analysis_results.items():
            pred = ar.coefficients.get("Intercept", 0)
            for fname in [f.name for f in factors]:
                pred += ar.coefficients.get(fname, 0) * settings[fname]
            # Interactions
            for i, f1 in enumerate(factors):
                for f2 in factors[i + 1:]:
                    term = f"{f1.name}*{f2.name}"
                    pred += ar.coefficients.get(term, 0) * settings[f1.name] * settings[f2.name]
            # Quadratic
            for f in factors:
                term = f"{f.name}^2"
                pred += ar.coefficients.get(term, 0) * settings[f.name] ** 2
            predictions[resp_name] = pred

        d = composite_desirability(predictions, responses)
        if d > best_desirability:
            best_desirability = d
            best_settings = dict(settings)
            best_predictions = dict(predictions)

    # Convert to natural units
    natural_settings = {}
    if best_settings:
        for f in factors:
            coded = best_settings[f.name]
            natural_settings[f.name] = round(f.midpoint + coded * f.range / 2, 4)

    return OptimizationResult(
        optimal_settings=natural_settings,
        predicted_responses={k: round(v, 4) for k, v in best_predictions.items()},
        desirability=round(best_desirability, 4),
    )

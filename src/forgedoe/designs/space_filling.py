"""Space-filling designs — Latin Hypercube, maximin, Sobol."""

from __future__ import annotations

import random

from ..core.types import DesignMatrix, Factor


def latin_hypercube(factors: list[Factor], n_samples: int = 20, randomize: bool = True) -> DesignMatrix:
    """Latin Hypercube Sampling (LHS).

    Ensures each factor's range is evenly sampled.
    Good for computer experiments and initial space exploration.
    """
    k = len(factors)
    matrix = []

    # Create intervals for each factor
    for j in range(k):
        col = [(i + random.random()) / n_samples for i in range(n_samples)]
        random.shuffle(col)
        if not matrix:
            matrix = [[0.0] * k for _ in range(n_samples)]
        for i in range(n_samples):
            # Map [0,1] to coded [-1, 1]
            matrix[i][j] = -1 + 2 * col[i]

    run_order = list(range(1, n_samples + 1))
    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True, design_type=f"Latin Hypercube ({n_samples} samples)",
    )


def maximin_lhs(factors: list[Factor], n_samples: int = 20, n_iterations: int = 100) -> DesignMatrix:
    """Maximin Latin Hypercube — optimized LHS that maximizes minimum distance.

    Generates multiple LHS designs and picks the one with the largest
    minimum inter-point distance.
    """
    import math

    best_design = None
    best_min_dist = -1

    for _ in range(n_iterations):
        candidate = latin_hypercube(factors, n_samples, randomize=True)

        # Compute minimum pairwise distance
        min_dist = float("inf")
        for i in range(len(candidate.matrix)):
            for j in range(i + 1, len(candidate.matrix)):
                dist = math.sqrt(sum(
                    (candidate.matrix[i][d] - candidate.matrix[j][d]) ** 2
                    for d in range(len(factors))
                ))
                min_dist = min(min_dist, dist)

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_design = candidate

    if best_design:
        best_design.design_type = f"Maximin LHS ({n_samples} samples, min_dist={best_min_dist:.4f})"
    return best_design

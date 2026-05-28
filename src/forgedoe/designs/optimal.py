"""Optimal designs — D-optimal, I-optimal via coordinate exchange."""

from __future__ import annotations

import itertools

import numpy as np

from ..core.types import DesignMatrix, Factor


def _build_model_matrix(X: np.ndarray, model: str = "linear") -> np.ndarray:
    """Build expanded model matrix from coded factor levels.

    Args:
        X: n×k matrix of coded factor levels
        model: 'linear', 'interaction', or 'quadratic'
    """
    n, k = X.shape
    cols = [np.ones(n)]  # intercept

    # Main effects
    for j in range(k):
        cols.append(X[:, j].astype(float))

    # Two-factor interactions
    if model in ("interaction", "quadratic"):
        for i in range(k):
            for j in range(i + 1, k):
                cols.append(X[:, i] * X[:, j])

    # Quadratic terms
    if model == "quadratic":
        for j in range(k):
            cols.append(X[:, j].astype(float) ** 2)

    return np.column_stack(cols)


def _n_params(k: int, model: str) -> int:
    """Number of model parameters for k factors."""
    n = 1 + k  # intercept + main effects
    if model in ("interaction", "quadratic"):
        n += k * (k - 1) // 2
    if model == "quadratic":
        n += k
    return n


def d_optimal(factors: list[Factor], n_runs: int, model: str = "linear",
              max_iter: int = 100, seed: int | None = None) -> DesignMatrix:
    """D-optimal design via coordinate exchange algorithm.

    Maximizes |X'X| (determinant of information matrix), which minimizes
    the volume of the confidence ellipsoid for the model parameters.

    Args:
        factors: list of Factor objects
        n_runs: number of experimental runs (>= number of parameters)
        model: 'linear', 'interaction', or 'quadratic'
        max_iter: maximum coordinate exchange iterations
        seed: random seed
    """
    rng = np.random.RandomState(seed)
    k = len(factors)
    n_par = _n_params(k, model)

    if n_runs < n_par:
        n_runs = n_par  # minimum for estimability

    # Candidate levels: -1, 0, +1 for each factor
    candidate_levels = [[-1, 0, 1]] * k

    # Initialize with random candidate points
    design = np.zeros((n_runs, k))
    for i in range(n_runs):
        for j in range(k):
            design[i, j] = rng.choice(candidate_levels[j])

    # Coordinate exchange
    for iteration in range(max_iter):
        improved = False
        for i in range(n_runs):
            for j in range(k):
                X_full = _build_model_matrix(design, model)
                try:
                    best_det = np.linalg.det(X_full.T @ X_full)
                except np.linalg.LinAlgError:
                    best_det = 0.0
                best_level = design[i, j]

                for level in candidate_levels[j]:
                    if level != design[i, j]:
                        design[i, j] = level
                        X_try = _build_model_matrix(design, model)
                        try:
                            new_det = np.linalg.det(X_try.T @ X_try)
                        except np.linalg.LinAlgError:
                            new_det = 0.0

                        if new_det > best_det:
                            best_det = new_det
                            best_level = level
                            improved = True

                design[i, j] = best_level

        if not improved:
            break

    matrix = design.tolist()
    run_order = list(range(1, n_runs + 1))
    # Shuffle run_order and matrix rows together so they stay correlated
    combined = list(zip(run_order, matrix))
    rng.shuffle(combined)
    run_order = [c[0] for c in combined]
    matrix = [c[1] for c in combined]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=f"D-Optimal ({model}, {n_runs} runs, {n_par} params, {iteration + 1} iters)",
    )


def i_optimal(factors: list[Factor], n_runs: int, model: str = "quadratic",
              max_iter: int = 100, seed: int | None = None) -> DesignMatrix:
    """I-optimal design via coordinate exchange.

    Minimizes average prediction variance across the design space.
    Better than D-optimal when the goal is prediction accuracy
    rather than parameter estimation.

    Args:
        factors: list of Factor objects
        n_runs: number of experimental runs
        model: 'linear', 'interaction', or 'quadratic'
        max_iter: maximum iterations
        seed: random seed
    """
    rng = np.random.RandomState(seed)
    k = len(factors)
    n_par = _n_params(k, model)

    if n_runs < n_par:
        n_runs = n_par

    # Build prediction grid — uniform points across design space
    grid_per_dim = max(3, min(5, int(100 ** (1.0 / k))))
    grid_levels = np.linspace(-1, 1, grid_per_dim)
    grid_points = np.array(list(itertools.product(*([grid_levels] * k))))
    X_grid = _build_model_matrix(grid_points, model)

    candidate_levels = [list(np.linspace(-1, 1, 5))] * k  # 5-level grid: -1, -0.5, 0, 0.5, 1

    # Initialize
    design = np.zeros((n_runs, k))
    for i in range(n_runs):
        for j in range(k):
            design[i, j] = rng.choice(candidate_levels[j])

    def i_criterion(D):
        """Average prediction variance across grid."""
        X_d = _build_model_matrix(D, model)
        try:
            XtX_inv = np.linalg.inv(X_d.T @ X_d)
        except np.linalg.LinAlgError:
            return 1e20
        # avg(x' (X'X)^-1 x) across grid
        pred_var = np.sum(X_grid @ XtX_inv * X_grid, axis=1)
        return np.mean(pred_var)

    # Coordinate exchange minimizing I-criterion
    for iteration in range(max_iter):
        improved = False
        for i in range(n_runs):
            for j in range(k):
                best_crit = i_criterion(design)
                best_level = design[i, j]

                for level in candidate_levels[j]:
                    if level != design[i, j]:
                        design[i, j] = level
                        new_crit = i_criterion(design)
                        if new_crit < best_crit:
                            best_crit = new_crit
                            best_level = level
                            improved = True

                design[i, j] = best_level

        if not improved:
            break

    matrix = design.tolist()
    run_order = list(range(1, n_runs + 1))
    combined = list(zip(run_order, matrix))
    rng.shuffle(combined)
    run_order = [c[0] for c in combined]
    matrix = [c[1] for c in combined]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=f"I-Optimal ({model}, {n_runs} runs, {iteration + 1} iters)",
    )

"""Mixture designs — for experiments where factors are proportions summing to 1.

Used in pharmaceutical formulation, food science, chemical blending, materials.
Factors are component proportions (x₁ + x₂ + ... + xₙ = 1).

Design types:
- Simplex lattice: evenly spaced points on the simplex
- Simplex centroid: vertices, edge midpoints, face centroids, overall centroid
- Extreme vertices: for constrained mixture spaces (lower/upper bounds on components)
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from ..core.types import DesignMatrix, Factor


def simplex_lattice(n_components: int, degree: int = 2,
                    randomize: bool = True) -> DesignMatrix:
    """Simplex lattice design — {n, m} design.

    Points where each component takes values 0, 1/m, 2/m, ..., 1
    and all components sum to 1.

    Args:
        n_components: number of mixture components (q)
        degree: lattice degree (m). Higher = more points.
                m=1: vertices only
                m=2: vertices + edge midpoints
                m=3: vertices + 1/3 and 2/3 points
    """
    if n_components < 2:
        raise ValueError("Mixture design requires at least 2 components")
    if degree < 1:
        raise ValueError("Degree must be >= 1")

    q = n_components
    m = degree

    # Generate all compositions of m into q parts
    # Each composition (a₁, a₂, ..., aₙ) where sum = m, aᵢ >= 0
    # gives proportions (a₁/m, a₂/m, ..., aₙ/m)
    points = []
    for combo in _compositions(m, q):
        point = [c / m for c in combo]
        points.append(point)

    matrix = points
    n_runs = len(matrix)
    run_order = list(range(1, n_runs + 1))

    if randomize:
        combined = list(zip(run_order, matrix))
        random.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    factors = [Factor(f"x{i+1}", 0.0, 1.0) for i in range(q)]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=False,
        design_type=f"Simplex Lattice {{{q},{m}}} ({n_runs} runs)",
    )


def simplex_centroid(n_components: int, augment_axial: bool = False,
                     randomize: bool = True) -> DesignMatrix:
    """Simplex centroid design — 2^q - 1 points.

    Includes:
    - Pure components (vertices): (1,0,...,0)
    - Binary blends (edge midpoints): (1/2, 1/2, 0, ..., 0)
    - Ternary blends (face centroids): (1/3, 1/3, 1/3, 0, ..., 0)
    - ... up to the overall centroid (1/q, 1/q, ..., 1/q)

    Args:
        n_components: number of mixture components
        augment_axial: add interior axial check blends
        randomize: randomize run order
    """
    if n_components < 2:
        raise ValueError("Mixture design requires at least 2 components")

    q = n_components
    points = []

    # Generate all subsets of size 1, 2, ..., q
    components = list(range(q))
    for size in range(1, q + 1):
        for subset in itertools.combinations(components, size):
            # Equal proportions for components in subset, 0 for others
            point = [0.0] * q
            proportion = 1.0 / size
            for idx in subset:
                point[idx] = proportion
            points.append(point)

    # Axial check blends (augmentation)
    if augment_axial and q >= 3:
        # Interior axial points: each component at (2q-1)/(2q), rest at 1/(2q(q-1))
        for i in range(q):
            point = [1.0 / (2 * q * (q - 1))] * q
            point[i] = (2 * q - 1) / (2 * q)
            # Normalize to sum to 1
            s = sum(point)
            point = [p / s for p in point]
            points.append(point)

    matrix = points
    n_runs = len(matrix)
    run_order = list(range(1, n_runs + 1))

    if randomize:
        combined = list(zip(run_order, matrix))
        random.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    factors = [Factor(f"x{i+1}", 0.0, 1.0) for i in range(q)]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=False,
        design_type=f"Simplex Centroid ({n_runs} runs, {q} components"
                    + (", axial augmented" if augment_axial else "") + ")",
    )


def extreme_vertices(n_components: int, lower_bounds: list[float] | None = None,
                     upper_bounds: list[float] | None = None,
                     n_interior: int = 0,
                     randomize: bool = True) -> DesignMatrix:
    """Extreme vertices design for constrained mixture space.

    When components have lower and/or upper bounds, the feasible region
    is a sub-simplex. This design places points at the vertices of that
    constrained region.

    Args:
        n_components: number of components
        lower_bounds: minimum proportion for each component (default 0)
        upper_bounds: maximum proportion for each component (default 1)
        n_interior: number of interior points to add (centroid + random)
        randomize: randomize run order
    """
    q = n_components
    if lower_bounds is None:
        lower_bounds = [0.0] * q
    if upper_bounds is None:
        upper_bounds = [1.0] * q

    if len(lower_bounds) != q or len(upper_bounds) != q:
        raise ValueError("Bounds must match number of components")

    # Validate bounds
    if sum(lower_bounds) > 1.0:
        raise ValueError(f"Sum of lower bounds ({sum(lower_bounds):.3f}) exceeds 1.0")
    if sum(upper_bounds) < 1.0:
        raise ValueError(f"Sum of upper bounds ({sum(upper_bounds):.3f}) is less than 1.0")

    # Use pseudo-component transformation
    # L_i = lower bound for component i
    # x_i = L_i + (1 - sum(L)) * x'_i where x' are pseudo-components
    L_sum = sum(lower_bounds)
    slack = 1.0 - L_sum

    if slack <= 0:
        # All slack consumed by lower bounds — only one point
        return DesignMatrix(
            factors=[Factor(f"x{i+1}", lower_bounds[i], upper_bounds[i]) for i in range(q)],
            matrix=[lower_bounds], run_order=[1],
            is_coded=False, design_type=f"Extreme Vertices (1 run, fully constrained)",
        )

    # Adjusted upper bounds in pseudo-component space
    U_pseudo = [(upper_bounds[i] - lower_bounds[i]) / slack for i in range(q)]

    # Generate vertices of constrained simplex
    # Start with simplex lattice degree 1 (vertices) in pseudo-space
    # then filter by upper bound constraints
    vertices = []

    # Vertices: each pseudo-component takes max possible, others fill remainder
    # Use recursive enumeration of feasible vertices
    _find_vertices(q, U_pseudo, [], 1.0, vertices)

    # Remove duplicates (within tolerance)
    unique = []
    for v in vertices:
        is_dup = False
        for u in unique:
            if all(abs(v[i] - u[i]) < 1e-8 for i in range(q)):
                is_dup = True
                break
        if not is_dup:
            unique.append(v)

    # Convert back to original proportions
    points = []
    for v in unique:
        point = [lower_bounds[i] + slack * v[i] for i in range(q)]
        points.append(point)

    # Add centroid and interior points
    if n_interior > 0 and len(points) > 0:
        # Centroid of feasible region
        centroid = [sum(p[i] for p in points) / len(points) for i in range(q)]
        points.append(centroid)

        # Random interior points along edges from centroid to vertices
        for _ in range(min(n_interior - 1, len(unique))):
            vi = random.randint(0, len(unique) - 1)
            t = random.uniform(0.3, 0.7)
            interior = [centroid[j] * t + points[vi][j] * (1 - t) for j in range(q)]
            points.append(interior)

    matrix = points
    n_runs = len(matrix)
    run_order = list(range(1, n_runs + 1))

    if randomize:
        combined = list(zip(run_order, matrix))
        random.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    factors = [Factor(f"x{i+1}", lower_bounds[i], upper_bounds[i]) for i in range(q)]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=False,
        design_type=f"Extreme Vertices ({n_runs} runs, {q} components)",
    )


def _compositions(n: int, k: int) -> list[list[int]]:
    """Generate all compositions of n into k non-negative parts."""
    if k == 1:
        return [[n]]
    result = []
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            result.append([i] + rest)
    return result


def _find_vertices(q: int, U: list[float], partial: list[float],
                   remaining: float, result: list):
    """Recursively find vertices of the constrained simplex."""
    idx = len(partial)
    if idx == q - 1:
        # Last component: must take remaining value
        if remaining <= U[idx] + 1e-10:
            result.append(partial + [min(remaining, U[idx])])
        return

    # Try extreme values for this component: 0 and min(remaining, U[idx])
    for val in [0.0, min(remaining, U[idx])]:
        new_remaining = remaining - val
        if new_remaining >= 0 - 1e-10:
            _find_vertices(q, U, partial + [val], max(0, new_remaining), result)

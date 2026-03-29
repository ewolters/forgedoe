"""Response surface designs — CCD, Box-Behnken."""

from __future__ import annotations

import itertools

from ..core.types import DesignMatrix, Factor


def central_composite_design(
    factors: list[Factor],
    alpha: str = "rotatable",
    center_points: int = 5,
    randomize: bool = True,
) -> DesignMatrix:
    """Central Composite Design (CCD).

    Factorial points + axial (star) points + center points.

    Args:
        alpha: "rotatable" (alpha = 2^(k/4)), "face" (alpha = 1), "spherical"
        center_points: number of center point replicates
    """
    import math
    import random

    k = len(factors)

    # Factorial portion (2^k or 2^(k-1) for large k)
    if k <= 5:
        factorial_points = [list(row) for row in itertools.product([-1, 1], repeat=k)]
    else:
        # Half fraction for large k
        factorial_points = [list(row) for row in itertools.product([-1, 1], repeat=k)]
        factorial_points = factorial_points[::2]

    # Axial (star) points
    if alpha == "rotatable":
        alpha_val = math.pow(len(factorial_points), 0.25)
    elif alpha == "face":
        alpha_val = 1.0
    elif alpha == "spherical":
        alpha_val = math.sqrt(k)
    else:
        alpha_val = float(alpha)

    axial_points = []
    for i in range(k):
        plus = [0.0] * k
        minus = [0.0] * k
        plus[i] = alpha_val
        minus[i] = -alpha_val
        axial_points.append(plus)
        axial_points.append(minus)

    # Center points
    center = [[0.0] * k for _ in range(center_points)]

    matrix = factorial_points + axial_points + center
    run_order = list(range(1, len(matrix) + 1))
    if randomize:
        combined = list(zip(run_order, matrix))
        random.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=f"CCD {alpha} ({len(matrix)} runs, alpha={alpha_val:.3f})",
        center_points=center_points,
    )


def box_behnken_design(factors: list[Factor], center_points: int = 3, randomize: bool = True) -> DesignMatrix:
    """Box-Behnken design.

    Three-level design that avoids extreme corners.
    Good for 3-7 factors. Fewer runs than CCD for 4+ factors.
    """
    import random

    k = len(factors)
    if k < 3:
        return central_composite_design(factors, alpha="face", center_points=center_points)

    # Generate Box-Behnken by combining 2^2 factorials for each pair
    # with other factors at center (0)
    matrix = []
    for i in range(k):
        for j in range(i + 1, k):
            for combo in itertools.product([-1, 1], repeat=2):
                row = [0.0] * k
                row[i] = combo[0]
                row[j] = combo[1]
                matrix.append(row)

    # Center points
    for _ in range(center_points):
        matrix.append([0.0] * k)

    run_order = list(range(1, len(matrix) + 1))
    if randomize:
        combined = list(zip(run_order, matrix))
        random.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=f"Box-Behnken ({len(matrix)} runs)",
        center_points=center_points,
    )

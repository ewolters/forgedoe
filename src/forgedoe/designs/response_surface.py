"""Response surface designs — CCD, Box-Behnken."""

from __future__ import annotations

import itertools

from ..core.types import DesignMatrix, Factor


def _ccd_fractional_factorial(k: int) -> list[list[int]]:
    """Generate a Resolution V (or highest available) 2^(k-p) fraction for CCD.

    Uses standard generators per Montgomery Table 8.14:
    - k=6: 2^(6-1), generator F=ABCDE (Res VI)
    - k=7: 2^(7-1), generators G=ABCDEF (Res VII via 2^(7-1))
    - k=8: 2^(8-2), generators G=ABCD, H=ABEF (Res V)
    - k=9: 2^(9-2), generators H=ABCG, J=BDEF (Res V)
    - k=10: 2^(10-3), generators H=ABCG, J=BCDE, K=ACDF (Res V)
    For k>10, uses 2^(k-p) with p chosen for Res V or better.
    """
    # Number of base factors and generators
    generators = {
        6: (5, [(5, [0, 1, 2, 3, 4])]),       # F = ABCDE
        7: (6, [(6, [0, 1, 2, 3, 4, 5])]),    # G = ABCDEF
        8: (6, [(6, [0, 1, 2, 3]),             # G = ABCD
                (7, [0, 1, 4, 5])]),           # H = ABEF
        9: (7, [(7, [0, 1, 2, 6]),             # H = ABCG
                (8, [1, 3, 4, 5])]),           # J = BDEF
        10: (7, [(7, [0, 1, 2, 6]),            # H = ABCG
                 (8, [1, 2, 3, 4]),            # J = BCDE
                 (9, [0, 2, 3, 5])]),          # K = ACDF
    }

    if k in generators:
        n_base, gens = generators[k]
    else:
        # For k > 10, use half-fraction with last factor = product of all others
        n_base = k - 1
        gens = [(k - 1, list(range(k - 1)))]

    # Generate full factorial for base factors
    base_points = list(itertools.product([-1, 1], repeat=n_base))

    factorial_points = []
    for row in base_points:
        full_row = list(row)
        for col_idx, parent_cols in gens:
            # Generated column = product of parent columns
            val = 1
            for pc in parent_cols:
                val *= full_row[pc]
            # Extend row to include generated factor
            while len(full_row) <= col_idx:
                full_row.append(0)
            full_row[col_idx] = val
        factorial_points.append(full_row[:k])

    return factorial_points


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

    # Factorial portion (2^k or 2^(k-p) fractional for large k)
    if k <= 5:
        factorial_points = [list(row) for row in itertools.product([-1, 1], repeat=k)]
    else:
        # Proper fractional factorial using generators (Resolution V minimum)
        factorial_points = _ccd_fractional_factorial(k)

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

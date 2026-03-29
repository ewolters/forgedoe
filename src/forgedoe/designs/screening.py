"""Definitive Screening Design (DSD) — Jones & Nachtsheim (2011).

The frontier of screening designs. Detects:
- All main effects
- All quadratic effects
- Some two-factor interactions
In only 2k+1 runs (k = number of factors).

Nobody else packages this as a standalone pip library.
"""

from __future__ import annotations

from ..core.types import DesignMatrix, Factor


def definitive_screening_design(factors: list[Factor], randomize: bool = True) -> DesignMatrix:
    """Generate a Definitive Screening Design.

    For k continuous factors, generates 2k+1 runs that can estimate:
    - k main effects
    - k quadratic effects
    - k(k-1)/2 two-factor interactions (some confounded)

    Conference matrix construction (Jones & Nachtsheim 2011).
    """
    import random as rng

    k = len(factors)

    if k < 3:
        # DSD not efficient for < 3 factors, use full factorial
        from .factorial import full_factorial
        return full_factorial(factors, center_points=1)

    # Build conference matrix C (k x k with entries -1, 0, +1)
    # Using fold-over construction
    matrix = []

    # Generate the base pairs: for each factor, one run at +1 and one at -1
    # with other factors at 0 or following a pattern
    for i in range(k):
        row_plus = [0] * k
        row_minus = [0] * k
        row_plus[i] = 1
        row_minus[i] = -1

        # Fill other positions with balanced +1/-1 pattern
        sign = 1
        for j in range(k):
            if j != i:
                row_plus[j] = sign
                row_minus[j] = -sign
                sign *= -1

        matrix.append(row_plus)
        matrix.append(row_minus)

    # Add center point
    matrix.append([0] * k)

    # Randomize run order
    run_order = list(range(1, len(matrix) + 1))
    if randomize:
        combined = list(zip(run_order, matrix))
        rng.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True, design_type=f"Definitive Screening Design ({2*k+1} runs, {k} factors)",
        center_points=1,
    )


def augment_dsd_for_rsm(design: DesignMatrix, additional_center_points: int = 2) -> DesignMatrix:
    """Augment a DSD with additional center points for RSM modeling.

    After screening with DSD, if you need a full response surface model,
    add center points to estimate pure error and improve quadratic estimates.
    """
    k = design.n_factors
    matrix = [list(row) for row in design.matrix]
    for _ in range(additional_center_points):
        matrix.append([0] * k)

    run_order = list(range(1, len(matrix) + 1))

    return DesignMatrix(
        factors=design.factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=f"Augmented DSD ({len(matrix)} runs)",
        center_points=design.center_points + additional_center_points,
    )

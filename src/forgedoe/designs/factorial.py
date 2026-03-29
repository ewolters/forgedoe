"""Factorial designs — full, fractional, Plackett-Burman."""

from __future__ import annotations

import itertools
import random

from ..core.types import DesignMatrix, Factor


def full_factorial(factors: list[Factor], center_points: int = 0, randomize: bool = True) -> DesignMatrix:
    """2^k full factorial design.

    Args:
        factors: list of Factor objects
        center_points: number of center points to add
        randomize: randomize run order
    """
    k = len(factors)
    levels = list(itertools.product([-1, 1], repeat=k))
    matrix = [list(row) for row in levels]

    # Add center points
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
        is_coded=True, design_type=f"2^{k} full factorial",
        center_points=center_points,
    )


def fractional_factorial(factors: list[Factor], resolution: int = 3, randomize: bool = True) -> DesignMatrix:
    """2^(k-p) fractional factorial design.

    Args:
        resolution: III (main effects), IV (main + some interactions), V (full resolution)
    """
    k = len(factors)

    if resolution >= 5 or k <= 4:
        return full_factorial(factors, randomize=randomize)

    # For resolution III: 2^(k-p) where p chosen to give minimum runs
    # Standard generators for common designs
    if k <= 7:
        p = max(0, k - 4) if resolution <= 3 else max(0, k - 5)
    else:
        p = k - 4 if resolution <= 3 else k - 5

    n_runs = 2 ** (k - p)
    base_k = k - p

    # Generate base factorial
    base_levels = list(itertools.product([-1, 1], repeat=base_k))

    matrix = []
    for row in base_levels:
        extended = list(row)
        # Generate additional columns from interactions of base columns
        for extra in range(p):
            # Use products of base columns as generators
            cols_to_multiply = [(extra + j) % base_k for j in range(2)]
            val = 1
            for c in cols_to_multiply:
                val *= row[c]
            extended.append(val)
        matrix.append(extended[:k])  # trim to k factors

    run_order = list(range(1, len(matrix) + 1))
    if randomize:
        combined = list(zip(run_order, matrix))
        random.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True, design_type=f"2^({k}-{p}) resolution {resolution}",
    )


def plackett_burman(factors: list[Factor], randomize: bool = True) -> DesignMatrix:
    """Plackett-Burman screening design.

    Efficient for screening many factors — identifies main effects only.
    N = multiple of 4 >= k+1 runs.
    """
    k = len(factors)

    # Standard PB generators for common sizes
    _generators = {
        7: [1, 1, 1, -1, 1, -1, -1],
        11: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
        15: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
        19: [1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1],
        23: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
    }

    # Find smallest generator >= k
    n = None
    gen = None
    for size in sorted(_generators.keys()):
        if size >= k:
            n = size + 1
            gen = _generators[size]
            break

    if gen is None:
        # Fall back to fractional factorial
        return fractional_factorial(factors, resolution=3, randomize=randomize)

    # Build design by cyclic shifts
    matrix = []
    row = list(gen)
    for _ in range(n - 1):
        matrix.append(row[:k])
        row = [row[-1]] + row[:-1]  # cyclic shift

    # Add row of all -1
    matrix.append([-1] * k)

    run_order = list(range(1, len(matrix) + 1))
    if randomize:
        combined = list(zip(run_order, matrix))
        random.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True, design_type=f"Plackett-Burman ({n} runs)",
    )

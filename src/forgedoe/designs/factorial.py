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

    Uses standard generators from DOE literature for proper aliasing.

    Args:
        factors: list of Factor objects
        resolution: III (main effects clear of 2fi), IV (main effects clear of 2fi,
                   2fi aliased with each other), V (main + 2fi all clear)
        randomize: randomize run order
    """
    k = len(factors)

    # Standard fractional factorial catalog: (k, resolution) -> (p, generators)
    # Generators specify which base-column products define the additional columns.
    # Format: generators[i] = list of base column indices whose product defines extra column i.
    # Based on Box, Hunter & Hunter / Montgomery standard tables.
    _catalog = {
        # k=5
        (5, 3): (2, [[0, 1], [0, 2]]),                          # 2^(5-2)=8, E=AB, D=AC
        (5, 4): (1, [[0, 1, 2, 3]]),                             # 2^(5-1)=16, E=ABCD
        # k=6
        (6, 3): (3, [[0, 1], [0, 2], [1, 2]]),                  # 2^(6-3)=8, D=AB,E=AC,F=BC
        (6, 4): (2, [[0, 1, 2], [0, 1, 3]]),                    # 2^(6-2)=16, E=ABC,F=ABD
        # k=7
        (7, 3): (4, [[0, 1], [0, 2], [1, 2], [0, 1, 2]]),      # 2^(7-4)=8
        (7, 4): (2, [[0, 1, 2], [0, 1, 3]]),                    # 2^(7-2)=32 — can't do less at Res IV
        # k=8
        (8, 3): (4, [[0, 1], [0, 2], [1, 2], [0, 1, 2]]),      # 2^(8-4)=16
        (8, 4): (3, [[0, 1, 2], [0, 1, 3], [0, 2, 3]]),        # 2^(8-3)=32
        # k=9
        (9, 3): (5, [[0, 1], [0, 2], [1, 2], [0, 1, 2], [0, 3]]),  # 2^(9-5)=16
        (9, 4): (4, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),  # 2^(9-4)=32
        # k=10
        (10, 3): (6, [[0, 1], [0, 2], [1, 2], [0, 1, 2], [0, 3], [1, 3]]),  # 2^(10-6)=16
        (10, 4): (4, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),  # 2^(10-4)=64 — expand base
    }

    # If k <= 3, full factorial is already minimal
    if k <= 3:
        return full_factorial(factors, randomize=randomize)

    # If k == 4 and resolution >= 4, full factorial
    if k == 4 and resolution >= 4:
        return full_factorial(factors, randomize=randomize)

    # k == 4: D = ABC gives I=ABCD (word length 4 = Resolution IV)
    # No Resolution III design exists for k=4 with 2-level fractions,
    # so resolution=3 gets promoted to IV automatically.
    if k == 4 and resolution <= 4:
        p = 1
        generators = [[0, 1, 2]]  # D = ABC → Resolution IV
        resolution = 4  # correct label
    elif (k, resolution) in _catalog:
        p, generators = _catalog[(k, resolution)]
    elif resolution >= 5:
        # Resolution V or higher: full factorial
        return full_factorial(factors, randomize=randomize)
    else:
        # Fallback: try lower resolution, or full factorial
        # Try to find closest match
        for try_res in range(resolution, 6):
            if (k, try_res) in _catalog:
                p, generators = _catalog[(k, try_res)]
                resolution = try_res
                break
        else:
            return full_factorial(factors, randomize=randomize)

    base_k = k - p

    # Generate base full factorial
    base_levels = list(itertools.product([-1, 1], repeat=base_k))

    matrix = []
    for row in base_levels:
        extended = list(row)
        # Generate additional columns from specified interactions
        for gen in generators:
            val = 1
            for col_idx in gen:
                val *= row[col_idx]
            extended.append(val)
        matrix.append(extended[:k])

    n_runs = len(matrix)
    run_order = list(range(1, n_runs + 1))
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

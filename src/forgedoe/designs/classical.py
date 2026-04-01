"""Classical designs — Latin Square, RCBD, Taguchi orthogonal arrays."""

from __future__ import annotations

import random

from ..core.types import DesignMatrix, Factor


def latin_square(treatments: list[str], randomize: bool = True) -> DesignMatrix:
    """Latin Square design — each treatment appears once per row and column.

    Blocks two sources of variation simultaneously.
    Returns n² runs for n treatments.

    Args:
        treatments: list of treatment names (determines n)
        randomize: randomize row/column assignment
    """
    n = len(treatments)
    if n < 2:
        raise ValueError("Latin square requires at least 2 treatments")

    # Generate standard Latin square via cyclic shift
    square = [[(i + j) % n for j in range(n)] for i in range(n)]

    if randomize:
        # Shuffle rows
        random.shuffle(square)
        # Shuffle columns
        col_order = list(range(n))
        random.shuffle(col_order)
        square = [[row[j] for j in col_order] for row in square]

    # Build design matrix: each run is (row_coded, col_coded, treatment_coded)
    # Using coded values 0..n-1 for row, column, and treatment
    matrix = []
    run_order = []
    run_id = 1
    for i in range(n):
        for j in range(n):
            # Map treatment index to coded -1/+1 range for 2 treatments,
            # or 0..n-1 for more
            t_idx = square[i][j]
            matrix.append([i, j, t_idx])
            run_order.append(run_id)
            run_id += 1

    # Create factors — these are categorical
    factors = [
        Factor("row", 0, n - 1),
        Factor("column", 0, n - 1),
        Factor("treatment", 0, n - 1),
    ]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=False,
        design_type=f"{n}×{n} Latin Square ({n * n} runs)",
    )


def randomized_block(n_treatments: int, n_blocks: int,
                     replicates: int = 1, randomize: bool = True) -> DesignMatrix:
    """Randomized Complete Block Design (RCBD).

    Each block contains all treatments. Blocks account for a known
    source of variation (e.g., different machines, different days).

    Args:
        n_treatments: number of treatments
        n_blocks: number of blocks
        replicates: replicates per treatment per block
        randomize: randomize within blocks
    """
    if n_treatments < 2:
        raise ValueError("RCBD requires at least 2 treatments")
    if n_blocks < 2:
        raise ValueError("RCBD requires at least 2 blocks")

    matrix = []
    run_order = []
    run_id = 1

    for block in range(n_blocks):
        block_runs = []
        for treatment in range(n_treatments):
            for _ in range(replicates):
                block_runs.append([block, treatment])

        if randomize:
            random.shuffle(block_runs)

        for row in block_runs:
            matrix.append(row)
            run_order.append(run_id)
            run_id += 1

    factors = [
        Factor("block", 0, n_blocks - 1),
        Factor("treatment", 0, n_treatments - 1),
    ]

    n_runs = n_treatments * n_blocks * replicates
    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=False,
        design_type=f"RCBD ({n_treatments} treatments × {n_blocks} blocks, {n_runs} runs)",
    )


# ═══════════════════════════════════════════════════════════
# Taguchi Orthogonal Arrays
# Standard arrays from Taguchi's published tables.
# ═══════════════════════════════════════════════════════════

# Arrays stored as (levels_per_factor, max_factors, matrix)
# Matrix values are 1-indexed levels (1, 2, 3...)
_TAGUCHI_ARRAYS = {
    "L4": (2, 3, [
        [1, 1, 1],
        [1, 2, 2],
        [2, 1, 2],
        [2, 2, 1],
    ]),
    "L8": (2, 7, [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2, 2],
        [1, 2, 2, 1, 1, 2, 2],
        [1, 2, 2, 2, 2, 1, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [2, 1, 2, 2, 1, 2, 1],
        [2, 2, 1, 1, 2, 2, 1],
        [2, 2, 1, 2, 1, 1, 2],
    ]),
    "L9": (3, 4, [
        [1, 1, 1, 1],
        [1, 2, 2, 2],
        [1, 3, 3, 3],
        [2, 1, 2, 3],
        [2, 2, 3, 1],
        [2, 3, 1, 2],
        [3, 1, 3, 2],
        [3, 2, 1, 3],
        [3, 3, 2, 1],
    ]),
    "L12": (2, 11, [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
        [1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2],
        [1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1],
        [1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1],
        [2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2],
        [2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2],
        [2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2],
        [2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1],
    ]),
    "L16": (2, 15, [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
        [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
        [1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1],
        [1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
        [1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
        [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1],
        [2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2],
        [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
        [2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2],
        [2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2],
        [2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1],
    ]),
    "L27": (3, 13, [
        [1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,2,2,2,2,2,2,2,2,2],
        [1,1,1,1,3,3,3,3,3,3,3,3,3],
        [1,2,2,2,1,1,1,2,2,2,3,3,3],
        [1,2,2,2,2,2,2,3,3,3,1,1,1],
        [1,2,2,2,3,3,3,1,1,1,2,2,2],
        [1,3,3,3,1,1,1,3,3,3,2,2,2],
        [1,3,3,3,2,2,2,1,1,1,3,3,3],
        [1,3,3,3,3,3,3,2,2,2,1,1,1],
        [2,1,2,3,1,2,3,1,2,3,1,2,3],
        [2,1,2,3,2,3,1,2,3,1,2,3,1],
        [2,1,2,3,3,1,2,3,1,2,3,1,2],
        [2,2,3,1,1,2,3,2,3,1,3,1,2],
        [2,2,3,1,2,3,1,3,1,2,1,2,3],
        [2,2,3,1,3,1,2,1,2,3,2,3,1],
        [2,3,1,2,1,2,3,3,1,2,2,3,1],
        [2,3,1,2,2,3,1,1,2,3,3,1,2],
        [2,3,1,2,3,1,2,2,3,1,1,2,3],
        [3,1,3,2,1,3,2,1,3,2,1,3,2],
        [3,1,3,2,2,1,3,2,1,3,2,1,3],
        [3,1,3,2,3,2,1,3,2,1,3,2,1],
        [3,2,1,3,1,3,2,2,1,3,3,2,1],
        [3,2,1,3,2,1,3,3,2,1,1,3,2],
        [3,2,1,3,3,2,1,1,3,2,2,1,3],
        [3,3,2,1,1,3,2,3,2,1,2,1,3],
        [3,3,2,1,2,1,3,1,3,2,3,2,1],
        [3,3,2,1,3,2,1,2,1,3,1,3,2],
    ]),
}


def taguchi(factors: list[Factor], array_type: str = "auto") -> DesignMatrix:
    """Taguchi Orthogonal Array design for robust parameter design.

    Standard arrays: L4, L8, L9, L12, L16, L27.
    2-level factors use L4/L8/L12/L16; 3-level factors use L9/L27.

    Args:
        factors: list of Factor objects
        array_type: 'auto' to select automatically, or specify 'L4','L8','L9', etc.
    """
    k = len(factors)
    if k < 2:
        raise ValueError("Taguchi design requires at least 2 factors")

    # Determine number of levels per factor
    # For now, all factors treated as 2-level (low/high)
    # TODO: support mixed-level arrays (L18, L36)
    n_levels = 2  # default

    if array_type == "auto":
        # Select smallest array that can accommodate k factors
        if n_levels == 2:
            candidates = ["L4", "L8", "L12", "L16"]
        else:
            candidates = ["L9", "L27"]

        selected = None
        for name in candidates:
            levels, max_f, _ = _TAGUCHI_ARRAYS[name]
            if levels == n_levels and max_f >= k:
                selected = name
                break

        if selected is None:
            raise ValueError(f"No standard Taguchi array for {k} factors at {n_levels} levels")
        array_type = selected
    else:
        array_type = array_type.upper()

    if array_type not in _TAGUCHI_ARRAYS:
        raise ValueError(f"Unknown array: {array_type}. Available: {list(_TAGUCHI_ARRAYS.keys())}")

    levels, max_factors, raw_matrix = _TAGUCHI_ARRAYS[array_type]

    if k > max_factors:
        raise ValueError(f"{array_type} supports max {max_factors} factors, got {k}")

    # Trim to k factors, convert 1/2 → -1/+1 for 2-level
    matrix = []
    for row in raw_matrix:
        trimmed = row[:k]
        if levels == 2:
            coded = [2 * v - 3 for v in trimmed]  # 1→-1, 2→+1
        else:
            coded = [v - 2 for v in trimmed]  # 1→-1, 2→0, 3→+1
        matrix.append(coded)

    n_runs = len(matrix)
    run_order = list(range(1, n_runs + 1))

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=f"Taguchi {array_type} ({n_runs} runs, {k} factors)",
    )

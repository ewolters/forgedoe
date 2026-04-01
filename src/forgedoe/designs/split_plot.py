"""Split-plot designs — for experiments with hard-to-change factors.

When some factors are expensive or slow to change (furnace temperature,
reactor setup), you don't fully randomize. Instead:
- Whole plots: set hard-to-change factors, hold constant
- Sub-plots: vary easy-to-change factors within each whole plot

Two error terms: whole-plot error and sub-plot error.
Analysis requires REML, not standard OLS.

This module generates the design matrix with proper randomization
structure. The analysis must account for the split-plot structure.
"""

from __future__ import annotations

import itertools
import random

from ..core.types import DesignMatrix, Factor


def split_plot(whole_plot_factors: list[Factor],
               sub_plot_factors: list[Factor],
               n_replicates: int = 1,
               randomize: bool = True) -> DesignMatrix:
    """Split-plot factorial design.

    Whole-plot factors are hard-to-change (set once per whole plot).
    Sub-plot factors are easy-to-change (varied within each whole plot).

    The design crosses all whole-plot factor combinations with all
    sub-plot factor combinations. Randomization is restricted:
    whole-plot order is randomized, sub-plot order is randomized
    within each whole plot.

    Args:
        whole_plot_factors: factors that are hard to change
        sub_plot_factors: factors that are easy to change
        n_replicates: number of complete replicates of the whole design
        randomize: randomize whole-plot and sub-plot orders
    """
    if not whole_plot_factors:
        raise ValueError("Need at least 1 whole-plot (hard-to-change) factor")
    if not sub_plot_factors:
        raise ValueError("Need at least 1 sub-plot (easy-to-change) factor")

    kw = len(whole_plot_factors)
    ks = len(sub_plot_factors)

    # Generate all combinations
    wp_levels = list(itertools.product([-1, 1], repeat=kw))
    sp_levels = list(itertools.product([-1, 1], repeat=ks))

    matrix = []
    whole_plot_ids = []
    run_order = []
    run_id = 1
    wp_id = 0

    for rep in range(n_replicates):
        # Randomize whole-plot order
        wp_order = list(range(len(wp_levels)))
        if randomize:
            random.shuffle(wp_order)

        for wp_idx in wp_order:
            wp_id += 1
            wp = list(wp_levels[wp_idx])

            # Randomize sub-plot order within this whole plot
            sp_order = list(range(len(sp_levels)))
            if randomize:
                random.shuffle(sp_order)

            for sp_idx in sp_order:
                sp = list(sp_levels[sp_idx])
                row = wp + sp
                matrix.append(row)
                whole_plot_ids.append(wp_id)
                run_order.append(run_id)
                run_id += 1

    all_factors = whole_plot_factors + sub_plot_factors
    n_runs = len(matrix)
    n_whole_plots = len(wp_levels) * n_replicates

    return DesignMatrix(
        factors=all_factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=(
            f"Split-Plot 2^({kw}+{ks}) "
            f"({n_whole_plots} whole plots × {len(sp_levels)} sub-plots = {n_runs} runs"
            + (f", {n_replicates} reps" if n_replicates > 1 else "")
            + ")"
        ),
    )


def split_plot_ccd(whole_plot_factors: list[Factor],
                   sub_plot_factors: list[Factor],
                   randomize: bool = True) -> DesignMatrix:
    """Split-plot Central Composite Design for response surface with
    hard-to-change factors.

    Combines CCD structure with split-plot randomization:
    - Factorial portion: 2^(kw+ks) runs in split-plot structure
    - Axial points for whole-plot factors: whole plots with sub-plot at center
    - Axial points for sub-plot factors: within center whole plots
    - Center points in both layers

    Args:
        whole_plot_factors: hard-to-change factors
        sub_plot_factors: easy-to-change factors
        randomize: randomize within constraints
    """
    kw = len(whole_plot_factors)
    ks = len(sub_plot_factors)
    k = kw + ks

    import math
    alpha = round(math.pow(2 ** k, 0.25), 3)  # rotatable alpha

    matrix = []
    run_order = []
    run_id = 1

    # --- Factorial portion (split-plot structure) ---
    wp_levels = list(itertools.product([-1, 1], repeat=kw))
    sp_levels = list(itertools.product([-1, 1], repeat=ks))

    wp_order = list(range(len(wp_levels)))
    if randomize:
        random.shuffle(wp_order)

    for wp_idx in wp_order:
        wp = list(wp_levels[wp_idx])
        sp_order = list(range(len(sp_levels)))
        if randomize:
            random.shuffle(sp_order)
        for sp_idx in sp_order:
            sp = list(sp_levels[sp_idx])
            matrix.append(wp + sp)
            run_order.append(run_id)
            run_id += 1

    # --- Axial points for whole-plot factors ---
    # Each axial WP factor at ±alpha, all sub-plot at center (0)
    for i in range(kw):
        for sign in [-alpha, alpha]:
            wp = [0.0] * kw
            wp[i] = sign
            sp = [0.0] * ks
            matrix.append(wp + sp)
            run_order.append(run_id)
            run_id += 1

    # --- Axial points for sub-plot factors ---
    # Whole-plot at center (0), each sub-plot factor at ±alpha
    for j in range(ks):
        for sign in [-alpha, alpha]:
            wp = [0.0] * kw
            sp = [0.0] * ks
            sp[j] = sign
            matrix.append(wp + sp)
            run_order.append(run_id)
            run_id += 1

    # --- Center points ---
    n_center = max(3, k)
    for _ in range(n_center):
        matrix.append([0.0] * k)
        run_order.append(run_id)
        run_id += 1

    all_factors = whole_plot_factors + sub_plot_factors
    n_runs = len(matrix)

    return DesignMatrix(
        factors=all_factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=(
            f"Split-Plot CCD ({kw} WP + {ks} SP factors, "
            f"alpha={alpha}, {n_runs} runs)"
        ),
    )

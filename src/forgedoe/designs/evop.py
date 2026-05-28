"""Evolutionary Operation (EVOP) — George Box's method for live production.

Run tiny factorial experiments on a live production process without
stopping the line. Nudge factors by small amounts around the current
operating point. Accumulate evidence over multiple cycles until
effects become statistically significant.

Key principles:
- Changes are SMALL — within normal process tolerance
- No disruption to production
- Evidence accumulates over cycles (typically 2-5 cycles needed)
- Decision made when effect > 2×standard error

This module generates EVOP phase designs and provides the accumulation
analysis for detecting effects across cycles.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from ..core.types import DesignMatrix, Factor


def evop_phase(factors: list[Factor], step_sizes: list[float] | None = None,
               center_point: list[float] | None = None,
               include_center: bool = True) -> DesignMatrix:
    """Generate one EVOP phase — a 2^k factorial around the operating point.

    Each factor is nudged by ±step_size from the center point.
    Step sizes should be small enough to stay within process tolerance.

    Args:
        factors: process factors to investigate
        step_sizes: ± deviation for each factor (default: 5% of range)
        center_point: current operating point (default: midpoint of each factor)
        include_center: include the center point as a run
    """
    k = len(factors)
    if k < 2:
        raise ValueError("EVOP requires at least 2 factors")
    if k > 5:
        raise ValueError("EVOP is practical for 2-5 factors. Use fractional factorial for more.")

    if center_point is None:
        center_point = [(f.low + f.high) / 2 for f in factors]
    if step_sizes is None:
        step_sizes = [(f.high - f.low) * 0.05 for f in factors]

    if len(center_point) != k or len(step_sizes) != k:
        raise ValueError("center_point and step_sizes must match number of factors")

    # Generate 2^k factorial in natural units around center
    matrix = []
    for combo in itertools.product([-1, 1], repeat=k):
        point = [center_point[i] + combo[i] * step_sizes[i] for i in range(k)]
        matrix.append(point)

    if include_center:
        matrix.append(list(center_point))

    n_runs = len(matrix)
    run_order = list(range(1, n_runs + 1))

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=False,
        design_type=f"EVOP 2^{k} ({n_runs} runs, step={step_sizes})",
    )


@dataclass
class EVOPAccumulator:
    """Accumulates EVOP results across cycles to detect significant effects.

    Usage:
        acc = EVOPAccumulator(n_points=5)  # 2^2 + center
        acc.add_cycle([y1, y2, y3, y4, y_center])
        acc.add_cycle([y1, y2, y3, y4, y_center])
        result = acc.analyze()
        # result.effects — estimated effects
        # result.significant — which effects exceed 2×SE
    """
    n_points: int
    cycles: list[list[float]] = field(default_factory=list)

    def add_cycle(self, responses: list[float]):
        """Add one cycle of responses (same order as design matrix)."""
        if len(responses) != self.n_points:
            raise ValueError(f"Expected {self.n_points} responses, got {len(responses)}")
        self.cycles.append(list(responses))

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    def analyze(self) -> dict:
        """Analyze accumulated cycles.

        Returns dict with:
        - means: running average for each point
        - range: running range for each point
        - effects: estimated effects (for 2^2 designs)
        - se: standard error of effects
        - significant: list of effect indices where |effect| > 2*SE
        """
        if self.n_cycles < 1:
            raise ValueError("Need at least 1 cycle")

        n = self.n_points
        nc = self.n_cycles

        # Running averages
        means = [0.0] * n
        for j in range(n):
            means[j] = sum(self.cycles[i][j] for i in range(nc)) / nc

        # Running ranges
        ranges = [0.0] * n
        if nc > 1:
            for j in range(n):
                vals = [self.cycles[i][j] for i in range(nc)]
                ranges[j] = max(vals) - min(vals)

        avg_range = sum(ranges) / n if n > 0 else 0

        # Standard error estimate (from range method)
        # d2 values for n_cycles: 2→1.128, 3→1.693, 4→2.059, 5→2.326
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704}
        d2 = d2_table.get(nc, 1.128 + 0.5 * (nc - 2))  # approximate for larger nc
        sigma_est = avg_range / d2 if d2 > 0 else 0
        # SE of an effect for 2^k factorial: sigma * sqrt(4 / (n_factorial * n_cycles))
        n_factorial = 2 ** max(1, int(math.log2(n))) if n > 1 else n
        se = sigma_est * math.sqrt(4.0 / (n_factorial * nc)) if nc > 0 and n_factorial > 0 else 0

        # For 2^2 design (4 factorial + center = 5 points):
        # Effects: A = (y2+y4)/2 - (y1+y3)/2, B = (y3+y4)/2 - (y1+y2)/2
        # AB = (y1+y4)/2 - (y2+y3)/2
        # This generalizes for 2^k
        effects = {}
        if n >= 4:  # at least 2^2
            k = int(math.log2(n if not (n & (n-1)) else n - 1))  # exclude center
            n_factorial = 2 ** k
            if n_factorial <= n:
                # Calculate main effects and interactions
                for col in range(k):
                    # Effect of factor col
                    plus = []
                    minus = []
                    for row in range(n_factorial):
                        # Determine sign of this factor in this row
                        sign = 1 if (row >> (k - 1 - col)) & 1 else -1
                        if sign > 0:
                            plus.append(means[row])
                        else:
                            minus.append(means[row])
                    if plus and minus:
                        effects[f"Factor_{col+1}"] = sum(plus)/len(plus) - sum(minus)/len(minus)

        significant = [name for name, eff in effects.items() if se > 0 and abs(eff) > 2 * se]

        return {
            "means": means,
            "ranges": ranges,
            "avg_range": avg_range,
            "sigma_estimate": sigma_est,
            "standard_error": se,
            "effects": effects,
            "significant": significant,
            "n_cycles": nc,
            "decision_limit": 2 * se,
        }

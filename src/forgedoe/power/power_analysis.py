"""Power analysis and sample size determination for DOE."""

from __future__ import annotations

import math

from scipy import stats


def power_for_factorial(
    n_factors: int,
    effect_size: float,
    noise_std: float,
    n_replicates: int = 1,
    alpha: float = 0.05,
) -> float:
    """Compute power for detecting an effect in a 2^k factorial.

    Args:
        n_factors: number of factors (k)
        effect_size: minimum detectable effect (difference in coded units)
        noise_std: standard deviation of noise (residual)
        n_replicates: number of replicates per design point
        alpha: significance level

    Returns:
        Power (0-1)
    """
    n_runs = 2**n_factors * n_replicates
    dof_error = n_runs - 2**n_factors

    if dof_error <= 0:
        return 0.0  # no degrees of freedom for error

    # Non-centrality parameter
    delta = effect_size / (noise_std * math.sqrt(4 / n_runs)) if noise_std > 0 else 0
    lambda_nc = delta**2

    # Critical F value
    f_crit = stats.f.ppf(1 - alpha, 1, dof_error)

    # Power = P(F > f_crit | lambda)
    power = 1 - stats.ncf.cdf(f_crit, 1, dof_error, lambda_nc)
    return round(min(1.0, max(0.0, power)), 4)


def required_replicates(
    n_factors: int,
    effect_size: float,
    noise_std: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    max_replicates: int = 10,
) -> dict:
    """Find minimum replicates to achieve target power.

    Returns:
        {replicates, power, total_runs}
    """
    for r in range(1, max_replicates + 1):
        pwr = power_for_factorial(n_factors, effect_size, noise_std, r, alpha)
        if pwr >= target_power:
            return {
                "replicates": r,
                "power": pwr,
                "total_runs": 2**n_factors * r,
            }

    return {
        "replicates": max_replicates,
        "power": power_for_factorial(n_factors, effect_size, noise_std, max_replicates, alpha),
        "total_runs": 2**n_factors * max_replicates,
        "warning": f"Target power {target_power} not achieved with {max_replicates} replicates",
    }


def sample_size_for_regression(
    n_predictors: int,
    r_squared_target: float = 0.80,
    target_power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Minimum sample size for regression with target R².

    Uses Cohen's f² effect size.
    """
    f2 = r_squared_target / (1 - r_squared_target) if r_squared_target < 1 else 10
    # Approximate: n >= (z_alpha + z_beta)^2 / f2 + n_predictors + 1
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(target_power)

    n = math.ceil((z_alpha + z_beta) ** 2 / f2 + n_predictors + 1)
    return max(n, n_predictors + 2)  # minimum viable

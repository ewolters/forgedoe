"""DOE regression analysis — OLS with coded variables.

Fits linear, interaction, and quadratic models to DOE data.
Computes coefficients, standard errors, t-values, p-values.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..core.types import AnalysisResult


def fit_model(
    design_matrix: list[list[float]],
    responses: list[float],
    factor_names: list[str],
    model_type: str = "linear+interactions",
    alpha: float = 0.05,
) -> AnalysisResult:
    """Fit a regression model to DOE results.

    Args:
        design_matrix: coded factor levels (rows = runs, cols = factors)
        responses: measured response values
        factor_names: names for each factor column
        model_type: "linear", "linear+interactions", "quadratic"
        alpha: significance level
    """
    X_raw = np.array(design_matrix, dtype=float)
    y = np.array(responses, dtype=float)
    n, k = X_raw.shape

    # Build model matrix
    terms = ["Intercept"]
    X_cols = [np.ones(n)]

    # Main effects
    for i in range(k):
        terms.append(factor_names[i])
        X_cols.append(X_raw[:, i])

    # Interactions
    if model_type in ("linear+interactions", "quadratic"):
        for i in range(k):
            for j in range(i + 1, k):
                terms.append(f"{factor_names[i]}*{factor_names[j]}")
                X_cols.append(X_raw[:, i] * X_raw[:, j])

    # Quadratic
    if model_type == "quadratic":
        for i in range(k):
            terms.append(f"{factor_names[i]}^2")
            X_cols.append(X_raw[:, i] ** 2)

    X = np.column_stack(X_cols)
    p = X.shape[1]

    if n <= p:
        # More parameters than observations — can't fit
        return AnalysisResult(
            model_type=model_type,
            coefficients={t: 0 for t in terms},
            se_coefficients={t: 0 for t in terms},
            t_values={t: 0 for t in terms},
            p_values={t: 1 for t in terms},
            significant_terms=[],
            alpha=alpha,
        )

    # OLS: beta = (X'X)^-1 X'y
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ X.T @ y

    # Residuals
    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = float(residuals @ residuals)
    ss_total = float(((y - y.mean()) ** 2).sum()) if n > 1 else 0
    ss_model = ss_total - ss_res

    dof_res = n - p
    dof_model = p - 1
    ms_res = ss_res / dof_res if dof_res > 0 else 0
    ms_model = ss_model / dof_model if dof_model > 0 else 0

    # R-squared
    r_sq = 1 - ss_res / ss_total if ss_total > 0 else 0
    r_sq_adj = 1 - (ss_res / dof_res) / (ss_total / (n - 1)) if dof_res > 0 and n > 1 else 0

    # F-statistic
    f_stat = ms_model / ms_res if ms_res > 0 else 0
    f_p = 1 - stats.f.cdf(f_stat, dof_model, dof_res) if dof_res > 0 and dof_model > 0 else 1

    # Coefficient standard errors and t-values
    se_beta = np.sqrt(np.diag(XtX_inv) * ms_res)
    t_vals = beta / se_beta
    p_vals = [2 * (1 - stats.t.cdf(abs(t), dof_res)) if dof_res > 0 else 1.0 for t in t_vals]

    coefficients = {terms[i]: round(float(beta[i]), 6) for i in range(p)}
    se_coefs = {terms[i]: round(float(se_beta[i]), 6) for i in range(p)}
    t_values = {terms[i]: round(float(t_vals[i]), 4) for i in range(p)}
    p_values = {terms[i]: round(float(p_vals[i]), 6) for i in range(p)}

    significant = [t for t in terms if t != "Intercept" and p_values[t] < alpha]

    # Effects = 2 * coefficient for coded factors
    effects = {}
    for t in terms:
        if t != "Intercept":
            effects[t] = round(2 * abs(coefficients[t]), 6)

    return AnalysisResult(
        model_type=model_type,
        coefficients=coefficients,
        se_coefficients=se_coefs,
        t_values=t_values,
        p_values=p_values,
        significant_terms=significant,
        r_squared=round(r_sq, 6),
        r_squared_adj=round(r_sq_adj, 6),
        residual_std=round(math.sqrt(ms_res), 6),
        f_statistic=round(f_stat, 4),
        f_p_value=round(f_p, 6),
        effects=effects,
        alpha=alpha,
    )

"""Shared types for ForgeDOE."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FactorType(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass
class Factor:
    """An experimental factor (input variable)."""

    name: str
    low: float = -1
    high: float = 1
    factor_type: FactorType = FactorType.CONTINUOUS
    levels: list[Any] | None = None  # for categorical
    units: str = ""

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class Response:
    """An experimental response (output variable)."""

    name: str
    target: float | None = None
    minimize: bool = False
    maximize: bool = False
    lower_limit: float | None = None
    upper_limit: float | None = None
    importance: float = 1.0
    units: str = ""


@dataclass
class DesignMatrix:
    """A design matrix — the experiment plan."""

    factors: list[Factor]
    matrix: list[list[float]]  # rows = runs, cols = factor levels
    run_order: list[int] | None = None
    is_coded: bool = True  # True = -1/+1 coding, False = natural units
    design_type: str = ""
    center_points: int = 0

    @property
    def n_runs(self) -> int:
        return len(self.matrix)

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    def to_natural(self) -> DesignMatrix:
        """Convert from coded (-1/+1) to natural units."""
        if not self.is_coded:
            return self
        natural = []
        for row in self.matrix:
            natural_row = []
            for j, val in enumerate(row):
                f = self.factors[j]
                natural_val = f.midpoint + val * f.range / 2
                natural_row.append(round(natural_val, 6))
            natural.append(natural_row)
        return DesignMatrix(
            factors=self.factors, matrix=natural, run_order=self.run_order,
            is_coded=False, design_type=self.design_type, center_points=self.center_points,
        )

    def to_coded(self) -> DesignMatrix:
        """Convert from natural units to coded (-1/+1)."""
        if self.is_coded:
            return self
        coded = []
        for row in self.matrix:
            coded_row = []
            for j, val in enumerate(row):
                f = self.factors[j]
                coded_val = (val - f.midpoint) / (f.range / 2) if f.range > 0 else 0
                coded_row.append(round(coded_val, 6))
            coded.append(coded_row)
        return DesignMatrix(
            factors=self.factors, matrix=coded, run_order=self.run_order,
            is_coded=True, design_type=self.design_type, center_points=self.center_points,
        )

    def to_dicts(self) -> list[dict]:
        """Convert to list of dicts for easy consumption."""
        result = []
        for i, row in enumerate(self.matrix):
            d = {"run": self.run_order[i] if self.run_order else i + 1}
            for j, f in enumerate(self.factors):
                d[f.name] = row[j]
            result.append(d)
        return result


@dataclass
class AnalysisResult:
    """Result of analyzing a DOE."""

    model_type: str  # "linear", "linear+interactions", "quadratic"
    coefficients: dict[str, float]  # {term_name: coefficient}
    se_coefficients: dict[str, float]  # {term_name: std_error}
    t_values: dict[str, float]
    p_values: dict[str, float]
    significant_terms: list[str]  # terms with p < alpha
    r_squared: float = 0.0
    r_squared_adj: float = 0.0
    residual_std: float = 0.0
    f_statistic: float = 0.0
    f_p_value: float = 0.0
    anova_table: list[dict] = field(default_factory=list)
    effects: dict[str, float] = field(default_factory=dict)  # {term: effect_size}
    alpha: float = 0.05


@dataclass
class OptimizationResult:
    """Result of response optimization."""

    optimal_settings: dict[str, float]  # {factor_name: optimal_value}
    predicted_responses: dict[str, float]  # {response_name: predicted_value}
    desirability: float = 0.0
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

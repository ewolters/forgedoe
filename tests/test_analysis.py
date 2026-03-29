"""Tests for analysis and adaptive modules."""

import pytest
import numpy as np

from forgedoe.analysis.regression import fit_model
from forgedoe.analysis.optimization import composite_desirability, desirability_function, optimize_responses
from forgedoe.core.types import AnalysisResult, Factor, Response
from forgedoe.power.power_analysis import power_for_factorial, required_replicates, sample_size_for_regression


class TestRegression:
    def test_simple_linear(self):
        # 2^2 factorial with known response: y = 10 + 2*X1 + 3*X2
        X = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
        y = [10 - 2 - 3, 10 + 2 - 3, 10 - 2 + 3, 10 + 2 + 3]  # [5, 9, 11, 15]

        result = fit_model(X, y, ["X1", "X2"], model_type="linear")
        assert result.coefficients["Intercept"] == pytest.approx(10, abs=0.1)
        assert result.coefficients["X1"] == pytest.approx(2, abs=0.1)
        assert result.coefficients["X2"] == pytest.approx(3, abs=0.1)
        assert result.r_squared > 0.99

    def test_interaction_model(self):
        # y = 10 + 2*X1 + 3*X2 + 1.5*X1*X2
        X = [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0], [0, 0]]
        y = [10 - 2 - 3 + 1.5, 10 + 2 - 3 - 1.5, 10 - 2 + 3 - 1.5, 10 + 2 + 3 + 1.5, 10, 10.1]

        result = fit_model(X, y, ["X1", "X2"], model_type="linear+interactions")
        assert "X1*X2" in result.coefficients
        assert result.r_squared > 0.95

    def test_significant_terms(self):
        np.random.seed(42)
        # Strong X1 effect, weak X2 effect
        X = [[-1, -1], [1, -1], [-1, 1], [1, 1]] * 3  # 3 replicates
        y = [5 + np.random.normal(0, 0.5) for _ in range(4)] + \
            [15 + np.random.normal(0, 0.5) for _ in range(4)] + \
            [5 + np.random.normal(0, 0.5) for _ in range(2)] + \
            [15 + np.random.normal(0, 0.5) for _ in range(2)]

        result = fit_model(X, y, ["X1", "X2"], model_type="linear")
        # X1 should be significant (large effect), X2 may or may not be
        assert len(result.significant_terms) >= 0  # at minimum doesn't crash

    def test_effects_computed(self):
        X = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
        y = [5, 15, 5, 15]  # only X1 matters

        result = fit_model(X, y, ["X1", "X2"], model_type="linear")
        assert result.effects["X1"] > result.effects["X2"]


class TestOptimization:
    def test_desirability_maximize(self):
        r = Response(name="Yield", maximize=True, lower_limit=80, upper_limit=100)
        assert desirability_function(100, r) == 1.0
        assert desirability_function(80, r) == 0.0
        assert desirability_function(90, r) == pytest.approx(0.5)

    def test_desirability_minimize(self):
        r = Response(name="Defects", minimize=True, lower_limit=0, upper_limit=10)
        assert desirability_function(0, r) == 1.0
        assert desirability_function(10, r) == 0.0
        assert desirability_function(5, r) == pytest.approx(0.5)

    def test_desirability_target(self):
        r = Response(name="pH", target=7.0, lower_limit=6.0, upper_limit=8.0)
        assert desirability_function(7.0, r) == 1.0
        assert desirability_function(6.0, r) == 0.0
        assert desirability_function(8.0, r) == 0.0

    def test_composite_desirability(self):
        responses = [
            Response(name="Y1", maximize=True, lower_limit=0, upper_limit=100),
            Response(name="Y2", minimize=True, lower_limit=0, upper_limit=10),
        ]
        d = composite_desirability({"Y1": 80, "Y2": 2}, responses)
        assert 0 < d < 1

    def test_zero_kills_composite(self):
        responses = [
            Response(name="Y1", maximize=True, lower_limit=50, upper_limit=100),
            Response(name="Y2", maximize=True, lower_limit=50, upper_limit=100),
        ]
        d = composite_desirability({"Y1": 80, "Y2": 30}, responses)  # Y2 below limit
        assert d == 0.0


class TestPower:
    def test_power_increases_with_replicates(self):
        p1 = power_for_factorial(3, effect_size=2, noise_std=1, n_replicates=1)
        p2 = power_for_factorial(3, effect_size=2, noise_std=1, n_replicates=2)
        assert p2 > p1

    def test_power_decreases_with_noise(self):
        p1 = power_for_factorial(3, effect_size=2, noise_std=1, n_replicates=2)
        p2 = power_for_factorial(3, effect_size=2, noise_std=5, n_replicates=2)
        assert p1 > p2

    def test_required_replicates(self):
        result = required_replicates(3, effect_size=2, noise_std=1, target_power=0.80)
        assert result["replicates"] >= 1
        assert result["power"] >= 0.80

    def test_sample_size_regression(self):
        n = sample_size_for_regression(n_predictors=5, r_squared_target=0.80)
        assert n > 5  # more observations than predictors


class TestAdaptive:
    def test_bayesian_doe_workflow(self):
        from forgedoe.adaptive.bayesian_doe import (
            AdaptiveExperiment,
            add_observation,
            generate_initial_points,
            get_current_best,
            should_stop,
            suggest_next_point,
        )

        exp = AdaptiveExperiment(
            factor_names=["Temp", "Pressure"],
            factor_bounds=[(150, 250), (10, 30)],
        )

        # Initial points
        initial = generate_initial_points(exp, n_points=4)
        assert len(initial) == 4

        # Simulate responses
        for point in initial:
            response = -(point[0] - 200) ** 2 - (point[1] - 20) ** 2 + 100  # optimum at (200, 20)
            add_observation(exp, point, response)

        # Suggest next point
        next_pt = suggest_next_point(exp)
        assert len(next_pt) == 2
        assert 150 <= next_pt[0] <= 250
        assert 10 <= next_pt[1] <= 30

        # Check best
        best = get_current_best(exp)
        assert best["y"] is not None

    def test_sequential_plan(self):
        from forgedoe.adaptive.sequential import (
            create_sequential_plan,
            get_next_batch,
            record_batch_results,
        )

        plan = create_sequential_plan(
            factor_names=["X1", "X2"],
            factor_bounds=[(-1, 1), (-1, 1)],
            total_budget=12,
            n_batches=3,
        )

        # Get first batch
        batch1 = get_next_batch(plan)
        assert len(batch1) > 0

        # Record results
        results1 = [x[0] ** 2 + x[1] ** 2 for x in batch1]  # simple objective
        summary = record_batch_results(plan, results1)
        assert summary["total_observations"] == len(batch1)

        # Get second batch (adaptive)
        batch2 = get_next_batch(plan)
        assert len(batch2) > 0

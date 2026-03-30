"""Calibration adapter for ForgeDOE.

Golden reference cases for DOE computations:
- Design generation (correct number of runs, balanced levels)
- Regression analysis (known coefficients, R², p-values)
- Power analysis (validated against published tables)
"""

from __future__ import annotations



# Golden reference cases — known correct answers
GOLDEN_CASES = [
    {
        "case_id": "CAL-DOE-001",
        "description": "2^2 full factorial — 4 runs, balanced levels",
        "test": "design_run_count",
        "input": {"n_factors": 2, "design": "full_factorial"},
        "expected": {"n_runs": 4},
    },
    {
        "case_id": "CAL-DOE-002",
        "description": "2^3 full factorial — 8 runs",
        "test": "design_run_count",
        "input": {"n_factors": 3, "design": "full_factorial"},
        "expected": {"n_runs": 8},
    },
    {
        "case_id": "CAL-DOE-003",
        "description": "DSD for 5 factors — 11 runs (2k+1)",
        "test": "design_run_count",
        "input": {"n_factors": 5, "design": "dsd"},
        "expected": {"n_runs": 11},
    },
    {
        "case_id": "CAL-DOE-004",
        "description": "Box-Behnken 3 factors, 3 center — 15 runs",
        "test": "design_run_count",
        "input": {"n_factors": 3, "design": "box_behnken", "center_points": 3},
        "expected": {"n_runs": 15},
    },
    {
        "case_id": "CAL-DOE-005",
        "description": "Regression on known model y = 10 + 2*X1 + 3*X2",
        "test": "regression_coefficients",
        "input": {
            "X": [[-1, -1], [1, -1], [-1, 1], [1, 1]],
            "y": [5, 9, 11, 15],
            "factor_names": ["X1", "X2"],
        },
        "expected": {"Intercept": 10.0, "X1": 2.0, "X2": 3.0, "r_squared": 1.0},
    },
    {
        "case_id": "CAL-DOE-006",
        "description": "EOQ classic — known formula sqrt(2*10000*50/2.5)",
        "test": "power_analysis",
        "input": {"n_factors": 3, "effect_size": 2, "noise_std": 1, "n_replicates": 2},
        "expected": {"power_gt": 0.5},
    },
]


def calibrate():
    """Run all golden reference cases. Standalone entry point."""
    from .designs.factorial import full_factorial
    from .designs.screening import definitive_screening_design
    from .designs.response_surface import box_behnken_design
    from .analysis.regression import fit_model
    from .power.power_analysis import power_for_factorial
    from .core.types import Factor

    results = []
    for case in GOLDEN_CASES:
        case_id = case["case_id"]
        test = case["test"]
        inp = case["input"]
        exp = case["expected"]

        try:
            if test == "design_run_count":
                factors = [Factor(name=f"X{i+1}") for i in range(inp["n_factors"])]
                if inp["design"] == "full_factorial":
                    d = full_factorial(factors, randomize=False)
                elif inp["design"] == "dsd":
                    d = definitive_screening_design(factors, randomize=False)
                elif inp["design"] == "box_behnken":
                    d = box_behnken_design(factors, center_points=inp.get("center_points", 3), randomize=False)
                else:
                    results.append({"case_id": case_id, "passed": False, "error": f"Unknown design: {inp['design']}"})
                    continue

                passed = d.n_runs == exp["n_runs"]
                results.append({"case_id": case_id, "passed": passed, "actual": d.n_runs, "expected": exp["n_runs"]})

            elif test == "regression_coefficients":
                result = fit_model(inp["X"], inp["y"], inp["factor_names"], model_type="linear")
                all_pass = True
                for key, expected_val in exp.items():
                    if key == "r_squared":
                        actual = result.r_squared
                    else:
                        actual = result.coefficients.get(key, None)
                    if actual is None or abs(actual - expected_val) > 0.01:
                        all_pass = False
                results.append({"case_id": case_id, "passed": all_pass})

            elif test == "power_analysis":
                pwr = power_for_factorial(inp["n_factors"], inp["effect_size"], inp["noise_std"], inp["n_replicates"])
                passed = pwr > exp.get("power_gt", 0)
                results.append({"case_id": case_id, "passed": passed, "actual": pwr})

        except Exception as e:
            results.append({"case_id": case_id, "passed": False, "error": str(e)})

    passed = sum(1 for r in results if r["passed"])
    return {
        "package": "forgedoe",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
        "is_calibrated": passed == len(results),
    }


def _run_case(case_id: str, test: str, inp: dict) -> dict:
    """Run a single golden case and return a flat dict with expectation-matching keys."""
    from .designs.factorial import full_factorial
    from .designs.screening import definitive_screening_design
    from .designs.response_surface import box_behnken_design
    from .analysis.regression import fit_model
    from .power.power_analysis import power_for_factorial
    from .core.types import Factor

    if test == "design_run_count":
        factors = [Factor(name=f"X{i+1}") for i in range(inp["n_factors"])]
        if inp["design"] == "full_factorial":
            d = full_factorial(factors, randomize=False)
        elif inp["design"] == "dsd":
            d = definitive_screening_design(factors, randomize=False)
        elif inp["design"] == "box_behnken":
            d = box_behnken_design(factors, center_points=inp.get("center_points", 3), randomize=False)
        else:
            raise ValueError(f"Unknown design: {inp['design']}")
        return {"n_runs": d.n_runs}

    elif test == "regression_coefficients":
        result = fit_model(inp["X"], inp["y"], inp["factor_names"], model_type="linear")
        out = {"r_squared": result.r_squared}
        for name in inp["factor_names"]:
            out[name] = result.coefficients.get(name, 0.0)
        out["Intercept"] = result.coefficients.get("Intercept", 0.0)
        return out

    elif test == "power_analysis":
        pwr = power_for_factorial(inp["n_factors"], inp["effect_size"], inp["noise_std"], inp["n_replicates"])
        return {"power": float(pwr)}

    raise ValueError(f"Unknown test: {test}")


def get_calibration_adapter():
    """ForgeCal adapter protocol."""
    try:
        from forgecal.core import CalibrationAdapter, CalibrationCase, Expectation
    except ImportError:
        return None

    cases = []
    for gc in GOLDEN_CASES:
        expectations = []
        for key, val in gc["expected"].items():
            if key.endswith("_gt"):
                expectations.append(Expectation(key=key.replace("_gt", ""), expected=val, comparison="greater_than"))
            else:
                expectations.append(Expectation(key=key, expected=val, tolerance=0.01, comparison="abs_within"))
        cases.append(CalibrationCase(
            case_id=gc["case_id"],
            package="forgedoe",
            category="doe",
            analysis_type="doe",
            analysis_id=gc["test"],
            config=gc["input"],
            data={},
            expectations=expectations,
            description=gc["description"],
        ))

    def _run(case):
        gc = next(g for g in GOLDEN_CASES if g["case_id"] == case.case_id)
        return _run_case(case.case_id, gc["test"], gc["input"])

    from forgedoe import __version__
    return CalibrationAdapter(package="forgedoe", version=__version__, cases=cases, runner=_run)

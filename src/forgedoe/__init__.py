"""ForgeDOE — Design of Experiments engine.

Classical, optimal, screening, and adaptive designs.
Full analysis pipeline: regression, ANOVA, effects, diagnostics, optimization.
Bayesian adaptive DOE for frontier experimentation.

Pure Python + numpy/scipy. No ERP. No Minitab. pip install forgedoe.
"""

__version__ = "0.1.0"

__all__ = [
    # core
    "Factor",
    "FactorType",
    "Response",
    "DesignMatrix",
    "AnalysisResult",
    "OptimizationResult",
    "encode",
    "decode",
    "encode_matrix",
    "decode_matrix",
    # designs
    "full_factorial",
    "fractional_factorial",
    "plackett_burman",
    "central_composite_design",
    "box_behnken_design",
    "definitive_screening_design",
    "latin_hypercube",
    "maximin_lhs",
    # analysis
    "fit_model",
    "desirability_function",
    "composite_desirability",
    "optimize_responses",
    # adaptive
    "AdaptiveExperiment",
    "SequentialPlan",
    "create_sequential_plan",
    # power
    "power_for_factorial",
    "required_replicates",
    "sample_size_for_regression",
    # posteriors (conjugate Normal-Inverse-Gamma)
    "bayesian_linear_posterior",
    "contrast_posterior",
    "predictive_posterior",
    "marginal_log_likelihood",
    # calibration
    "calibrate",
]

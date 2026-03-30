# ForgeDOE

Design of Experiments engine. Pure Python + numpy/scipy. No web framework, no database.

## Architecture

```
forgedoe/
├── core/
│   ├── types.py          # Factor, Response, DesignMatrix, AnalysisResult dataclasses
│   └── coding.py         # Coded <-> natural unit transforms
├── designs/
│   ├── factorial.py      # Full/fractional factorial, Plackett-Burman
│   ├── response_surface.py  # CCD, Box-Behnken
│   ├── screening.py      # Definitive screening designs
│   └── space_filling.py  # Latin hypercube, maximin LHS
├── analysis/
│   ├── regression.py     # Model fitting (effects, ANOVA, diagnostics)
│   └── optimization.py   # Desirability functions, multi-response optimization
├── adaptive/
│   ├── bayesian_doe.py   # Bayesian adaptive DOE (AdaptiveExperiment class)
│   └── sequential.py     # Sequential experiment plans
├── power/
│   └── power_analysis.py # Power, sample size, required replicates
└── calibration.py        # Self-calibration with golden references
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

37 tests across test_designs.py and test_analysis.py.

## Key Design Decisions

- DesignMatrix is the central data structure passed between design generation and analysis
- Coded units (-1/+1) used internally; natural unit conversion via core.coding
- AdaptiveExperiment is stateful: create, suggest points, add observations, check stopping
- All designs return DesignMatrix; all analysis takes DesignMatrix + responses
- Power analysis is separate from design generation (compose as needed)

## Dependencies

- numpy, scipy (required)
- No optional dependencies

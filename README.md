# ForgeDOE

Design of Experiments engine. Classical, optimal, screening, adaptive, and space-filling designs with full analysis pipeline.

## Install

```bash
pip install forgedoe
```

## Quick Start

```python
from forgedoe.core.types import Factor
from forgedoe.designs.factorial import full_factorial
from forgedoe.analysis.regression import fit_model

factors = [Factor("Temp", 150, 200), Factor("Time", 10, 30)]
design = full_factorial(factors)
result = fit_model(design, responses=[85, 90, 78, 95])
```

### Adaptive Bayesian DOE

```python
from forgedoe.adaptive.bayesian_doe import AdaptiveExperiment

exp = AdaptiveExperiment(factors, n_initial=6)
while not exp.should_stop():
    point = exp.suggest_next_point()
    exp.add_observation(point, measured_value)
```

## Modules

| Module | Contents |
|---|---|
| `core.types` | Factor, Response, DesignMatrix, AnalysisResult |
| `core.coding` | Coded/natural unit encoding |
| `designs.factorial` | Full factorial, fractional factorial, Plackett-Burman |
| `designs.response_surface` | Central composite (CCD), Box-Behnken |
| `designs.screening` | Definitive screening designs (DSD) |
| `designs.space_filling` | Latin hypercube, maximin LHS |
| `analysis.regression` | Model fitting, ANOVA, effects, diagnostics |
| `analysis.optimization` | Desirability functions, multi-response optimization |
| `adaptive.bayesian_doe` | Bayesian adaptive experimentation |
| `adaptive.sequential` | Sequential experiment planning |
| `power.power_analysis` | Power analysis, sample size, required replicates |
| `calibration` | Self-calibration against golden references |

## Dependencies

- numpy
- scipy

## License

MIT

# Project: Evolved Habituation & Sensitization (Paper Reproduction)

Reproduces results from "Exploring Adaptive Agency III: Simulating the Evolution of Habituation and Sensitization" by Todd & Miller (2009).

## Overview

A JAX-based neuroevolution system that evolves 2-neuron recurrent networks with Hebbian learning. Agents learn food/poison discrimination via sensory noise and habituation.

- **Details:** [.claude/docs/overview.md](.claude/docs/overview.md)
- **Framework patterns:** [.claude/docs/shared-patterns.md](.claude/docs/shared-patterns.md)

## Neural Network

```
Sensory (S) ──forward──> Motor (M)
    ^                       |
    └───recurrent───────────┘
```

- 2 neurons: Sensory (linear) and Motor (tanh)
- 4 connections + 2 biases = 6 parameters
- 40-bit binary genotype encoding

## Key Features

| Aspect | Description |
|--------|-------------|
| Inputs | 1 (sensory with noise) |
| Fitness | Per-timestep reward/penalty |
| Selection | Fitness-proportionate + elitism |
| Environment | Food/poison with clumpy structure |
| Learning signal | Sensory noise |

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

## Running Experiments

```bash
python scripts/run_experiment.py --quick --name test_run
python scripts/run_experiment.py --runs 5 --name paper_reproduction
python scripts/analyze_results.py data/experiments/paper_reproduction
```

## Paper Reproduction Goals

1. **Figure 1**: 3D surface plot (sensory accuracy × clump scale × generations)
2. **Lifespan hypothesis**: clump-scale 80, 70% accuracy, lifespan 1000 vs 4000
3. **Network architecture analysis**: which weights enable cluster-tracking

## Reference Files

- `habituation_paper.pdf` / `habituation_paper.md` - Source paper
- `simulation_requirements.md` - Extracted requirements
- `coding_plan.md` - Implementation plan
- `REPRODUCTION_PLAN.md` - Reproduction strategy

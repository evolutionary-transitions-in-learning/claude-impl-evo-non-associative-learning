# Project: Evolved Non-Associative Learning

Two separate JAX-based neuroevolution codebases sharing the same framework patterns. Both evolve small recurrent neural networks with Hebbian learning using binary genotype encoding.

## Codebases

### `habituation_evolution/` — Paper Reproduction
Reproduces results from "Exploring Adaptive Agency III" (Todd & Miller). 2-neuron network learns food/poison discrimination via sensory noise and habituation.
- **Details:** [.claude/docs/habituation-evolution.md](.claude/docs/habituation-evolution.md)

### `habituation_experiment_ALICE/` — Threat Discrimination (ALICE)
New experiment design. 3-neuron network with pain channel learns to discriminate true vs false threats via delayed pain signals. Two-phase evaluation + tournament selection.
- **Details:** [.claude/docs/alice-experiment.md](.claude/docs/alice-experiment.md)

## Shared Patterns
Both codebases use the same framework — see [.claude/docs/shared-patterns.md](.claude/docs/shared-patterns.md)

## Project Layout

```
habituation_paper.pdf/md     # Source paper
simulation_requirements.md   # Extracted requirements from paper
coding_plan.md               # Implementation plan for habituation_evolution
habituation_evolution/       # Paper reproduction codebase
habituation_experiment_ALICE/ # ALICE experiment codebase (independent)
```

## Development

Each codebase has its own `.venv/`, `pyproject.toml`, and test suite. They share no imports — fully independent.

```bash
# habituation_evolution
cd habituation_evolution && python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest tests/

# ALICE experiment
cd habituation_experiment_ALICE && python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest tests/
```

## Key Differences at a Glance

| Aspect | habituation_evolution | habituation_experiment_ALICE |
|--------|----------------------|------------------------------|
| Network | 2 neurons (S, M) | 3 neurons (S, P, O) |
| Genotype | 40 bits | 81 bits |
| Inputs | 1 (sensory) | 2 (stimulus + pain) |
| Fitness | Per-timestep reward/penalty | Health at end of lifetime |
| Selection | Fitness-proportionate + elitism | Tournament selection |
| Evaluation | Single phase | Two-phase (survival + discrimination) |
| Environment | Food/poison with noise | True/false threats (identical stimulus) |
| Learning signal | Sensory noise | Delayed pain signal |

## Running Experiments (ALICE)

```bash
cd habituation_experiment_ALICE

# Run experiment
python scripts/run_experiment.py --config config/default.yaml --name my_exp

# Generate all visualizations (16 plot types)
python scripts/gen_all_viz.py runs/<experiment_dir>/
```

See [.claude/docs/alice-visualization.md](.claude/docs/alice-visualization.md) for visualization details.

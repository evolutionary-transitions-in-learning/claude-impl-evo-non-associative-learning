# habituation_evolution — Paper Reproduction

Reproduces "Exploring Adaptive Agency III: Simulating the Evolution of Habituation and Sensitization" by Todd & Miller.

## Neural Network (2 neurons)

```
Sensory (S) ──forward──> Motor (M)
    ^                       |
    |                       |
    └───recurrent───────────┘
    S→S (self)          M→M (self)
```

- **Sensory neuron**: linear activation, receives environment input
- **Motor neuron**: tanh activation, output in [-1, +1]
- **4 connections** + **2 biases** = 6 parameters
- **40-bit genotype** (4 connections × 7 bits + 2 biases × 6 bits)

## Environment

- Binary food/poison sequence with clumpy structure
- Sensory noise: discrete (bit-flip) or continuous (Gaussian)
- Agent perceives noisy version of the true stimulus
- Configurable sensory accuracy (55-95%)

## Fitness (per-timestep reward)

Each timestep:
- Eat food: `+eat_food_reward` (default 1.0)
- Eat poison: `+eat_poison_penalty` (default -1.0)
- Don't eat: `+no_eat_cost` (default 0.0)

Total fitness = sum over lifetime. Higher = better.

## Selection

- **Fitness-proportionate** (roulette wheel) with optional **elitism**
- Linear scaling with `target_max_ratio=1.5`

## Key Modules

| File | Purpose |
|------|---------|
| `config.py` | 6 dataclasses (Environment, Network, Learning, Fitness, Genetic, Simulation) |
| `network.py` | 2-unit RNN with 4 connections, Hebbian learning |
| `genetics.py` | 40-bit genotype, fitness-proportionate selection |
| `environment.py` | Clumpy food/poison, two noise modes |
| `fitness.py` | Per-timestep reward evaluation, baseline computations |
| `simulation.py` | Main loop, convergence criterion |

## Paper Reproduction Goals

1. **Figure 1**: 3D surface plot (sensory accuracy × clump scale × generations to evolve)
2. **Lifespan hypothesis**: clump-scale 80, 70% accuracy, lifespan 1000 vs 4000
3. **Network architecture analysis**: which weights enable cluster-tracking

## Scripts

```bash
python scripts/run_experiment.py --quick --name test_run
python scripts/run_experiment.py --runs 5 --name paper_reproduction
python scripts/analyze_results.py data/experiments/paper_reproduction
```

## Data Storage

```
data/experiments/<name>/
    config.json          # Parameters used
    results.json         # Per-condition results
    generations_data/    # Detailed per-generation logs
```

# Habituation/Sensitization Evolution Simulation

A JAX implementation of the cluster-tracking evolution simulation from:

> Todd, P.M. & Miller, G.F. "Exploring Adaptive Agency III: Simulating the Evolution of Habituation and Sensitization"

## Overview

This simulation explores how simple neural networks can evolve to exhibit habituation and sensitization behaviors when faced with "clumpy" environments where food and poison occur in temporal clusters.

### Key Concepts

- **Cluster-tracking**: The ability to exploit temporal regularities in the environment
- **Sensitization**: Increasing response probability after positive stimuli
- **Habituation**: Decreasing response probability after repeated neutral/negative stimuli

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.random
from habituation_evolution import SimulationConfig, run_simulation

# Create configuration
config = SimulationConfig()
config.environment.clump_scale = 10
config.environment.sensory_accuracy = 0.75

# Run simulation
key = jax.random.PRNGKey(42)
result = run_simulation(key, config, verbose=True)

print(f"Converged: {result.success}")
print(f"Generations: {result.generations_to_success}")
```

## Project Structure

```
├── config/
│   └── default.yaml          # Default parameters
├── src/habituation_evolution/
│   ├── config.py             # Configuration dataclasses
│   ├── environment.py        # Environment generation
│   ├── network.py            # Neural network implementation
│   ├── genetics.py           # Genetic algorithm operators
│   ├── fitness.py            # Fitness evaluation
│   └── simulation.py         # Main simulation loop
├── tests/                    # Unit and integration tests
├── scripts/                  # Experiment runners
├── habituation_paper.pdf     # Source paper
├── simulation_requirements.md  # Extracted requirements
└── REPRODUCTION_PLAN.md      # Reproduction strategy
```

## Configuration

The simulation is highly configurable. Key parameters:

### Environment
- `clump_scale`: Size of food/poison clusters (1-80)
- `sensory_accuracy`: Probability of correct perception (0.5-1.0)
- `lifespan`: Number of timesteps per creature (default: 1000)
- `noise_mode`: `"discrete"` (binary flip) or `"continuous"` (Gaussian noise)

### Network
- `num_weight_magnitudes`: Number of possible weight values (default: 16)
- `max_weight`: Maximum weight magnitude (default: 4.1)
- `motor_activation`: `"tanh"` or `"scaled_sigmoid"`

### Genetic Algorithm
- `population_size`: Number of creatures (default: 100)
- `max_generations`: Maximum evolution time (default: 2000)
- `mutation_rate`: Per-bit mutation probability (default: 0.01)
- `crossover_rate`: Probability of crossover (default: 0.7)

### Learning
- `enabled`: Whether to use Hebbian learning
- `rule`: `"basic"`, `"oja"`, or `"normalized"`
- `learning_rate`: Hebbian learning rate

## Running Experiments

### Single Simulation

```python
from habituation_evolution import SimulationConfig, run_simulation
import jax.random

config = SimulationConfig()
result = run_simulation(jax.random.PRNGKey(42), config)
```

### Full Experimental Grid

```python
from habituation_evolution import SimulationConfig, run_full_experiment
import jax.random

config = SimulationConfig()
# Uses config.experiment settings for clump_scales and accuracies
results = run_full_experiment(jax.random.PRNGKey(42), config)
```

## Neural Network Architecture

The network consists of two units:

1. **Sensory Unit**: Receives smell input (sweet=food, sour=poison)
   - Linear activation
   - Can have self-recurrent connection (memory)

2. **Motor Unit**: Determines eating behavior
   - Logistic activation [-1, +1]
   - Positive = eat, Negative = don't eat

Connections:
- Forward: sensory → motor
- Feedback: motor → sensory (1-timestep delay)
- Self-recurrent on each unit (1-timestep delay)

## Genotype Encoding

40-bit binary genotype encoding:
- 4 connections × 7 bits (present, sign, magnitude, learnable)
- 2 biases × 6 bits (sign, magnitude, learnable)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=habituation_evolution

# Run specific test file
pytest tests/test_environment.py
```

## References

- Todd, P.M., and Miller, G.F. (1990). "Exploring adaptive agency II: Simulating the evolution of associative learning."
- Miller, G.F., and Todd, P.M. (1990). "Exploring adaptive agency I: Theory and methods for simulating the evolution of learning."
- Miller, G.F., Todd, P.M., and Hegde, S.U. (1989). "Designing neural networks using genetic algorithms."

## License

MIT License

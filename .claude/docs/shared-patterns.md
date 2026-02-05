# Shared Framework Patterns

Both codebases use identical patterns. Understanding one transfers to the other.

## Binary Genotype Encoding

Each network parameter encoded as bits in a flat integer array:

**Connection (7 bits):**
- Bit 0: present (1) or absent (0)
- Bit 1: sign (1=positive, 0=negative)
- Bits 2-5: magnitude index (0-15 into weight lookup table)
- Bit 6: learnable flag (1=Hebbian learning applies)

**Bias (6 bits):**
- Bit 0: sign
- Bits 1-4: magnitude index
- Bit 5: learnable flag

**Time constant (5 bits, CTRNN mode only):**
- 32 levels linearly mapped to `[tau_min, tau_max]`

Weight magnitudes: 16 linearly-spaced values from 0.0 to 4.1.

## Continuous Genotype Encoding (ALICE only)

Float array with direct parameter values:

**Layout:** `[weights, biases, learnable_values, (time_constants if CTRNN)]`
- Weights/biases: float values clipped to `[-max_weight, max_weight]`
- Learnable values: thresholded at 0.5 to produce boolean mask
- Time constants (CTRNN): float values clipped to `[tau_min, tau_max]`

Sizes for N=1: SIMPLE = 24 floats, CTRNN = 27 floats.

## Hebbian Learning (within lifetime)

Applied per-timestep to parameters where `learnable_mask=True`:
- **Basic:** `dw = lr * pre * post`
- **Oja:** `dw = lr * post * (pre - post * w)`
- **Normalized:** `dw = lr * pre * post / ||w||`

Weights clipped to [-4.1, 4.1] after each update.

## JAX Patterns

- **`jax.lax.scan`** for temporal simulation (network steps, health dynamics)
- **`jax.vmap`** for batch population evaluation
- **`@jax.jit`** on the generation runner for speed
- **Pure functional** — no mutation, all state passed explicitly
- **NamedTuples** for all data structures (JIT-compatible)
- **Python-level dispatch** via `if/else` on config enums in JIT closures (traced at compile time)

## Genetic Operators

### Binary mode
- **Two-point crossover**: swap middle segment between two parents
- **Point mutation**: per-bit flip with configurable rate (default 0.01)
- Both operate on raw bit arrays (shape-agnostic)

### Continuous mode (ALICE only)
- **Uniform crossover**: each gene independently selected from either parent (p=0.5)
- **Gaussian mutation**: perturb each gene with probability `mutation_rate` by adding N(0, `mutation_std`) noise, then clip to valid ranges

## Configuration

Nested dataclasses with YAML serialization:
- `SimulationConfig` (top-level) contains sub-configs
- `from_yaml()`, `to_dict()`, `save_yaml()`, `with_updates()`
- `with_updates()` supports dotted notation: `config.with_updates(**{"genetic.mutation_rate": 0.05})`

## Project Structure (both follow this)

```
src/<package_name>/
    config.py       # Configuration dataclasses
    network.py      # Neural network + Hebbian learning
    genetics.py     # Genotype encode/decode + operators
    environment.py  # Stimulus sequence generation
    simulation.py   # Main evolutionary loop
scripts/            # Entry points
tests/              # pytest test suite
config/             # YAML config files
```

## Clumpy Environments

Both generate threat/food sequences with configurable clump scale:
1. Random binary base sequence of length `ceil(lifetime / clump_scale)`
2. Repeat each element `clump_scale` times
3. Truncate to exact lifetime

This creates temporally correlated stimulus patterns.

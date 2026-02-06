# Framework Patterns

Core patterns used throughout the codebase.

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

Weight magnitudes: 16 linearly-spaced values from 0.0 to 4.1.

## Hebbian Learning (within lifetime)

Applied per-timestep to parameters where `learnable_mask=True`:
- **Basic:** `dw = lr * pre * post`
- **Oja:** `dw = lr * post * (pre - post * w)`
- **Normalized:** `dw = lr * pre * post / ||w||`

Weights clipped to [-4.1, 4.1] after each update.

## JAX Patterns

- **`jax.lax.scan`** for temporal simulation (network steps, fitness accumulation)
- **`jax.vmap`** for batch population evaluation
- **`@jax.jit`** on the generation runner for speed
- **Pure functional** — no mutation, all state passed explicitly
- **NamedTuples** for all data structures (JIT-compatible)

## Genetic Operators

- **Two-point crossover**: swap middle segment between two parents
- **Point mutation**: per-bit flip with configurable rate (default 0.01)
- Both operate on raw bit arrays (shape-agnostic)

## Configuration

Nested dataclasses with YAML serialization:
- `SimulationConfig` (top-level) contains sub-configs
- `from_yaml()`, `to_dict()`, `save_yaml()`, `with_updates()`
- `with_updates()` supports dotted notation: `config.with_updates(**{"genetic.mutation_rate": 0.05})`

## Project Structure

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

Food/poison sequences generated with configurable clump scale:
1. Random binary base sequence of length `ceil(lifetime / clump_scale)`
2. Repeat each element `clump_scale` times
3. Truncate to exact lifetime

This creates temporally correlated stimulus patterns.

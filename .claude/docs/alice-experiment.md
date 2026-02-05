# habituation_experiment_ALICE — Threat Discrimination

New experiment: organisms face stimuli that may be true or false threats with identical sensory profiles. Must learn from delayed pain signals to discriminate.

## Neural Network (3 neurons)

```
Stimulus (S) ──forward──> Output (O)     output in [-1, +1]
Pain     (P) ──forward──>                +1 = withdraw, -1 = eat
                                          0 = neutral
Full connectivity (9 weights + 3 biases = 12 parameters)
```

All 9 directed connections exist (including cross-input S↔P, self-recurrent, recurrent O→S/P). Input neurons use linear activation; output uses tanh.

### Network Modes (`network_mode`)

**SIMPLE** (default): Feedforward-with-recurrence. Each timestep:
```
activation[i] = sum(w[j→i] * input[j]) + bias[i]
output = tanh(activation[OUTPUT])
```

**CTRNN**: Continuous-Time Recurrent Neural Network with per-neuron time constants. Euler integration (dt=1):
```
x_i(t+1) = x_i(t) + (1/tau_i) * (-x_i(t) + sum_j w_ij * tanh(x_j(t)) + I_i(t) + b_i)
output = tanh(x_output)
```
- Time constants (`tau`) evolved per neuron, range `[tau_min, tau_max]`
- Higher tau = slower dynamics, lower tau = faster response
- Adds 3 time constant parameters (one per neuron)

## Genotype Modes (`genotype_mode`)

### Binary (default)
- **SIMPLE**: 81 bits (9 connections × 7 bits + 3 biases × 6 bits)
- **CTRNN**: 96 bits (81 + 3 time constants × 5 bits each)
- Genetic operators: two-point crossover + per-bit point mutation

### Continuous
- **SIMPLE**: 24 floats (9 weights + 3 biases + 12 learnable values)
- **CTRNN**: 27 floats (24 + 3 time constants)
- Weights/biases: direct float values clipped to `[-max_weight, max_weight]`
- Learnable values: thresholded at 0.5 → boolean mask
- Genetic operators: uniform crossover + Gaussian mutation (`mutation_std`)

## Two-Phase Evaluation

**Phase 1 — Survival Test:**
- All threats are TRUE (100% true threat ratio)
- Must learn to withdraw to survive
- Death = health reaches 0

**Phase 2 — Discrimination Test:**
- Mix of true and false threats (default 50/50)
- Identical stimulus for both; only true threats cause delayed pain
- Network params reset (same genotype, fresh decode)
- Fitness = final health

## Health System

Per-timestep delta:
```
delta = -passive_decay                                    # always (-0.1)
      + max(-output, 0) * eating_gain_rate                # eating (+1.0)
      - true_threat * (1 - max(output, 0)) * threat_damage  # damage (-5.0)
```

Defaults: starting_health=20, death at 0, no health cap.

## Tiered Fitness (bootstrap)

Prevents evolutionary dead end where all random agents die:
- **Tier 1** (died in phase 1): `survival_fraction * 0.99` → [0, 0.99)
- **Tier 2** (survived phase 1, died in phase 2): `1.0`
- **Tier 3** (survived both): `1.0 + phase2_final_health`

## Tournament Selection

N tournaments per generation (N = pop_size):
1. Pick `tournament_size` random individuals (configurable, default 2)
2. Winner = highest fitness, loser = lowest fitness
3. With probability `crossover_rate`: replace loser with crossover(winner, loser) + mutation
4. Otherwise: replace loser with copy(winner) + mutation

Larger `tournament_size` increases selection pressure. Fitness scores held constant during all N tournaments (not updated mid-generation).

## Key Modules

| File | Purpose |
|------|---------|
| `config.py` | 7 dataclasses + enums (`NetworkMode`, `GenotypeMode`, `HebbianRule`, `WeightSpacing`) |
| `network.py` | 3-neuron RNN (SIMPLE/CTRNN modes), Hebbian learning via `lax.scan` |
| `genetics.py` | Binary (81/96 bit) and continuous (24/27 float) genotype encoding/decoding |
| `environment.py` | Clumpy threats, true/false assignment, delayed pain generation |
| `health.py` | Health dynamics with death, `lax.scan` simulation |
| `evaluation.py` | Two-phase pipeline, `PopulationEvalSummary`, `AgentTrace` |
| `selection.py` | Tournament selection (binary/continuous dispatch) via `lax.scan` |
| `simulation.py` | Main loop with checkpoint traces and per-gen statistics |
| `io.py` | Structured save/load for `runs/` folder hierarchy |
| `viz/` | 18 visualization types (see alice-visualization.md) |

## Configuration Enums

| Enum | Values | Config field |
|------|--------|-------------|
| `NetworkMode` | `simple`, `ctrnn` | `network.network_mode` |
| `GenotypeMode` | `binary`, `continuous` | `genetic.genotype_mode` |
| `HebbianRule` | `basic`, `oja`, `normalized` | `learning.rule` |
| `WeightSpacing` | `linear`, `logarithmic` | `network.weight_spacing` |

## Data Collection

Per generation: mean/best/min/std fitness, phase1 survival fraction, phase2 mean health, genotype diversity, best agent weights, population learnable fractions.

Checkpoint traces (~10 per run): full behavioral data for best agent at intervals.

## Running

```bash
cd habituation_experiment_ALICE
python scripts/run_experiment.py --config config/default.yaml --name my_exp
python scripts/gen_all_viz.py runs/<experiment_dir>/
```

### Example: CTRNN + Continuous genotype
```yaml
network:
  network_mode: ctrnn
  tau_min: 0.5
  tau_max: 10.0
genetic:
  genotype_mode: continuous
  mutation_std: 0.1
  tournament_size: 3
```

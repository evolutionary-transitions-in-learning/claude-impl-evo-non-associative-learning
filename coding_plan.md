# Coding Plan: JAX Implementation of Habituation/Sensitization Evolution

## Overview

This document outlines the implementation plan for recreating the Todd & Miller cluster-tracking evolution simulation using Python and JAX. The implementation prioritizes vectorization, JIT compilation, and functional programming patterns native to JAX.

**Reference:** `simulation_requirements.md` for detailed specifications and design decisions.

---

## 1. Project Structure

```
habituation_evolution/
├── pyproject.toml              # Project dependencies and metadata
├── README.md
├── config/
│   └── default.yaml            # Default simulation parameters
├── src/
│   └── habituation_evolution/
│       ├── __init__.py
│       ├── config.py           # Configuration dataclasses
│       ├── environment.py      # Environment generation
│       ├── network.py          # Neural network (phenotype)
│       ├── genetics.py         # Genotype encoding/decoding, GA operators
│       ├── simulation.py       # Main simulation loop
│       ├── fitness.py          # Fitness evaluation
│       ├── analysis.py         # Results analysis and metrics
│       └── visualization.py    # Plotting utilities
├── tests/
│   ├── test_environment.py
│   ├── test_network.py
│   ├── test_genetics.py
│   ├── test_fitness.py
│   └── test_integration.py
├── notebooks/
│   └── exploration.ipynb       # Interactive experimentation
└── scripts/
    ├── run_experiment.py       # Main experiment runner
    └── analyze_results.py      # Post-hoc analysis
```

---

## 2. Dependencies

```toml
[project]
dependencies = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "flax>=0.8.0",          # For neural network utilities (optional)
    "optax>=0.1.0",         # For potential future learning extensions
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "pandas>=2.0.0",        # For results aggregation
]
```

---

## 3. Implementation Phases

### Phase 1: Core Data Structures and Configuration
### Phase 2: Environment System
### Phase 3: Neural Network (Phenotype)
### Phase 4: Genetic Encoding/Decoding
### Phase 5: Fitness Evaluation
### Phase 6: Genetic Algorithm Operators
### Phase 7: Main Simulation Loop
### Phase 8: Analysis and Visualization
### Phase 9: Experiment Runner and Validation

---

## 4. Detailed Module Implementation

### 4.1 Configuration (`config.py`)

**Purpose:** Define all configurable parameters as structured dataclasses.

```python
@dataclass
class EnvironmentConfig:
    clump_scale: int = 10
    sensory_accuracy: float = 0.75
    lifespan: int = 1000

@dataclass
class NetworkConfig:
    weight_magnitudes: int = 16
    max_weight: float = 4.1
    # Design decision: weight spacing
    weight_spacing: str = "linear"  # or "logarithmic"
    # Design decision: input encoding
    sweet_input: float = 1.0
    sour_input: float = -1.0
    # Design decision: noise model
    noise_model: str = "binary_flip"  # or "gaussian"

@dataclass
class GeneticConfig:
    population_size: int = 100
    max_generations: int = 2000
    # Design decisions: GA parameters
    mutation_rate: float = 0.01      # NEEDS TUNING
    crossover_rate: float = 0.6      # NEEDS TUNING
    # Linear scaling parameters
    scaling_target_max: float = 2.0  # NEEDS TUNING

@dataclass
class FitnessConfig:
    eat_food_reward: float = 1.0     # NEEDS TUNING
    eat_poison_penalty: float = -1.0 # NEEDS TUNING
    no_eat_cost: float = 0.0         # NEEDS TUNING
    success_threshold: float = 0.8   # NEEDS TUNING (relative to optimal)

@dataclass
class LearningConfig:
    hebbian_rate: float = 0.01       # NEEDS TUNING
    rule: str = "basic"              # "basic", "oja", "normalized"

@dataclass
class SimulationConfig:
    environment: EnvironmentConfig
    network: NetworkConfig
    genetic: GeneticConfig
    fitness: FitnessConfig
    learning: LearningConfig
    seed: int = 42
    num_runs: int = 5
```

**Implementation Notes:**
- Use `chex.dataclass` for JAX compatibility if needed
- Implement YAML loading/saving for configuration

---

### 4.2 Environment (`environment.py`)

**Purpose:** Generate clumpy food/poison sequences with sensory noise.

```python
def generate_base_sequence(key: PRNGKey, length: int) -> Array:
    """Generate random food(1)/poison(0) base sequence."""
    pass

def expand_to_clumps(base_sequence: Array, clump_scale: int) -> Array:
    """Expand each element to clump_scale consecutive copies."""
    pass

def generate_environment(key: PRNGKey, config: EnvironmentConfig) -> Array:
    """Generate full environment sequence for one creature's lifespan."""
    pass

def apply_sensory_noise(
    key: PRNGKey,
    true_sequence: Array,
    accuracy: float,
    noise_model: str
) -> Array:
    """Apply perceptual noise to environment sequence."""
    pass

# Vectorized versions for batch processing
def generate_environments_batch(
    key: PRNGKey,
    config: EnvironmentConfig,
    batch_size: int
) -> Array:
    """Generate environments for entire population."""
    pass
```

**JAX Patterns:**
- Use `jax.random.split` for generating independent sequences
- Use `jax.vmap` for batch environment generation
- `jax.lax.scan` or array operations for clump expansion

**Testing:**
- Verify clump statistics (min, mean, max lengths)
- Verify noise application rates match accuracy parameter
- Test edge cases (clump_scale=1, accuracy=1.0)

---

### 4.3 Neural Network (`network.py`)

**Purpose:** Implement the 2-unit recurrent network as pure functions.

```python
@dataclass
class NetworkState:
    sensory_activation: Array  # Shape: () or (batch,)
    motor_activation: Array    # Shape: () or (batch,)
    weights: Array             # Shape: (4,) - [fwd, rec_m2s, self_s, self_m]
    biases: Array              # Shape: (2,) - [sensory, motor]

@dataclass
class NetworkParams:
    weights: Array             # Shape: (4,)
    biases: Array              # Shape: (2,)
    learnable_mask: Array      # Shape: (6,) - which params are learnable

def init_network_state(params: NetworkParams) -> NetworkState:
    """Initialize network state with zero activations."""
    pass

def motor_activation_fn(x: Array) -> Array:
    """Motor unit activation: scaled sigmoid to [-1, +1]."""
    # Design decision: 2*sigmoid(x)-1 or tanh(x)
    return jnp.tanh(x)  # Simpler, same range

def sensory_activation_fn(x: Array) -> Array:
    """Sensory unit activation: linear."""
    return x

def network_step(
    state: NetworkState,
    sensory_input: Array,
    params: NetworkParams
) -> tuple[NetworkState, Array]:
    """
    Single timestep network update.

    Returns: (new_state, motor_output)
    """
    pass

def get_eating_decision(motor_activation: Array) -> Array:
    """Convert motor activation to binary eat/don't-eat decision."""
    return motor_activation > 0.0

def apply_hebbian_learning(
    state: NetworkState,
    params: NetworkParams,
    config: LearningConfig
) -> NetworkParams:
    """Apply Hebbian weight updates to learnable connections."""
    pass

# Vectorized for batch processing
def network_step_batch(
    states: NetworkState,  # batched
    sensory_inputs: Array,
    params: NetworkParams  # batched
) -> tuple[NetworkState, Array]:
    """Batched network step using vmap."""
    pass
```

**JAX Patterns:**
- Pure functions with explicit state passing
- Use `jax.lax.scan` for processing entire lifespan
- `jax.vmap` over population dimension

**Testing:**
- Unit test each activation function
- Test network update with known weights
- Verify recurrent connection delays work correctly

---

### 4.4 Genetic Encoding (`genetics.py`)

**Purpose:** Encode/decode genotypes, implement GA operators.

```python
# Genotype structure (suggested):
# Per connection (4 total): 1 bit present + 1 bit sign + 4 bits magnitude + 1 bit learnable = 7 bits
# Per bias (2 total): 1 bit sign + 4 bits magnitude + 1 bit learnable = 6 bits
# Total: 4*7 + 2*6 = 40 bits

GENOTYPE_LENGTH = 40  # Adjust based on final encoding scheme

def create_random_genotype(key: PRNGKey) -> Array:
    """Generate random binary genotype."""
    pass

def create_random_population(key: PRNGKey, pop_size: int) -> Array:
    """Generate population of random genotypes. Shape: (pop_size, GENOTYPE_LENGTH)"""
    pass

def decode_genotype(genotype: Array, config: NetworkConfig) -> NetworkParams:
    """Convert binary genotype to network parameters."""
    pass

def decode_weight(bits: Array, config: NetworkConfig) -> float:
    """Decode weight bits to float value."""
    # Extract sign bit
    # Extract magnitude bits
    # Apply spacing (linear: magnitude * max_weight / 15)
    pass

def encode_genotype(params: NetworkParams, config: NetworkConfig) -> Array:
    """Convert network parameters back to genotype (for testing)."""
    pass

# Genetic operators
def two_point_crossover(
    key: PRNGKey,
    parent1: Array,
    parent2: Array
) -> tuple[Array, Array]:
    """Two-point crossover producing two offspring."""
    pass

def point_mutation(key: PRNGKey, genotype: Array, rate: float) -> Array:
    """Apply point mutation with given per-bit rate."""
    pass

def select_parents_fitness_proportionate(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    num_parents: int
) -> Array:
    """Stochastic fitness-proportionate selection."""
    pass

def apply_linear_scaling(
    fitness_scores: Array,
    target_max: float
) -> Array:
    """Apply linear fitness scaling."""
    # f' = a*f + b where we set constraints
    pass

# Batched operations
def crossover_population(
    key: PRNGKey,
    parents: Array,
    crossover_rate: float
) -> Array:
    """Apply crossover to create new population."""
    pass

def mutate_population(
    key: PRNGKey,
    population: Array,
    mutation_rate: float
) -> Array:
    """Apply mutation to entire population."""
    pass
```

**JAX Patterns:**
- Bit manipulation using JAX array operations
- `jax.random.choice` for selection with replacement
- `jax.vmap` for batch crossover/mutation

**Testing:**
- Round-trip test: encode → decode → encode
- Verify crossover produces valid genotypes
- Verify mutation rate statistically
- Test selection pressure

---

### 4.5 Fitness Evaluation (`fitness.py`)

**Purpose:** Evaluate creature fitness over lifespan.

```python
def compute_step_fitness(
    ate: Array,           # bool: did creature eat?
    is_food: Array,       # bool: was it food?
    config: FitnessConfig
) -> Array:
    """Compute fitness delta for single timestep."""
    # ate & is_food → +reward
    # ate & !is_food → +penalty (negative)
    # !ate → +no_eat_cost
    pass

def evaluate_creature_lifespan(
    key: PRNGKey,
    genotype: Array,
    env_config: EnvironmentConfig,
    net_config: NetworkConfig,
    fitness_config: FitnessConfig,
    learning_config: LearningConfig
) -> float:
    """
    Evaluate single creature over full lifespan.

    1. Decode genotype to network params
    2. Generate environment sequence
    3. Apply sensory noise
    4. Run network for each timestep
    5. Accumulate fitness
    """
    pass

def evaluate_creature_lifespan_given_env(
    genotype: Array,
    true_environment: Array,
    perceived_environment: Array,
    net_config: NetworkConfig,
    fitness_config: FitnessConfig,
    learning_config: LearningConfig
) -> float:
    """Evaluate creature with pre-generated environment (for fair comparison)."""
    pass

# Batched evaluation
def evaluate_population(
    key: PRNGKey,
    population: Array,
    config: SimulationConfig
) -> Array:
    """Evaluate entire population. Returns fitness array of shape (pop_size,)."""
    pass
```

**JAX Patterns:**
- Use `jax.lax.scan` for lifespan iteration with carried state
- `jax.vmap` over population for parallel evaluation
- JIT compile entire evaluation function

**Testing:**
- Test with known "always eat" and "never eat" strategies
- Verify fitness accumulation
- Test with perfect accuracy (should achieve optimal)

---

### 4.6 Simulation Loop (`simulation.py`)

**Purpose:** Main evolutionary loop.

```python
@dataclass
class GenerationState:
    population: Array           # Shape: (pop_size, genotype_length)
    fitness_history: Array      # Shape: (generation,)
    best_fitness_history: Array
    generation: int
    key: PRNGKey

def init_simulation(key: PRNGKey, config: SimulationConfig) -> GenerationState:
    """Initialize simulation with random population."""
    pass

def run_generation(
    state: GenerationState,
    config: SimulationConfig
) -> GenerationState:
    """
    Execute one generation:
    1. Evaluate population fitness
    2. Check termination condition
    3. Apply linear scaling
    4. Select parents
    5. Apply crossover
    6. Apply mutation
    7. Update state
    """
    pass

def check_termination(
    mean_fitness: float,
    config: SimulationConfig
) -> bool:
    """Check if cluster-tracking has evolved (fitness threshold reached)."""
    pass

def run_simulation(
    key: PRNGKey,
    config: SimulationConfig
) -> dict:
    """
    Run full simulation until termination or max generations.

    Returns dict with:
    - generations_to_success: int (or max_generations if failed)
    - final_population: Array
    - fitness_history: Array
    - best_genotype: Array
    - success: bool
    """
    pass

def run_simulation_jit(
    key: PRNGKey,
    config: SimulationConfig
) -> dict:
    """JIT-compiled simulation using lax.while_loop."""
    pass
```

**JAX Patterns:**
- `jax.lax.while_loop` for generation loop (JIT-compatible)
- `jax.lax.cond` for conditional termination
- Consider `jax.lax.fori_loop` if fixed generations

**Testing:**
- Test single generation execution
- Verify fitness improves over generations
- Test termination conditions

---

### 4.7 Analysis (`analysis.py`)

**Purpose:** Analyze simulation results and extract metrics.

```python
def analyze_best_network(genotype: Array, config: NetworkConfig) -> dict:
    """
    Analyze evolved network architecture.

    Returns:
    - Which connections are present
    - Weight values
    - Which connections are learnable
    - Interpretation (sensitization/habituation indicators)
    """
    pass

def compute_cluster_tracking_score(
    genotype: Array,
    config: SimulationConfig,
    num_trials: int = 100
) -> float:
    """Evaluate how well a network tracks clusters vs random baseline."""
    pass

def find_average_creature(population: Array, fitness: Array) -> Array:
    """Find representative 'average' creature (fitness-weighted centroid)."""
    pass

def run_experiment_grid(
    clump_scales: list[int],
    accuracies: list[float],
    num_runs: int,
    base_config: SimulationConfig
) -> pd.DataFrame:
    """
    Run full experimental grid as in paper.

    Returns DataFrame with columns:
    - clump_scale, accuracy, run_id, generations, success
    """
    pass

def compute_summary_statistics(results: pd.DataFrame) -> pd.DataFrame:
    """Compute mean generations per condition."""
    pass
```

---

### 4.8 Visualization (`visualization.py`)

**Purpose:** Plotting utilities for results.

```python
def plot_fitness_history(history: Array, ax=None) -> plt.Axes:
    """Plot fitness over generations for single run."""
    pass

def plot_3d_surface(
    results: pd.DataFrame,
    metric: str = "generations"
) -> plt.Figure:
    """
    Recreate Figure 1 from paper.

    3D surface plot:
    - X axis: sensory accuracy (50% to 100%)
    - Y axis: clump scale (1 to 80)
    - Z axis: generations to evolve cluster-tracking
    """
    pass

def plot_network_behavior(
    genotype: Array,
    environment: Array,
    config: SimulationConfig
) -> plt.Figure:
    """
    Visualize network behavior over time:
    - True environment (food/poison)
    - Perceived environment
    - Network activations
    - Eating decisions
    - Cumulative fitness
    """
    pass

def plot_weight_evolution(
    population_history: list[Array],
    config: NetworkConfig
) -> plt.Figure:
    """Plot evolution of weight distributions over generations."""
    pass
```

---

## 5. Implementation Order and Milestones

### Milestone 1: Foundation (Phase 1-2)
**Goal:** Environment generation working and tested.

1. Set up project structure and dependencies
2. Implement `config.py` with all dataclasses
3. Implement `environment.py`:
   - `generate_base_sequence`
   - `expand_to_clumps`
   - `apply_sensory_noise`
   - Batch versions
4. Write comprehensive tests for environment
5. **Checkpoint:** Can generate valid clumpy environments with noise

### Milestone 2: Network (Phase 3)
**Goal:** Neural network forward pass working.

1. Implement `network.py`:
   - Data structures
   - Activation functions
   - `network_step`
   - `apply_hebbian_learning`
2. Write tests verifying network dynamics
3. **Checkpoint:** Can run network on environment sequence

### Milestone 3: Genetics (Phase 4)
**Goal:** Genotype encoding/decoding and GA operators working.

1. Implement `genetics.py`:
   - Genotype structure definition
   - `decode_genotype` / `encode_genotype`
   - `two_point_crossover`
   - `point_mutation`
   - Selection operators
2. Write round-trip and statistical tests
3. **Checkpoint:** Can create, decode, and evolve genotypes

### Milestone 4: Fitness (Phase 5)
**Goal:** Full creature evaluation working.

1. Implement `fitness.py`:
   - `compute_step_fitness`
   - `evaluate_creature_lifespan`
   - `evaluate_population`
2. Test with known strategies
3. **Checkpoint:** Can evaluate population fitness

### Milestone 5: Evolution (Phase 6-7)
**Goal:** Full simulation loop working.

1. Implement `simulation.py`:
   - `run_generation`
   - `run_simulation`
   - JIT-compiled version
2. Run initial experiments, verify evolution occurs
3. **Checkpoint:** Can run complete simulation, see fitness improvement

### Milestone 6: Validation (Phase 8-9)
**Goal:** Reproduce paper results.

1. Implement `analysis.py` and `visualization.py`
2. Run full experimental grid (42 conditions × 5+ runs)
3. Generate Figure 1 equivalent
4. Compare results to paper
5. **Checkpoint:** Results qualitatively match paper (U-shaped curves, ravine pattern)

### Milestone 7: Polish
**Goal:** Production-ready code.

1. Performance optimization
2. Documentation
3. Example notebooks
4. Parameter sensitivity analysis

---

## 6. Testing Strategy

### Unit Tests
- Each module has corresponding test file
- Test pure functions with known inputs/outputs
- Use `hypothesis` for property-based testing where appropriate

### Integration Tests
- Test environment → network → fitness pipeline
- Test full generation cycle
- Test multi-generation evolution

### Validation Tests
- Compare against expected paper results
- Statistical tests for GA operator correctness
- Verify clump statistics match theory

### Performance Tests
- Benchmark evaluation throughput
- Profile JIT compilation times
- Test scaling with population size

---

## 7. Design Decision Defaults

For initial implementation, use these defaults (to be tuned):

| Decision | Default Value | Rationale |
|----------|---------------|-----------|
| Noise model | Binary flip | Matches paper description most directly |
| Motor activation | `tanh(x)` | Simpler, same range as described |
| Weight spacing | Linear | Simplest assumption |
| Input encoding | +1 (sweet), -1 (sour) | Symmetric, simple |
| Decision threshold | 0.0 | Natural for tanh output |
| Hebbian rule | Basic: Δw = η * pre * post | Simplest form |
| Hebbian rate | 0.01 | Conservative starting point |
| Eat food reward | +1.0 | Symmetric with penalty |
| Eat poison penalty | -1.0 | Symmetric with reward |
| No-eat cost | 0.0 | Start without starvation |
| Success threshold | Mean fitness > 0.7 × optimal | Needs calibration |
| Mutation rate | 0.01 per bit | Common default |
| Crossover rate | 0.7 | Common default |
| Scaling target max | 1.5 × mean | Moderate selection pressure |

---

## 8. Potential Optimizations

### JIT Compilation
- Compile entire generation step
- Use `jax.lax.scan` over lifespan to avoid Python loops
- Use `jax.lax.while_loop` for main evolution loop

### Vectorization
- Evaluate all creatures in parallel via `vmap`
- Process all timesteps in parallel where possible (environment generation)

### Memory
- Don't store full population history unless needed
- Use `jax.checkpoint` for gradient computation if adding learning

### Parallelization
- Use `jax.pmap` for multi-device execution
- Run independent simulation runs in parallel

---

## 9. Future Extensions

1. **GPU Acceleration:** Ensure all operations are GPU-compatible
2. **Hyperparameter Tuning:** Systematic search for GA parameters
3. **Extended Analysis:** Network architecture categorization
4. **Visualization Dashboard:** Interactive exploration of results
5. **Comparison Studies:** Different learning rules, network sizes

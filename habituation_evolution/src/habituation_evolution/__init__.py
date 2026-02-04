"""Habituation/Sensitization Evolution Simulation.

A JAX implementation of the cluster-tracking evolution simulation from:
Todd & Miller "Exploring Adaptive Agency III: Simulating the Evolution
of Habituation and Sensitization"

Main modules:
- config: Configuration dataclasses
- environment: Environment generation with clumpy food/poison
- network: Neural network implementation
- genetics: Genetic algorithm operators
- fitness: Fitness evaluation
- simulation: Main evolutionary loop

Quick start:
    >>> from habituation_evolution import SimulationConfig, run_simulation
    >>> import jax.random
    >>> key = jax.random.PRNGKey(42)
    >>> config = SimulationConfig()
    >>> result = run_simulation(key, config)
"""

from .config import (
    EnvironmentConfig,
    ExperimentConfig,
    FitnessConfig,
    GeneticConfig,
    HebbianRule,
    LearningConfig,
    MotorActivation,
    NetworkConfig,
    NoiseMode,
    SimulationConfig,
    WeightSpacing,
)
from .environment import (
    Environment,
    generate_environment,
    generate_environments_batch,
    generate_clumpy_sequence,
)
from .network import (
    NetworkParams,
    NetworkState,
    init_network_state,
    init_network_params,
    network_step,
    run_network_lifespan,
)
from .genetics import (
    GENOTYPE_LENGTH,
    create_random_genotype,
    create_random_population,
    decode_genotype,
    decode_population,
    two_point_crossover,
    point_mutation,
    create_next_generation,
)
from .fitness import (
    evaluate_creature,
    evaluate_population,
    compute_optimal_fitness,
    compute_sensory_only_fitness,
    compute_success_threshold,
)
from .simulation import (
    SimulationState,
    SimulationHistory,
    SimulationResult,
    init_simulation,
    run_generation,
    run_simulation,
    run_simulation_jit,
    run_multiple_simulations,
    run_full_experiment,
)

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Config
    "EnvironmentConfig",
    "ExperimentConfig",
    "FitnessConfig",
    "GeneticConfig",
    "HebbianRule",
    "LearningConfig",
    "MotorActivation",
    "NetworkConfig",
    "NoiseMode",
    "SimulationConfig",
    "WeightSpacing",
    # Environment
    "Environment",
    "generate_environment",
    "generate_environments_batch",
    "generate_clumpy_sequence",
    # Network
    "NetworkParams",
    "NetworkState",
    "init_network_state",
    "init_network_params",
    "network_step",
    "run_network_lifespan",
    # Genetics
    "GENOTYPE_LENGTH",
    "create_random_genotype",
    "create_random_population",
    "decode_genotype",
    "decode_population",
    "two_point_crossover",
    "point_mutation",
    "create_next_generation",
    # Fitness
    "evaluate_creature",
    "evaluate_population",
    "compute_optimal_fitness",
    "compute_sensory_only_fitness",
    "compute_success_threshold",
    # Simulation
    "SimulationState",
    "SimulationHistory",
    "SimulationResult",
    "init_simulation",
    "run_generation",
    "run_simulation",
    "run_simulation_jit",
    "run_multiple_simulations",
    "run_full_experiment",
]

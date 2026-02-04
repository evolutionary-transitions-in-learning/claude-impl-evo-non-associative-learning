"""Fitness evaluation for the habituation/sensitization simulation.

This module handles:
- Computing fitness for individual timesteps
- Evaluating creatures over their entire lifespan
- Batch evaluation for entire populations
- Computing theoretical optimal fitness for comparison
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from .config import (
    EnvironmentConfig,
    FitnessConfig,
    LearningConfig,
    NetworkConfig,
    SimulationConfig,
)
from .environment import Environment, generate_environment, generate_environments_batch
from .genetics import decode_genotype, decode_population
from .network import NetworkParams, run_network_lifespan, run_network_lifespan_batch


class EvaluationResult(NamedTuple):
    """Results from evaluating a creature or population.

    Attributes:
        fitness: Total accumulated fitness
        decisions: Eating decisions made (True=eat, False=don't eat)
        motor_outputs: Raw motor unit outputs
        correct_decisions: Number of correct decisions made
        total_decisions: Total number of decisions (lifespan)
    """

    fitness: Array
    decisions: Array
    motor_outputs: Array
    correct_decisions: Array
    total_decisions: Array


def compute_step_fitness(
    ate: Array, is_food: Array, config: FitnessConfig
) -> Array:
    """Compute fitness delta for a single timestep.

    Args:
        ate: Whether creature ate (boolean)
        is_food: Whether substance was food (boolean)
        config: Fitness configuration

    Returns:
        Fitness delta for this timestep
    """
    # Four cases:
    # ate=True, is_food=True -> +eat_food_reward
    # ate=True, is_food=False -> +eat_poison_penalty (negative)
    # ate=False, is_food=True -> +no_eat_cost (missed food)
    # ate=False, is_food=False -> +no_eat_cost (correct avoidance, but still cost)

    ate_food = ate & is_food
    ate_poison = ate & ~is_food

    fitness_delta = jnp.where(
        ate_food,
        config.eat_food_reward,
        jnp.where(ate_poison, config.eat_poison_penalty, config.no_eat_cost),
    )

    return fitness_delta


def compute_lifespan_fitness(
    decisions: Array, true_sequence: Array, config: FitnessConfig
) -> Array:
    """Compute total fitness over entire lifespan.

    Args:
        decisions: Eating decisions, shape (lifespan,)
        true_sequence: True food(1)/poison(0) sequence, shape (lifespan,)
        config: Fitness configuration

    Returns:
        Total accumulated fitness
    """
    is_food = true_sequence == 1

    # Vectorized fitness computation
    step_fitness = jax.vmap(lambda ate, food: compute_step_fitness(ate, food, config))(
        decisions, is_food
    )

    return jnp.sum(step_fitness)


def count_correct_decisions(decisions: Array, true_sequence: Array) -> Array:
    """Count number of correct decisions.

    Correct decision = eat food OR don't eat poison.

    Args:
        decisions: Eating decisions, shape (lifespan,)
        true_sequence: True food(1)/poison(0) sequence, shape (lifespan,)

    Returns:
        Number of correct decisions
    """
    is_food = true_sequence == 1

    # Correct: (ate and food) or (didn't eat and poison)
    correct = (decisions & is_food) | (~decisions & ~is_food)

    return jnp.sum(correct)


def evaluate_creature(
    genotype: Array,
    environment: Environment,
    net_config: NetworkConfig,
    learn_config: LearningConfig,
    fitness_config: FitnessConfig,
) -> EvaluationResult:
    """Evaluate a single creature over its lifespan.

    Args:
        genotype: Binary genotype
        environment: Environment with true and perceived sequences
        net_config: Network configuration
        learn_config: Learning configuration
        fitness_config: Fitness configuration

    Returns:
        EvaluationResult with fitness and decision data
    """
    # Decode genotype to network parameters
    params = decode_genotype(genotype, net_config)

    # Run network over lifespan
    motor_outputs, decisions, _ = run_network_lifespan(
        params, environment.perceived_sequence, net_config, learn_config
    )

    # Compute fitness
    fitness = compute_lifespan_fitness(
        decisions, environment.true_sequence, fitness_config
    )

    # Count correct decisions
    correct = count_correct_decisions(decisions, environment.true_sequence)
    total = len(decisions)

    return EvaluationResult(
        fitness=fitness,
        decisions=decisions,
        motor_outputs=motor_outputs,
        correct_decisions=correct,
        total_decisions=jnp.array(total),
    )


def evaluate_creature_with_env_generation(
    key: PRNGKey,
    genotype: Array,
    env_config: EnvironmentConfig,
    net_config: NetworkConfig,
    learn_config: LearningConfig,
    fitness_config: FitnessConfig,
) -> EvaluationResult:
    """Evaluate a creature, generating its environment on the fly.

    Args:
        key: JAX random key
        genotype: Binary genotype
        env_config: Environment configuration
        net_config: Network configuration
        learn_config: Learning configuration
        fitness_config: Fitness configuration

    Returns:
        EvaluationResult
    """
    # Generate environment
    environment = generate_environment(key, env_config, net_config)

    # Evaluate creature
    return evaluate_creature(
        genotype, environment, net_config, learn_config, fitness_config
    )


def evaluate_population(
    key: PRNGKey,
    population: Array,
    env_config: EnvironmentConfig,
    net_config: NetworkConfig,
    learn_config: LearningConfig,
    fitness_config: FitnessConfig,
) -> Array:
    """Evaluate entire population, returning fitness scores.

    Each creature gets its own randomly generated environment.

    Args:
        key: JAX random key
        population: Population of genotypes, shape (pop_size, genotype_length)
        env_config: Environment configuration
        net_config: Network configuration
        learn_config: Learning configuration
        fitness_config: Fitness configuration

    Returns:
        Fitness scores, shape (pop_size,)
    """
    pop_size = len(population)
    keys = jax.random.split(key, pop_size)

    # Vectorize evaluation over population
    eval_fn = partial(
        evaluate_creature_with_env_generation,
        env_config=env_config,
        net_config=net_config,
        learn_config=learn_config,
        fitness_config=fitness_config,
    )

    results = jax.vmap(eval_fn)(keys, population)

    return results.fitness


def evaluate_population_shared_env(
    key: PRNGKey,
    population: Array,
    env_config: EnvironmentConfig,
    net_config: NetworkConfig,
    learn_config: LearningConfig,
    fitness_config: FitnessConfig,
) -> Array:
    """Evaluate population with shared base environment but individual noise.

    All creatures experience the same underlying food/poison sequence,
    but each has independent sensory noise.

    Args:
        key: JAX random key
        population: Population of genotypes
        env_config: Environment configuration
        net_config: Network configuration
        learn_config: Learning configuration
        fitness_config: Fitness configuration

    Returns:
        Fitness scores, shape (pop_size,)
    """
    from .environment import generate_shared_environment

    pop_size = len(population)

    # Generate shared environment with individual noise
    environment = generate_shared_environment(key, env_config, net_config, pop_size)

    # Decode all genotypes
    params_batch = decode_population(population, net_config)

    # Run all networks
    motor_outputs, decisions, _ = run_network_lifespan_batch(
        params_batch, environment.perceived_sequence, net_config, learn_config
    )

    # Compute fitness for each creature
    fitness_fn = partial(compute_lifespan_fitness, config=fitness_config)
    fitness_scores = jax.vmap(fitness_fn)(decisions, environment.true_sequence)

    return fitness_scores


# ============================================================================
# Theoretical Optimal Fitness
# ============================================================================


def compute_optimal_fitness(
    env_config: EnvironmentConfig, fitness_config: FitnessConfig
) -> float:
    """Compute theoretical optimal fitness with perfect decisions.

    Assumes 50% food, 50% poison on average, and perfect decisions
    (eat all food, avoid all poison).

    Args:
        env_config: Environment configuration
        fitness_config: Fitness configuration

    Returns:
        Theoretical maximum fitness
    """
    lifespan = env_config.lifespan

    # With 50% food and 50% poison on average:
    expected_food = lifespan / 2
    expected_poison = lifespan / 2

    # Optimal: eat all food, avoid all poison
    optimal = expected_food * fitness_config.eat_food_reward + expected_poison * fitness_config.no_eat_cost

    return float(optimal)


def compute_random_baseline_fitness(
    env_config: EnvironmentConfig, fitness_config: FitnessConfig
) -> float:
    """Compute expected fitness with random eating decisions.

    Assumes 50% probability of eating at each timestep.

    Args:
        env_config: Environment configuration
        fitness_config: Fitness configuration

    Returns:
        Expected fitness with random decisions
    """
    lifespan = env_config.lifespan

    # Expected outcomes with random 50% eating:
    # eat_food: 0.5 * 0.5 = 0.25 of timesteps
    # eat_poison: 0.5 * 0.5 = 0.25 of timesteps
    # not_eat: 0.5 of timesteps

    expected_fitness = lifespan * (
        0.25 * fitness_config.eat_food_reward
        + 0.25 * fitness_config.eat_poison_penalty
        + 0.5 * fitness_config.no_eat_cost
    )

    return float(expected_fitness)


def compute_always_eat_fitness(
    env_config: EnvironmentConfig, fitness_config: FitnessConfig
) -> float:
    """Compute expected fitness with 'always eat' strategy.

    Args:
        env_config: Environment configuration
        fitness_config: Fitness configuration

    Returns:
        Expected fitness with always eating
    """
    lifespan = env_config.lifespan

    # 50% food, 50% poison, always eat
    expected_fitness = lifespan * (
        0.5 * fitness_config.eat_food_reward + 0.5 * fitness_config.eat_poison_penalty
    )

    return float(expected_fitness)


def compute_never_eat_fitness(
    env_config: EnvironmentConfig, fitness_config: FitnessConfig
) -> float:
    """Compute expected fitness with 'never eat' strategy.

    Args:
        env_config: Environment configuration
        fitness_config: Fitness configuration

    Returns:
        Expected fitness with never eating
    """
    lifespan = env_config.lifespan

    return float(lifespan * fitness_config.no_eat_cost)


def compute_sensory_only_fitness(
    env_config: EnvironmentConfig, fitness_config: FitnessConfig
) -> float:
    """Compute expected fitness with sensory-only strategy.

    This is the expected fitness when the creature eats whenever
    it perceives food (with no cluster-tracking).

    Args:
        env_config: Environment configuration
        fitness_config: Fitness configuration

    Returns:
        Expected fitness with sensory-only strategy
    """
    lifespan = env_config.lifespan
    accuracy = env_config.sensory_accuracy

    # With accuracy a:
    # - Correctly perceive food and eat: 0.5 * a -> reward
    # - Incorrectly perceive food as poison and don't eat: 0.5 * (1-a) -> no_eat_cost
    # - Correctly perceive poison and don't eat: 0.5 * a -> no_eat_cost
    # - Incorrectly perceive poison as food and eat: 0.5 * (1-a) -> penalty

    expected_fitness = lifespan * (
        0.5 * accuracy * fitness_config.eat_food_reward  # correct food
        + 0.5 * (1 - accuracy) * fitness_config.no_eat_cost  # missed food
        + 0.5 * accuracy * fitness_config.no_eat_cost  # correct avoid
        + 0.5 * (1 - accuracy) * fitness_config.eat_poison_penalty  # wrong eat
    )

    return float(expected_fitness)


def compute_success_threshold(config: SimulationConfig) -> float:
    """Compute fitness threshold for successful cluster-tracking evolution.

    Args:
        config: Simulation configuration

    Returns:
        Fitness threshold value
    """
    optimal = compute_optimal_fitness(config.environment, config.fitness)
    sensory_only = compute_sensory_only_fitness(config.environment, config.fitness)

    # Threshold is between sensory-only and optimal
    # success_threshold_ratio determines how close to optimal
    threshold = sensory_only + config.fitness.success_threshold_ratio * (
        optimal - sensory_only
    )

    return threshold


def check_cluster_tracking_evolved(
    mean_fitness: float, config: SimulationConfig
) -> bool:
    """Check if cluster-tracking has evolved based on mean fitness.

    Args:
        mean_fitness: Mean population fitness
        config: Simulation configuration

    Returns:
        True if cluster-tracking has evolved
    """
    threshold = compute_success_threshold(config)
    return mean_fitness >= threshold

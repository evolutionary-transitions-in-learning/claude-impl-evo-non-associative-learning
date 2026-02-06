"""Main simulation loop for the habituation/sensitization evolution.

This module handles:
- Running individual generations
- Managing the evolutionary loop
- Tracking statistics and history
- Termination conditions
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from .config import SimulationConfig
from .fitness import (
    check_cluster_tracking_evolved,
    compute_optimal_fitness,
    compute_sensory_only_fitness,
    compute_success_threshold,
    evaluate_population,
)
from .genetics import create_next_generation_with_elitism, create_random_population


class SimulationState(NamedTuple):
    """State of the simulation at a given generation.

    Attributes:
        population: Current population genotypes, shape (pop_size, genotype_length)
        fitness_scores: Fitness scores for current population, shape (pop_size,)
        generation: Current generation number
        best_fitness: Best fitness in current generation
        mean_fitness: Mean fitness of current generation
        best_genotype: Best genotype in current generation
        key: Current random key
        converged: Whether simulation has converged (cluster-tracking evolved)
    """

    population: Array
    fitness_scores: Array
    generation: Array
    best_fitness: Array
    mean_fitness: Array
    best_genotype: Array
    key: PRNGKey
    converged: Array


class SimulationHistory(NamedTuple):
    """History of simulation statistics over generations.

    Attributes:
        mean_fitness: Mean fitness per generation, shape (num_generations,)
        best_fitness: Best fitness per generation, shape (num_generations,)
        min_fitness: Min fitness per generation, shape (num_generations,)
        std_fitness: Std deviation of fitness per generation, shape (num_generations,)
    """

    mean_fitness: Array
    best_fitness: Array
    min_fitness: Array
    std_fitness: Array


class SimulationResult(NamedTuple):
    """Final results of a simulation run.

    Attributes:
        generations_to_success: Number of generations to evolve cluster-tracking
            (or max_generations if not successful)
        success: Whether cluster-tracking evolved before max_generations
        final_state: Final simulation state
        history: Statistics history over all generations
        config: Configuration used for the simulation
    """

    generations_to_success: int
    success: bool
    final_state: SimulationState
    history: SimulationHistory
    config: SimulationConfig


def init_simulation(key: PRNGKey, config: SimulationConfig) -> SimulationState:
    """Initialize simulation with random population.

    Args:
        key: JAX random key
        config: Simulation configuration

    Returns:
        Initial simulation state
    """
    key_pop, key_eval, key_next = jax.random.split(key, 3)

    # Create initial random population
    population = create_random_population(key_pop, config.genetic.population_size)

    # Evaluate initial population
    fitness_scores = evaluate_population(
        key_eval,
        population,
        config.environment,
        config.network,
        config.learning,
        config.fitness,
    )

    # Find best individual
    best_idx = jnp.argmax(fitness_scores)
    best_fitness = fitness_scores[best_idx]
    mean_fitness = jnp.mean(fitness_scores)
    best_genotype = population[best_idx]

    # Check if already converged (unlikely but possible)
    threshold = compute_success_threshold(config)
    converged = mean_fitness >= threshold

    return SimulationState(
        population=population,
        fitness_scores=fitness_scores,
        generation=jnp.array(0),
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
        best_genotype=best_genotype,
        key=key_next,
        converged=jnp.array(converged),
    )


def run_generation(state: SimulationState, config: SimulationConfig) -> SimulationState:
    """Execute one generation of evolution.

    Steps:
    1. Create next generation through selection, crossover, mutation
    2. Evaluate new population
    3. Update statistics
    4. Check termination condition

    Args:
        state: Current simulation state
        config: Simulation configuration

    Returns:
        Updated simulation state
    """
    key_ga, key_eval, key_next = jax.random.split(state.key, 3)

    # Create next generation (with elitism to preserve best individual)
    new_population = create_next_generation_with_elitism(
        key_ga, state.population, state.fitness_scores, config.genetic
    )

    # Evaluate new population
    new_fitness = evaluate_population(
        key_eval,
        new_population,
        config.environment,
        config.network,
        config.learning,
        config.fitness,
    )

    # Compute statistics
    best_idx = jnp.argmax(new_fitness)
    best_fitness = new_fitness[best_idx]
    mean_fitness = jnp.mean(new_fitness)
    best_genotype = new_population[best_idx]

    # Check convergence
    threshold = compute_success_threshold(config)
    converged = mean_fitness >= threshold

    return SimulationState(
        population=new_population,
        fitness_scores=new_fitness,
        generation=state.generation + 1,
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
        best_genotype=best_genotype,
        key=key_next,
        converged=jnp.array(converged),
    )


def run_simulation(
    key: PRNGKey,
    config: SimulationConfig,
    verbose: bool = True,
    progress_callback: callable = None,
) -> SimulationResult:
    """Run full simulation until convergence or max generations.

    This is the non-JIT version that allows for printing progress.

    Args:
        key: JAX random key
        config: Simulation configuration
        verbose: Whether to print progress
        progress_callback: Optional callback(generation, max_generations, mean_fitness, best_fitness)
            called after each generation for progress reporting

    Returns:
        SimulationResult with final state and history
    """
    # Initialize
    state = init_simulation(key, config)

    # Storage for history
    mean_fitness_history = [float(state.mean_fitness)]
    best_fitness_history = [float(state.best_fitness)]
    min_fitness_history = [float(jnp.min(state.fitness_scores))]
    std_fitness_history = [float(jnp.std(state.fitness_scores))]

    if verbose:
        optimal = compute_optimal_fitness(config.environment, config.fitness)
        sensory_only = compute_sensory_only_fitness(config.environment, config.fitness)
        threshold = compute_success_threshold(config)
        print(f"Optimal fitness: {optimal:.2f}")
        print(f"Sensory-only fitness: {sensory_only:.2f}")
        print(f"Success threshold: {threshold:.2f}")
        print(f"Initial mean fitness: {state.mean_fitness:.2f}")
        print("-" * 50)

    # JIT compile the generation runner (config captured in closure as constants)
    @jax.jit
    def jit_run_generation(state):
        return run_generation(state, config)

    # Evolution loop - always run to max_generations to collect full fitness history
    while state.generation < config.genetic.max_generations:
        state = jit_run_generation(state)

        # Record history
        mean_fitness_history.append(float(state.mean_fitness))
        best_fitness_history.append(float(state.best_fitness))
        min_fitness_history.append(float(jnp.min(state.fitness_scores)))
        std_fitness_history.append(float(jnp.std(state.fitness_scores)))

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(
                int(state.generation),
                config.genetic.max_generations,
                float(state.mean_fitness),
                float(state.best_fitness),
            )

        if verbose and (state.generation % 100 == 0 or state.converged):
            print(
                f"Gen {state.generation:4d}: "
                f"mean={state.mean_fitness:.2f}, "
                f"best={state.best_fitness:.2f}, "
                f"std={std_fitness_history[-1]:.2f}"
            )

    # Create history
    history = SimulationHistory(
        mean_fitness=jnp.array(mean_fitness_history),
        best_fitness=jnp.array(best_fitness_history),
        min_fitness=jnp.array(min_fitness_history),
        std_fitness=jnp.array(std_fitness_history),
    )

    # Determine success by scanning mean fitness history for first threshold crossing
    threshold = compute_success_threshold(config)
    mean_arr = jnp.array(mean_fitness_history)
    crossed = mean_arr >= threshold
    if jnp.any(crossed):
        success = True
        # Generation index = first index where mean crossed threshold
        generations = int(jnp.argmax(crossed))
    else:
        success = False
        generations = config.genetic.max_generations

    if verbose:
        print("-" * 50)
        if success:
            print(f"Cluster-tracking evolved in {generations} generations!")
        else:
            print(f"Did not converge within {config.genetic.max_generations} generations.")

    return SimulationResult(
        generations_to_success=generations,
        success=success,
        final_state=state,
        history=history,
        config=config,
    )


def run_simulation_jit(
    key: PRNGKey,
    config: SimulationConfig,
    max_generations: int | None = None,
) -> tuple[SimulationState, SimulationHistory]:
    """Run simulation with JIT compilation using lax.while_loop.

    This version is faster but cannot print progress.

    Args:
        key: JAX random key
        config: Simulation configuration
        max_generations: Override max generations (for JIT tracing)

    Returns:
        Tuple of (final_state, history)
    """
    if max_generations is None:
        max_generations = config.genetic.max_generations

    # Initialize
    initial_state = init_simulation(key, config)

    # Pre-allocate history arrays
    history_shape = (max_generations + 1,)
    initial_history = SimulationHistory(
        mean_fitness=jnp.zeros(history_shape).at[0].set(initial_state.mean_fitness),
        best_fitness=jnp.zeros(history_shape).at[0].set(initial_state.best_fitness),
        min_fitness=jnp.zeros(history_shape).at[0].set(jnp.min(initial_state.fitness_scores)),
        std_fitness=jnp.zeros(history_shape).at[0].set(jnp.std(initial_state.fitness_scores)),
    )

    def cond_fn(carry):
        """Continue while not converged and under max generations."""
        state, _ = carry
        return ~state.converged & (state.generation < max_generations)

    def body_fn(carry):
        """Execute one generation and update history."""
        state, history = carry

        # Run generation
        new_state = run_generation(state, config)

        # Update history
        gen = new_state.generation
        new_history = SimulationHistory(
            mean_fitness=history.mean_fitness.at[gen].set(new_state.mean_fitness),
            best_fitness=history.best_fitness.at[gen].set(new_state.best_fitness),
            min_fitness=history.min_fitness.at[gen].set(jnp.min(new_state.fitness_scores)),
            std_fitness=history.std_fitness.at[gen].set(jnp.std(new_state.fitness_scores)),
        )

        return new_state, new_history

    # Run evolution loop
    final_state, final_history = jax.lax.while_loop(
        cond_fn, body_fn, (initial_state, initial_history)
    )

    # Trim history to actual generations
    actual_gens = final_state.generation + 1
    trimmed_history = SimulationHistory(
        mean_fitness=final_history.mean_fitness[:actual_gens],
        best_fitness=final_history.best_fitness[:actual_gens],
        min_fitness=final_history.min_fitness[:actual_gens],
        std_fitness=final_history.std_fitness[:actual_gens],
    )

    return final_state, trimmed_history


# ============================================================================
# Batch Simulation for Experiments
# ============================================================================


def run_multiple_simulations(
    key: PRNGKey,
    config: SimulationConfig,
    num_runs: int,
    verbose: bool = True,
) -> list[SimulationResult]:
    """Run multiple independent simulation runs.

    Args:
        key: JAX random key
        config: Simulation configuration
        num_runs: Number of independent runs
        verbose: Whether to print progress

    Returns:
        List of SimulationResults
    """
    keys = jax.random.split(key, num_runs)
    results = []

    for i, run_key in enumerate(keys):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Run {i+1}/{num_runs}")
            print(f"{'='*50}")

        result = run_simulation(run_key, config, verbose=verbose)
        results.append(result)

    return results


def run_experiment_condition(
    key: PRNGKey,
    base_config: SimulationConfig,
    clump_scale: int,
    sensory_accuracy: float,
    num_runs: int,
    verbose: bool = False,
) -> dict:
    """Run multiple simulations for one experimental condition.

    Args:
        key: JAX random key
        base_config: Base simulation configuration
        clump_scale: Clump scale for this condition
        sensory_accuracy: Sensory accuracy for this condition
        num_runs: Number of runs for this condition
        verbose: Whether to print progress

    Returns:
        Dictionary with condition results
    """
    # Update config for this condition
    config = base_config.with_updates(
        **{
            "environment.clump_scale": clump_scale,
            "environment.sensory_accuracy": sensory_accuracy,
        }
    )

    # Run simulations
    results = run_multiple_simulations(key, config, num_runs, verbose=verbose)

    # Aggregate results
    generations = [r.generations_to_success for r in results]
    successes = [r.success for r in results]

    return {
        "clump_scale": clump_scale,
        "sensory_accuracy": sensory_accuracy,
        "num_runs": num_runs,
        "generations": generations,
        "mean_generations": sum(generations) / len(generations),
        "success_rate": sum(successes) / len(successes),
        "results": results,
    }


def run_full_experiment(
    key: PRNGKey,
    config: SimulationConfig,
    verbose: bool = True,
) -> list[dict]:
    """Run the full experimental grid as in the paper.

    Args:
        key: JAX random key
        config: Simulation configuration (uses experiment settings)
        verbose: Whether to print progress

    Returns:
        List of condition results
    """
    clump_scales = config.experiment.clump_scales
    accuracies = config.experiment.sensory_accuracies
    runs_per = config.experiment.runs_per_condition

    total_conditions = len(clump_scales) * len(accuracies)
    condition_idx = 0

    all_results = []
    keys = jax.random.split(key, total_conditions)

    for clump_scale in clump_scales:
        for accuracy in accuracies:
            if verbose:
                print(f"\n{'#'*60}")
                print(f"Condition {condition_idx+1}/{total_conditions}")
                print(f"Clump scale: {clump_scale}, Accuracy: {accuracy:.0%}")
                print(f"{'#'*60}")

            result = run_experiment_condition(
                keys[condition_idx],
                config,
                clump_scale,
                accuracy,
                runs_per,
                verbose=verbose,
            )

            all_results.append(result)
            condition_idx += 1

            if verbose:
                print(f"Mean generations: {result['mean_generations']:.1f}")
                print(f"Success rate: {result['success_rate']:.0%}")

    return all_results

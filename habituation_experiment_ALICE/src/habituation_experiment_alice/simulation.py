"""Main simulation loop for the ALICE threat discrimination evolution.

This module handles:
- Running individual generations (tournament selection + evaluation)
- Managing the evolutionary loop
- Tracking statistics and history
- Generating checkpoint traces for visualization
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.random import PRNGKey

from .config import SimulationConfig
from .evaluation import (
    PopulationEvalSummary,
    evaluate_agent_detailed,
    evaluate_population,
)
from .genetics import (
    compute_genotype_diversity,
    compute_genotype_length,
    compute_pop_learnable_fractions,
    create_random_population,
    decode_genotype,
)
from .selection import create_next_generation_tournament


class SimulationState(NamedTuple):
    """State of the simulation at a given generation."""

    population: Array
    fitness_scores: Array
    generation: Array
    best_fitness: Array
    mean_fitness: Array
    best_genotype: Array
    key: PRNGKey
    num_survivors: Array  # how many survived phase 1


class GenerationStats(NamedTuple):
    """Per-generation statistics computed during evaluation (inside JIT)."""

    phase1_mean_survival_frac: Array
    phase2_mean_health: Array
    num_survived_phase1: Array


class SimulationHistory(NamedTuple):
    """History of simulation statistics over generations."""

    # Core fitness metrics
    mean_fitness: Array
    best_fitness: Array
    min_fitness: Array
    std_fitness: Array
    num_survivors: Array
    # Phase-specific metrics
    phase1_mean_survival_frac: Array
    phase2_mean_health: Array
    num_survived_phase1: Array
    # Genetics metrics
    genotype_diversity: Array
    best_weights: Array         # (num_gens, num_connections)
    best_biases: Array          # (num_gens, num_biases)
    best_learnable_mask: Array  # (num_gens, num_params)
    pop_learnable_frac: Array   # (num_gens, num_params)


class SimulationResult(NamedTuple):
    """Final results of a simulation run."""

    final_state: SimulationState
    history: SimulationHistory
    config: SimulationConfig
    traces: dict  # {generation_number: AgentTrace}


def _compute_gen_stats(
    eval_summary: PopulationEvalSummary,
    phase1_lifetime: int,
) -> GenerationStats:
    """Compute per-generation phase statistics from evaluation summary."""
    num_survived = jnp.sum(eval_summary.survived_phase1)

    # Mean survival fraction in phase 1
    phase1_mean_surv = jnp.mean(eval_summary.phase1_survival_time) / phase1_lifetime

    # Mean phase 2 health (only for survivors)
    phase2_health_survivors = jnp.where(
        eval_summary.survived_phase1,
        eval_summary.phase2_final_health,
        0.0,
    )
    phase2_mean_health = jnp.where(
        num_survived > 0,
        jnp.sum(phase2_health_survivors) / jnp.maximum(num_survived, 1),
        0.0,
    )

    return GenerationStats(
        phase1_mean_survival_frac=phase1_mean_surv,
        phase2_mean_health=phase2_mean_health,
        num_survived_phase1=num_survived,
    )


def init_simulation(
    key: PRNGKey, config: SimulationConfig
) -> tuple[SimulationState, GenerationStats]:
    """Initialize simulation with random population and evaluate."""
    key_pop, key_eval, key_next = jax.random.split(key, 3)

    genotype_length = compute_genotype_length(config.network)
    population = create_random_population(
        key_pop, config.genetic.population_size, genotype_length
    )

    # Evaluate initial population
    eval_summary = evaluate_population(key_eval, population, config)

    # Statistics
    best_idx = jnp.argmax(eval_summary.fitness)
    best_fitness = eval_summary.fitness[best_idx]
    mean_fitness = jnp.mean(eval_summary.fitness)
    best_genotype = population[best_idx]
    num_survivors = jnp.sum(eval_summary.fitness > 0)

    state = SimulationState(
        population=population,
        fitness_scores=eval_summary.fitness,
        generation=jnp.array(0),
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
        best_genotype=best_genotype,
        key=key_next,
        num_survivors=num_survivors,
    )

    gen_stats = _compute_gen_stats(eval_summary, config.environment.phase1_lifetime)

    return state, gen_stats


def run_generation(
    state: SimulationState, config: SimulationConfig
) -> tuple[SimulationState, GenerationStats]:
    """Execute one generation of evolution.

    Steps:
    1. Tournament selection to create next generation
    2. Evaluate new population (two-phase)
    3. Update statistics
    """
    key_select, key_eval, key_next = jax.random.split(state.key, 3)

    # Tournament selection
    new_population = create_next_generation_tournament(
        key_select, state.population, state.fitness_scores, config.genetic
    )

    # Evaluate new population
    eval_summary = evaluate_population(key_eval, new_population, config)

    # Statistics
    best_idx = jnp.argmax(eval_summary.fitness)
    best_fitness = eval_summary.fitness[best_idx]
    mean_fitness = jnp.mean(eval_summary.fitness)
    best_genotype = new_population[best_idx]
    num_survivors = jnp.sum(eval_summary.fitness > 0)

    new_state = SimulationState(
        population=new_population,
        fitness_scores=eval_summary.fitness,
        generation=state.generation + 1,
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
        best_genotype=best_genotype,
        key=key_next,
        num_survivors=num_survivors,
    )

    gen_stats = _compute_gen_stats(eval_summary, config.environment.phase1_lifetime)

    return new_state, gen_stats


def _collect_outside_jit_stats(state, config):
    """Collect statistics that are computed outside JIT (Python-side)."""
    # Decode best genotype
    best_params = decode_genotype(state.best_genotype, config.network)

    # Genotype diversity
    diversity = compute_genotype_diversity(state.population)

    # Population learnable fractions
    pop_learn_frac = compute_pop_learnable_fractions(state.population, config.network)

    return {
        "best_weights": np.array(best_params.weights),
        "best_biases": np.array(best_params.biases),
        "best_learnable_mask": np.array(best_params.learnable_mask),
        "diversity": diversity,
        "pop_learnable_frac": pop_learn_frac,
    }


def run_simulation(
    key: PRNGKey,
    config: SimulationConfig,
    verbose: bool = True,
    progress_callback: callable = None,
    trace_interval: int | None = None,
) -> SimulationResult:
    """Run full simulation to max_generations.

    Args:
        key: JAX random key
        config: Simulation configuration
        verbose: Whether to print progress
        progress_callback: Optional callback(generation, max_gen, mean_fitness, best_fitness)
        trace_interval: Generations between checkpoint traces. None = auto (~10 checkpoints).

    Returns:
        SimulationResult with final state, history, config, and checkpoint traces
    """
    max_gen = config.genetic.max_generations

    # Determine trace checkpoint generations
    if trace_interval is None:
        trace_interval = max(1, max_gen // 10)
    trace_gens = {0, max_gen}
    for g in range(trace_interval, max_gen, trace_interval):
        trace_gens.add(g)

    # Initialize
    state, init_stats = init_simulation(key, config)

    # History storage - core metrics
    mean_fitness_history = [float(state.mean_fitness)]
    best_fitness_history = [float(state.best_fitness)]
    min_fitness_history = [float(jnp.min(state.fitness_scores))]
    std_fitness_history = [float(jnp.std(state.fitness_scores))]
    survivors_history = [int(state.num_survivors)]

    # History storage - phase-specific
    phase1_surv_frac_history = [float(init_stats.phase1_mean_survival_frac)]
    phase2_health_history = [float(init_stats.phase2_mean_health)]
    num_survived_p1_history = [int(init_stats.num_survived_phase1)]

    # History storage - genetics (computed outside JIT)
    outside_stats = _collect_outside_jit_stats(state, config)
    diversity_history = [outside_stats["diversity"]]
    best_weights_history = [outside_stats["best_weights"]]
    best_biases_history = [outside_stats["best_biases"]]
    best_learnable_history = [outside_stats["best_learnable_mask"]]
    pop_learnable_history = [outside_stats["pop_learnable_frac"]]

    # Checkpoint traces
    traces = {}

    if verbose:
        pop = config.genetic.population_size
        print(f"Population size: {pop}")
        print(f"Phase 1 lifetime: {config.environment.phase1_lifetime}")
        print(f"Phase 2 lifetime: {config.environment.phase2_lifetime}")
        print(f"Initial mean fitness: {state.mean_fitness:.2f}")
        print(f"Initial survivors: {state.num_survivors}/{pop}")
        print("-" * 50)

    # JIT compile functions
    @jax.jit
    def jit_run_generation(state):
        return run_generation(state, config)

    @jax.jit
    def jit_trace(genotype, trace_key):
        return evaluate_agent_detailed(genotype, trace_key, config)

    # Initial trace
    if 0 in trace_gens:
        trace_key = jax.random.PRNGKey(config.seed * 1000)
        traces[0] = jit_trace(state.best_genotype, trace_key)
        if verbose:
            print("  Saved checkpoint trace for gen 0")

    # Evolution loop
    while state.generation < max_gen:
        state, gen_stats = jit_run_generation(state)
        gen_num = int(state.generation)

        # Record core history
        mean_fitness_history.append(float(state.mean_fitness))
        best_fitness_history.append(float(state.best_fitness))
        min_fitness_history.append(float(jnp.min(state.fitness_scores)))
        std_fitness_history.append(float(jnp.std(state.fitness_scores)))
        survivors_history.append(int(state.num_survivors))

        # Record phase-specific history
        phase1_surv_frac_history.append(float(gen_stats.phase1_mean_survival_frac))
        phase2_health_history.append(float(gen_stats.phase2_mean_health))
        num_survived_p1_history.append(int(gen_stats.num_survived_phase1))

        # Collect outside-JIT stats
        outside_stats = _collect_outside_jit_stats(state, config)
        diversity_history.append(outside_stats["diversity"])
        best_weights_history.append(outside_stats["best_weights"])
        best_biases_history.append(outside_stats["best_biases"])
        best_learnable_history.append(outside_stats["best_learnable_mask"])
        pop_learnable_history.append(outside_stats["pop_learnable_frac"])

        # Checkpoint trace
        if gen_num in trace_gens:
            trace_key = jax.random.PRNGKey(config.seed * 1000 + gen_num)
            traces[gen_num] = jit_trace(state.best_genotype, trace_key)
            if verbose:
                print(f"  Saved checkpoint trace for gen {gen_num}")

        if progress_callback is not None:
            progress_callback(
                gen_num,
                max_gen,
                float(state.mean_fitness),
                float(state.best_fitness),
            )

        if verbose and (gen_num % 100 == 0):
            print(
                f"Gen {gen_num:4d}: "
                f"mean={state.mean_fitness:.2f}, "
                f"best={state.best_fitness:.2f}, "
                f"survivors={state.num_survivors}/{config.genetic.population_size}, "
                f"std={std_fitness_history[-1]:.2f}, "
                f"diversity={diversity_history[-1]:.3f}"
            )

    # Build history
    history = SimulationHistory(
        mean_fitness=np.array(mean_fitness_history),
        best_fitness=np.array(best_fitness_history),
        min_fitness=np.array(min_fitness_history),
        std_fitness=np.array(std_fitness_history),
        num_survivors=np.array(survivors_history),
        phase1_mean_survival_frac=np.array(phase1_surv_frac_history),
        phase2_mean_health=np.array(phase2_health_history),
        num_survived_phase1=np.array(num_survived_p1_history),
        genotype_diversity=np.array(diversity_history),
        best_weights=np.stack(best_weights_history),
        best_biases=np.stack(best_biases_history),
        best_learnable_mask=np.stack(best_learnable_history),
        pop_learnable_frac=np.stack(pop_learnable_history),
    )

    if verbose:
        print("-" * 50)
        print(f"Final mean fitness: {state.mean_fitness:.2f}")
        print(f"Final best fitness: {state.best_fitness:.2f}")
        print(f"Final survivors: {state.num_survivors}/{config.genetic.population_size}")

    return SimulationResult(
        final_state=state,
        history=history,
        config=config,
        traces=traces,
    )


def run_multiple_simulations(
    key: PRNGKey,
    config: SimulationConfig,
    num_runs: int,
    verbose: bool = True,
) -> list[SimulationResult]:
    """Run multiple independent simulation runs."""
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

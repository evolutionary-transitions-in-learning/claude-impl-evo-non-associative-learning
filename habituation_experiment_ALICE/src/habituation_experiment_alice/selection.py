"""Tournament selection for the ALICE threat discrimination simulation.

Implements N-tournament selection where N = population size:
- Each tournament selects tournament_size random individuals
- The highest-fitness individual is the winner
- The lowest-fitness individual is the loser
- The loser is replaced by either:
  - A copy of the winner (probability = 1 - crossover_rate)
  - crossover(winner, loser) (probability = crossover_rate)
- Mutation is always applied to the replacement

All N tournaments within a generation use the ORIGINAL fitness scores
(not updated mid-generation), while the population array evolves
through the sequential tournaments.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from .config import GeneticConfig, GenotypeMode, NetworkConfig
from .genetics import (
    gaussian_mutation,
    point_mutation,
    two_point_crossover,
    uniform_crossover,
)


def _binary_tournament_ops(
    key_cross: PRNGKey,
    key_mutate: PRNGKey,
    winner_genotype: Array,
    loser_genotype: Array,
    do_crossover: Array,
    mutation_rate: float,
    **kwargs,
) -> Array:
    """Apply binary genetic operators: two-point crossover + point mutation."""
    offspring, _ = two_point_crossover(key_cross, winner_genotype, loser_genotype)
    replacement = jnp.where(do_crossover, offspring, winner_genotype)
    replacement = point_mutation(key_mutate, replacement, mutation_rate)
    return replacement


def _continuous_tournament_ops(
    key_cross: PRNGKey,
    key_mutate: PRNGKey,
    winner_genotype: Array,
    loser_genotype: Array,
    do_crossover: Array,
    mutation_rate: float,
    mutation_std: float,
    net_config: NetworkConfig,
) -> Array:
    """Apply continuous genetic operators: uniform crossover + Gaussian mutation."""
    offspring, _ = uniform_crossover(key_cross, winner_genotype, loser_genotype)
    replacement = jnp.where(do_crossover, offspring, winner_genotype)
    replacement = gaussian_mutation(key_mutate, replacement, mutation_rate, mutation_std, net_config)
    return replacement


def single_tournament_binary(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    crossover_rate: float,
    mutation_rate: float,
    tournament_size: int = 2,
) -> Array:
    """Execute a single tournament with binary genetic operators."""
    key_select, key_cross_choice, key_cross, key_mutate = jax.random.split(key, 4)

    pop_size = population.shape[0]
    candidate_indices = jax.random.randint(
        key_select, (tournament_size,), 0, pop_size
    )
    candidate_fitness = fitness_scores[candidate_indices]

    winner_pos = jnp.argmax(candidate_fitness)
    loser_pos = jnp.argmin(candidate_fitness)
    winner_idx = candidate_indices[winner_pos]
    loser_idx = candidate_indices[loser_pos]

    winner_genotype = population[winner_idx]
    loser_genotype = population[loser_idx]

    do_crossover = jax.random.bernoulli(key_cross_choice, p=crossover_rate)

    replacement = _binary_tournament_ops(
        key_cross, key_mutate,
        winner_genotype, loser_genotype,
        do_crossover, mutation_rate,
    )

    return population.at[loser_idx].set(replacement)


def single_tournament_continuous(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    crossover_rate: float,
    mutation_rate: float,
    mutation_std: float,
    net_config: NetworkConfig,
    tournament_size: int = 2,
) -> Array:
    """Execute a single tournament with continuous genetic operators."""
    key_select, key_cross_choice, key_cross, key_mutate = jax.random.split(key, 4)

    pop_size = population.shape[0]
    candidate_indices = jax.random.randint(
        key_select, (tournament_size,), 0, pop_size
    )
    candidate_fitness = fitness_scores[candidate_indices]

    winner_pos = jnp.argmax(candidate_fitness)
    loser_pos = jnp.argmin(candidate_fitness)
    winner_idx = candidate_indices[winner_pos]
    loser_idx = candidate_indices[loser_pos]

    winner_genotype = population[winner_idx]
    loser_genotype = population[loser_idx]

    do_crossover = jax.random.bernoulli(key_cross_choice, p=crossover_rate)

    replacement = _continuous_tournament_ops(
        key_cross, key_mutate,
        winner_genotype, loser_genotype,
        do_crossover, mutation_rate,
        mutation_std, net_config,
    )

    return population.at[loser_idx].set(replacement)


def create_next_generation_tournament(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    config: GeneticConfig,
    net_config: NetworkConfig | None = None,
) -> Array:
    """Create next generation using N tournament rounds.

    Runs pop_size tournaments sequentially using lax.scan.
    The population evolves through tournaments, but fitness scores
    remain fixed (from start of generation).

    Args:
        key: JAX random key
        population: Current population, shape (pop_size, genotype_length)
        fitness_scores: Fitness scores, shape (pop_size,)
        config: Genetic algorithm configuration
        net_config: Network configuration (required for continuous mode)

    Returns:
        New population after all tournaments
    """
    pop_size = config.population_size
    keys = jax.random.split(key, pop_size)

    if config.genotype_mode == GenotypeMode.CONTINUOUS:
        def tournament_step(pop, tournament_key):
            new_pop = single_tournament_continuous(
                tournament_key,
                pop,
                fitness_scores,
                config.crossover_rate,
                config.mutation_rate,
                config.mutation_std,
                net_config,
                config.tournament_size,
            )
            return new_pop, None
    else:
        def tournament_step(pop, tournament_key):
            new_pop = single_tournament_binary(
                tournament_key,
                pop,
                fitness_scores,
                config.crossover_rate,
                config.mutation_rate,
                config.tournament_size,
            )
            return new_pop, None

    final_pop, _ = jax.lax.scan(tournament_step, population, keys)
    return final_pop

"""Tournament selection for the ALICE threat discrimination simulation.

Implements N-tournament selection where N = population size:
- Each tournament selects two random distinct individuals
- The higher-fitness individual is the winner
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

from .config import GeneticConfig
from .genetics import point_mutation, two_point_crossover


def single_tournament(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    crossover_rate: float,
    mutation_rate: float,
) -> Array:
    """Execute a single tournament between two random individuals.

    Args:
        key: JAX random key
        population: Current population, shape (pop_size, genotype_length)
        fitness_scores: ORIGINAL fitness scores, shape (pop_size,)
        crossover_rate: Probability of crossover vs copying winner
        mutation_rate: Per-bit mutation rate

    Returns:
        Updated population with loser replaced
    """
    key_pair, key_offset, key_cross_choice, key_cross, key_mutate = jax.random.split(key, 5)

    pop_size = population.shape[0]

    # Select two random distinct indices
    idx1 = jax.random.randint(key_pair, (), 0, pop_size)
    offset = jax.random.randint(key_offset, (), 1, pop_size)
    idx2 = (idx1 + offset) % pop_size

    fit1 = fitness_scores[idx1]
    fit2 = fitness_scores[idx2]

    # Determine winner and loser
    winner_is_1 = fit1 >= fit2
    winner_idx = jnp.where(winner_is_1, idx1, idx2)
    loser_idx = jnp.where(winner_is_1, idx2, idx1)
    winner_genotype = population[winner_idx]
    loser_genotype = population[loser_idx]

    # Decide: crossover or copy winner
    do_crossover = jax.random.bernoulli(key_cross_choice, p=crossover_rate)

    # If crossover: crossover(winner, loser), take first offspring
    offspring, _ = two_point_crossover(key_cross, winner_genotype, loser_genotype)
    replacement = jnp.where(do_crossover, offspring, winner_genotype)

    # Always apply mutation
    replacement = point_mutation(key_mutate, replacement, mutation_rate)

    # Replace loser with replacement
    new_population = population.at[loser_idx].set(replacement)

    return new_population


def create_next_generation_tournament(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    config: GeneticConfig,
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

    Returns:
        New population after all tournaments
    """
    pop_size = config.population_size
    keys = jax.random.split(key, pop_size)

    def tournament_step(pop, tournament_key):
        new_pop = single_tournament(
            tournament_key,
            pop,
            fitness_scores,  # captured from closure - constant for generation
            config.crossover_rate,
            config.mutation_rate,
        )
        return new_pop, None

    final_pop, _ = jax.lax.scan(tournament_step, population, keys)
    return final_pop

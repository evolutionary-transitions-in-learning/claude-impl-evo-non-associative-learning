"""Tests for tournament selection module."""

import jax
import jax.numpy as jnp
import pytest

from habituation_experiment_alice.config import GeneticConfig
from habituation_experiment_alice.selection import (
    create_next_generation_tournament,
    single_tournament,
)


class TestSingleTournament:
    def test_population_shape_preserved(self):
        key = jax.random.PRNGKey(0)
        pop = jnp.zeros((10, 81), dtype=jnp.int32)
        fitness = jnp.arange(10, dtype=jnp.float32)
        new_pop = single_tournament(key, pop, fitness, 0.7, 0.01)
        assert new_pop.shape == pop.shape


class TestTournamentSelection:
    def test_population_shape_preserved(self):
        key = jax.random.PRNGKey(0)
        pop = jax.random.bernoulli(key, shape=(20, 81)).astype(jnp.int32)
        fitness = jnp.arange(20, dtype=jnp.float32)
        config = GeneticConfig(population_size=20)
        new_pop = create_next_generation_tournament(key, pop, fitness, config)
        assert new_pop.shape == (20, 81)

    def test_high_fitness_survives(self):
        """Higher fitness individuals should be more likely to have their genes persist."""
        key = jax.random.PRNGKey(42)
        pop_size = 50
        genotype_length = 81

        # Create population where individual 0 has all 0s (identifiable)
        # and high fitness, rest have all 1s and low fitness
        pop = jnp.ones((pop_size, genotype_length), dtype=jnp.int32)
        pop = pop.at[0].set(jnp.zeros(genotype_length, dtype=jnp.int32))
        fitness = jnp.zeros(pop_size)
        fitness = fitness.at[0].set(100.0)

        config = GeneticConfig(population_size=pop_size, mutation_rate=0.0, crossover_rate=0.0)
        new_pop = create_next_generation_tournament(key, pop, fitness, config)

        # With mutation_rate=0 and crossover_rate=0, winner is copied.
        # The best individual (idx 0) should spread its genes.
        # Count how many individuals are mostly 0s
        zero_fraction = 1.0 - jnp.mean(new_pop, axis=1)
        mostly_zeros = jnp.sum(zero_fraction > 0.9)
        # Should have spread to at least a few individuals
        assert int(mostly_zeros) > 1

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(0)
        pop = jax.random.bernoulli(key, shape=(10, 81)).astype(jnp.int32)
        fitness = jnp.arange(10, dtype=jnp.float32)
        config = GeneticConfig(population_size=10)

        @jax.jit
        def run(key, pop, fitness):
            return create_next_generation_tournament(key, pop, fitness, config)

        new_pop = run(key, pop, fitness)
        assert new_pop.shape == (10, 81)

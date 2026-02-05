"""Tests for evaluation module."""

import jax
import jax.numpy as jnp
import pytest

from habituation_experiment_alice.config import SimulationConfig
from habituation_experiment_alice.evaluation import (
    evaluate_creature,
    evaluate_population,
)
from habituation_experiment_alice.environment import (
    generate_phase1_environment,
    generate_phase2_environment,
)
from habituation_experiment_alice.genetics import create_random_genotype, create_random_population


class TestEvaluateCreature:
    def test_fitness_non_negative(self):
        key = jax.random.PRNGKey(0)
        config = SimulationConfig(
            environment=config_small_env(),
        )
        k1, k2, k3 = jax.random.split(key, 3)
        genotype = create_random_genotype(k1)
        p1_env = generate_phase1_environment(k2, config)
        p2_env = generate_phase2_environment(k3, config)
        result = evaluate_creature(genotype, p1_env, p2_env, config)
        assert float(result.fitness) >= 0.0

    def test_fitness_shape(self):
        key = jax.random.PRNGKey(0)
        config = SimulationConfig(
            environment=config_small_env(),
        )
        k1, k2, k3 = jax.random.split(key, 3)
        genotype = create_random_genotype(k1)
        p1_env = generate_phase1_environment(k2, config)
        p2_env = generate_phase2_environment(k3, config)
        result = evaluate_creature(genotype, p1_env, p2_env, config)
        assert result.fitness.shape == ()


class TestEvaluatePopulation:
    def test_fitness_shape(self):
        key = jax.random.PRNGKey(0)
        config = SimulationConfig(
            environment=config_small_env(),
            genetic=config_small_genetic(),
        )
        k1, k2 = jax.random.split(key)
        pop = create_random_population(k1, config.genetic.population_size)
        result = evaluate_population(k2, pop, config)
        assert result.fitness.shape == (config.genetic.population_size,)

    def test_fitness_all_non_negative(self):
        key = jax.random.PRNGKey(42)
        config = SimulationConfig(
            environment=config_small_env(),
            genetic=config_small_genetic(),
        )
        k1, k2 = jax.random.split(key)
        pop = create_random_population(k1, config.genetic.population_size)
        result = evaluate_population(k2, pop, config)
        assert jnp.all(result.fitness >= 0.0)

    def test_no_nans(self):
        key = jax.random.PRNGKey(123)
        config = SimulationConfig(
            environment=config_small_env(),
            genetic=config_small_genetic(),
        )
        k1, k2 = jax.random.split(key)
        pop = create_random_population(k1, config.genetic.population_size)
        result = evaluate_population(k2, pop, config)
        assert not jnp.any(jnp.isnan(result.fitness))

    def test_summary_fields(self):
        key = jax.random.PRNGKey(0)
        config = SimulationConfig(
            environment=config_small_env(),
            genetic=config_small_genetic(),
        )
        k1, k2 = jax.random.split(key)
        pop = create_random_population(k1, config.genetic.population_size)
        result = evaluate_population(k2, pop, config)
        assert result.survived_phase1.shape == (config.genetic.population_size,)
        assert result.phase1_survival_time.shape == (config.genetic.population_size,)
        assert result.phase2_final_health.shape == (config.genetic.population_size,)


# Helpers for smaller test configs
def config_small_env():
    from habituation_experiment_alice.config import EnvironmentConfig
    return EnvironmentConfig(phase1_lifetime=50, phase2_lifetime=100, clump_scale=5)


def config_small_genetic():
    from habituation_experiment_alice.config import GeneticConfig
    return GeneticConfig(population_size=10, max_generations=5)

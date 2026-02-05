"""Tests for simulation module (integration tests)."""

import jax
import jax.numpy as jnp
import pytest

from habituation_experiment_alice.config import (
    EnvironmentConfig,
    GeneticConfig,
    SimulationConfig,
)
from habituation_experiment_alice.simulation import (
    init_simulation,
    run_generation,
    run_simulation,
)


def small_config():
    return SimulationConfig(
        environment=EnvironmentConfig(
            phase1_lifetime=50, phase2_lifetime=100, clump_scale=5
        ),
        genetic=GeneticConfig(population_size=10, max_generations=5),
    )


class TestInitSimulation:
    def test_population_shape(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        state, gen_stats = init_simulation(key, config)
        assert state.population.shape == (10, 81)

    def test_fitness_shape(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        state, gen_stats = init_simulation(key, config)
        assert state.fitness_scores.shape == (10,)

    def test_generation_zero(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        state, gen_stats = init_simulation(key, config)
        assert int(state.generation) == 0

    def test_gen_stats_returned(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        state, gen_stats = init_simulation(key, config)
        assert gen_stats.phase1_mean_survival_frac.shape == ()
        assert gen_stats.phase2_mean_health.shape == ()


class TestRunGeneration:
    def test_generation_increments(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        state, _ = init_simulation(key, config)
        state, gen_stats = run_generation(state, config)
        assert int(state.generation) == 1

    def test_population_shape_preserved(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        state, _ = init_simulation(key, config)
        state, gen_stats = run_generation(state, config)
        assert state.population.shape == (10, 81)


class TestRunSimulation:
    def test_completes(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        result = run_simulation(key, config, verbose=False)
        assert int(result.final_state.generation) == 5

    def test_history_length(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        result = run_simulation(key, config, verbose=False)
        # History has initial + max_generations entries
        assert len(result.history.mean_fitness) == 6

    def test_no_nans_in_history(self):
        key = jax.random.PRNGKey(0)
        config = small_config()
        result = run_simulation(key, config, verbose=False)
        assert not jnp.any(jnp.isnan(result.history.mean_fitness))
        assert not jnp.any(jnp.isnan(result.history.best_fitness))

    def test_reproducibility(self):
        config = small_config()
        r1 = run_simulation(jax.random.PRNGKey(42), config, verbose=False)
        r2 = run_simulation(jax.random.PRNGKey(42), config, verbose=False)
        assert jnp.allclose(r1.history.mean_fitness, r2.history.mean_fitness)

    def test_different_seeds_differ(self):
        config = small_config()
        r1 = run_simulation(jax.random.PRNGKey(0), config, verbose=False)
        r2 = run_simulation(jax.random.PRNGKey(999), config, verbose=False)
        assert not jnp.allclose(r1.history.mean_fitness, r2.history.mean_fitness)

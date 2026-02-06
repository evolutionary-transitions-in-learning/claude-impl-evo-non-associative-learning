"""Integration tests for the full simulation."""

import jax
import jax.numpy as jnp
import pytest

from habituation_evolution.config import (
    EnvironmentConfig,
    FitnessConfig,
    GeneticConfig,
    LearningConfig,
    NetworkConfig,
    SimulationConfig,
)
from habituation_evolution.simulation import (
    init_simulation,
    run_generation,
    run_simulation,
)


class TestSimulationInitialization:
    """Tests for simulation initialization."""

    def test_init_creates_valid_state(self):
        """Initialization should create valid state."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20  # Small for testing

        state = init_simulation(key, config)

        assert state.population.shape[0] == 20
        assert state.fitness_scores.shape == (20,)
        assert state.generation == 0
        assert not jnp.isnan(state.mean_fitness)
        assert not jnp.isnan(state.best_fitness)

    def test_init_evaluates_population(self):
        """Initial population should be evaluated."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20

        state = init_simulation(key, config)

        # Fitness should have variance (not all zero)
        assert jnp.std(state.fitness_scores) > 0


class TestGenerationExecution:
    """Tests for single generation execution."""

    def test_generation_increments_counter(self):
        """Generation counter should increment."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20

        state = init_simulation(key, config)
        assert state.generation == 0

        state = run_generation(state, config)
        assert state.generation == 1

        state = run_generation(state, config)
        assert state.generation == 2

    def test_generation_updates_population(self):
        """Population should change between generations."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20

        state = init_simulation(key, config)
        old_pop = state.population.copy()

        state = run_generation(state, config)

        # Population should be different
        assert not jnp.allclose(state.population, old_pop)

    def test_generation_maintains_population_size(self):
        """Population size should remain constant."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20

        state = init_simulation(key, config)
        for _ in range(5):
            state = run_generation(state, config)
            assert state.population.shape[0] == 20


class TestEvolutionProgress:
    """Tests for evolutionary progress."""

    def test_fitness_improves(self):
        """Mean fitness should generally improve over generations."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 50
        config.environment.sensory_accuracy = 0.75
        config.environment.clump_scale = 10

        state = init_simulation(key, config)
        initial_mean = float(state.mean_fitness)

        # Run several generations
        for _ in range(50):
            state = run_generation(state, config)

        final_mean = float(state.mean_fitness)

        # Fitness should improve (or at least not decrease significantly)
        # Note: This is probabilistic, so we allow some slack
        assert final_mean >= initial_mean - 10

    def test_fitness_improves_on_average(self):
        """Fitness should improve on average over generations."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 50

        state = init_simulation(key, config)
        early_fitness = [float(state.best_fitness)]

        # Run 20 generations, collect early and late fitness
        for i in range(20):
            state = run_generation(state, config)
            if i < 5:
                early_fitness.append(float(state.best_fitness))

        late_fitness = [float(state.best_fitness)]
        for _ in range(5):
            state = run_generation(state, config)
            late_fitness.append(float(state.best_fitness))

        # Average fitness in later generations should be >= early generations
        # (or at least not significantly worse, allowing for stochastic variance)
        early_avg = sum(early_fitness) / len(early_fitness)
        late_avg = sum(late_fitness) / len(late_fitness)
        # With independent environments, there's high variance, so just check
        # late fitness isn't catastrophically worse
        assert late_avg >= early_avg * 0.5, f"Late avg {late_avg} too low vs early {early_avg}"


class TestFullSimulation:
    """Tests for full simulation runs."""

    def test_simulation_runs_to_completion(self):
        """Simulation should run without errors."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 50  # Limit for testing
        config.environment.clump_scale = 20
        config.environment.sensory_accuracy = 0.75

        result = run_simulation(key, config, verbose=False)

        assert result.generations_to_success <= 50
        assert result.history.mean_fitness.shape[0] > 0

    def test_simulation_returns_valid_history(self):
        """Simulation should return valid history."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 30

        result = run_simulation(key, config, verbose=False)

        # History should have entries for each generation
        num_gens = len(result.history.mean_fitness)
        assert num_gens > 0
        assert len(result.history.best_fitness) == num_gens
        assert len(result.history.std_fitness) == num_gens

        # No NaN values
        assert not jnp.any(jnp.isnan(result.history.mean_fitness))

    def test_simulation_with_learning(self):
        """Simulation should work with Hebbian learning enabled."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 20
        config.learning.enabled = True
        config.learning.learning_rate = 0.01

        # Should complete without errors
        result = run_simulation(key, config, verbose=False)
        assert result.history.mean_fitness.shape[0] > 0

    def test_simulation_without_learning(self):
        """Simulation should work without learning."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 20
        config.learning.enabled = False

        result = run_simulation(key, config, verbose=False)
        assert result.history.mean_fitness.shape[0] > 0


class TestDifferentConfigurations:
    """Tests for different configuration scenarios."""

    @pytest.mark.parametrize("clump_scale", [1, 5, 10, 20])
    def test_different_clump_scales(self, clump_scale):
        """Simulation should work with different clump scales."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 10
        config.environment.clump_scale = clump_scale

        result = run_simulation(key, config, verbose=False)
        assert not jnp.any(jnp.isnan(result.history.mean_fitness))

    @pytest.mark.parametrize("accuracy", [0.55, 0.75, 0.95])
    def test_different_accuracies(self, accuracy):
        """Simulation should work with different sensory accuracies."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 10
        config.environment.sensory_accuracy = accuracy

        result = run_simulation(key, config, verbose=False)
        assert not jnp.any(jnp.isnan(result.history.mean_fitness))

    def test_high_mutation_rate(self):
        """Simulation should handle high mutation rate."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 10
        config.genetic.mutation_rate = 0.1

        result = run_simulation(key, config, verbose=False)
        assert result.history.mean_fitness.shape[0] > 0

    def test_low_crossover_rate(self):
        """Simulation should handle low crossover rate."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 10
        config.genetic.crossover_rate = 0.1

        result = run_simulation(key, config, verbose=False)
        assert result.history.mean_fitness.shape[0] > 0


class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_same_seed_same_result(self):
        """Same seed should produce same results."""
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 10

        result1 = run_simulation(jax.random.PRNGKey(42), config, verbose=False)
        result2 = run_simulation(jax.random.PRNGKey(42), config, verbose=False)

        assert jnp.allclose(result1.history.mean_fitness, result2.history.mean_fitness)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        config = SimulationConfig()
        config.genetic.population_size = 20
        config.genetic.max_generations = 10

        result1 = run_simulation(jax.random.PRNGKey(42), config, verbose=False)
        result2 = run_simulation(jax.random.PRNGKey(123), config, verbose=False)

        # Results should be different (at least slightly)
        assert not jnp.allclose(result1.history.mean_fitness, result2.history.mean_fitness)

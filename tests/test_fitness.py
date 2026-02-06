"""Tests for fitness evaluation."""

import jax
import jax.numpy as jnp
import pytest

from habituation_evolution.config import (
    EnvironmentConfig,
    FitnessConfig,
    LearningConfig,
    NetworkConfig,
    SimulationConfig,
)
from habituation_evolution.environment import Environment
from habituation_evolution.fitness import (
    check_cluster_tracking_evolved,
    compute_always_eat_fitness,
    compute_lifespan_fitness,
    compute_never_eat_fitness,
    compute_optimal_fitness,
    compute_random_baseline_fitness,
    compute_sensory_only_fitness,
    compute_step_fitness,
    compute_success_threshold,
    count_correct_decisions,
    evaluate_creature,
    evaluate_population,
)
from habituation_evolution.genetics import create_random_genotype, create_random_population


class TestStepFitness:
    """Tests for single-step fitness computation."""

    def test_eat_food_reward(self):
        """Eating food should give positive reward."""
        config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0)
        fitness = compute_step_fitness(
            ate=jnp.array(True), is_food=jnp.array(True), config=config
        )
        assert fitness == 1.0

    def test_eat_poison_penalty(self):
        """Eating poison should give negative penalty."""
        config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0)
        fitness = compute_step_fitness(
            ate=jnp.array(True), is_food=jnp.array(False), config=config
        )
        assert fitness == -1.0

    def test_not_eat_cost(self):
        """Not eating should give no_eat_cost."""
        config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0, no_eat_cost=-0.1)
        fitness_food = compute_step_fitness(
            ate=jnp.array(False), is_food=jnp.array(True), config=config
        )
        fitness_poison = compute_step_fitness(
            ate=jnp.array(False), is_food=jnp.array(False), config=config
        )
        assert fitness_food == -0.1
        assert fitness_poison == -0.1


class TestLifespanFitness:
    """Tests for lifespan fitness computation."""

    def test_perfect_decisions(self):
        """Perfect decisions should give optimal fitness."""
        config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0, no_eat_cost=0.0)
        true_sequence = jnp.array([1, 1, 0, 0, 1, 0])  # 3 food, 3 poison
        decisions = jnp.array([True, True, False, False, True, False])  # Perfect

        fitness = compute_lifespan_fitness(decisions, true_sequence, config)
        # 3 eat_food (3*1.0) + 3 no_eat_poison (3*0.0) = 3.0
        assert fitness == 3.0

    def test_worst_decisions(self):
        """Worst decisions should give minimum fitness."""
        config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0, no_eat_cost=0.0)
        true_sequence = jnp.array([1, 1, 0, 0, 1, 0])
        decisions = jnp.array([False, False, True, True, False, True])  # All wrong

        fitness = compute_lifespan_fitness(decisions, true_sequence, config)
        # 3 no_eat_food (0) + 3 eat_poison (-3) = -3.0
        assert fitness == -3.0

    def test_always_eat(self):
        """Always eating should give food rewards minus poison penalties."""
        config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0)
        true_sequence = jnp.array([1, 1, 1, 0, 0, 0])  # 3 food, 3 poison
        decisions = jnp.ones(6, dtype=jnp.bool_)  # Always eat

        fitness = compute_lifespan_fitness(decisions, true_sequence, config)
        # 3 eat_food (3) + 3 eat_poison (-3) = 0
        assert fitness == 0.0


class TestCorrectDecisions:
    """Tests for counting correct decisions."""

    def test_all_correct(self):
        """Should count all correct decisions."""
        true_seq = jnp.array([1, 1, 0, 0])
        decisions = jnp.array([True, True, False, False])
        count = count_correct_decisions(decisions, true_seq)
        assert count == 4

    def test_all_wrong(self):
        """Should count zero when all wrong."""
        true_seq = jnp.array([1, 1, 0, 0])
        decisions = jnp.array([False, False, True, True])
        count = count_correct_decisions(decisions, true_seq)
        assert count == 0

    def test_half_correct(self):
        """Should count half when half correct."""
        true_seq = jnp.array([1, 1, 0, 0])
        decisions = jnp.array([True, False, True, False])  # 1 correct eat, 1 correct avoid
        count = count_correct_decisions(decisions, true_seq)
        assert count == 2


class TestTheoreticalFitness:
    """Tests for theoretical fitness calculations."""

    def test_optimal_fitness(self):
        """Optimal fitness should be positive."""
        env_config = EnvironmentConfig(lifespan=1000)
        fit_config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0, no_eat_cost=0.0)
        optimal = compute_optimal_fitness(env_config, fit_config)
        # 500 food * 1.0 + 500 avoid_poison * 0 = 500
        assert optimal == 500.0

    def test_random_baseline(self):
        """Random baseline should be near zero with symmetric rewards."""
        env_config = EnvironmentConfig(lifespan=1000)
        fit_config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0, no_eat_cost=0.0)
        baseline = compute_random_baseline_fitness(env_config, fit_config)
        # 250 eat_food + 250 eat_poison + 500 no_eat = 250 - 250 = 0
        assert baseline == 0.0

    def test_always_eat_fitness(self):
        """Always eat should be zero with symmetric rewards."""
        env_config = EnvironmentConfig(lifespan=1000)
        fit_config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0)
        always_eat = compute_always_eat_fitness(env_config, fit_config)
        # 500 * 1 + 500 * (-1) = 0
        assert always_eat == 0.0

    def test_never_eat_fitness(self):
        """Never eat should be lifespan * no_eat_cost."""
        env_config = EnvironmentConfig(lifespan=1000)
        fit_config = FitnessConfig(no_eat_cost=-0.1)
        never_eat = compute_never_eat_fitness(env_config, fit_config)
        assert never_eat == -100.0

    def test_sensory_only_fitness(self):
        """Sensory-only should be between random and optimal."""
        env_config = EnvironmentConfig(lifespan=1000, sensory_accuracy=0.75)
        fit_config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0, no_eat_cost=0.0)

        optimal = compute_optimal_fitness(env_config, fit_config)
        random = compute_random_baseline_fitness(env_config, fit_config)
        sensory = compute_sensory_only_fitness(env_config, fit_config)

        assert random <= sensory <= optimal

    def test_sensory_accuracy_affects_fitness(self):
        """Higher accuracy should give higher sensory-only fitness."""
        fit_config = FitnessConfig(eat_food_reward=1.0, eat_poison_penalty=-1.0, no_eat_cost=0.0)

        env_low = EnvironmentConfig(lifespan=1000, sensory_accuracy=0.6)
        env_high = EnvironmentConfig(lifespan=1000, sensory_accuracy=0.9)

        sensory_low = compute_sensory_only_fitness(env_low, fit_config)
        sensory_high = compute_sensory_only_fitness(env_high, fit_config)

        assert sensory_low < sensory_high


class TestSuccessThreshold:
    """Tests for success threshold computation."""

    def test_threshold_between_sensory_and_optimal(self):
        """Threshold should be between sensory-only and optimal."""
        config = SimulationConfig()
        threshold = compute_success_threshold(config)
        optimal = compute_optimal_fitness(config.environment, config.fitness)
        sensory = compute_sensory_only_fitness(config.environment, config.fitness)

        assert sensory < threshold < optimal

    def test_check_not_evolved_initially(self):
        """Low fitness should not pass threshold."""
        config = SimulationConfig()
        assert not check_cluster_tracking_evolved(0.0, config)

    def test_check_evolved_high_fitness(self):
        """High fitness should pass threshold."""
        config = SimulationConfig()
        optimal = compute_optimal_fitness(config.environment, config.fitness)
        assert check_cluster_tracking_evolved(optimal * 0.9, config)


class TestCreatureEvaluation:
    """Tests for creature evaluation."""

    def test_evaluate_returns_result(self):
        """Evaluation should return proper result structure."""
        key = jax.random.PRNGKey(42)
        net_config = NetworkConfig()
        learn_config = LearningConfig(enabled=False)
        fit_config = FitnessConfig()

        genotype = create_random_genotype(key)
        environment = Environment(
            true_sequence=jnp.ones(100, dtype=jnp.int32),
            perceived_sequence=jnp.ones(100),
        )

        result = evaluate_creature(
            genotype, environment, net_config, learn_config, fit_config
        )

        assert result.decisions.shape == (100,)
        assert result.motor_outputs.shape == (100,)
        assert result.total_decisions == 100


class TestPopulationEvaluation:
    """Tests for population evaluation."""

    def test_evaluate_population_shape(self):
        """Population evaluation should return correct shape."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()
        config.genetic.population_size = 50

        key_pop, key_eval = jax.random.split(key)
        population = create_random_population(key_pop, 50)

        fitness = evaluate_population(
            key_eval,
            population,
            config.environment,
            config.network,
            config.learning,
            config.fitness,
        )

        assert fitness.shape == (50,)

    def test_population_fitness_varies(self):
        """Different individuals should have different fitness."""
        key = jax.random.PRNGKey(42)
        config = SimulationConfig()

        key_pop, key_eval = jax.random.split(key)
        population = create_random_population(key_pop, 100)

        fitness = evaluate_population(
            key_eval,
            population,
            config.environment,
            config.network,
            config.learning,
            config.fitness,
        )

        # There should be variance in fitness
        assert jnp.std(fitness) > 0

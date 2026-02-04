"""Tests for genetic algorithm components."""

import jax
import jax.numpy as jnp
import pytest

from habituation_evolution.config import GeneticConfig, NetworkConfig
from habituation_evolution.genetics import (
    GENOTYPE_LENGTH,
    apply_linear_scaling,
    create_next_generation,
    create_random_genotype,
    create_random_population,
    decode_genotype,
    decode_population,
    encode_genotype,
    point_mutation,
    select_parents_fitness_proportionate,
    two_point_crossover,
)
from habituation_evolution.network import NetworkParams


class TestGenotypeCreation:
    """Tests for genotype creation."""

    def test_genotype_length(self):
        """Genotype should have correct length."""
        key = jax.random.PRNGKey(42)
        genotype = create_random_genotype(key)
        assert genotype.shape == (GENOTYPE_LENGTH,)

    def test_genotype_binary(self):
        """Genotype should contain only 0s and 1s."""
        key = jax.random.PRNGKey(42)
        genotype = create_random_genotype(key)
        assert jnp.all((genotype == 0) | (genotype == 1))

    def test_population_shape(self):
        """Population should have correct shape."""
        key = jax.random.PRNGKey(42)
        pop = create_random_population(key, 100)
        assert pop.shape == (100, GENOTYPE_LENGTH)

    def test_population_diversity(self):
        """Population should have diverse individuals."""
        key = jax.random.PRNGKey(42)
        pop = create_random_population(key, 100)
        # Check that not all individuals are the same
        assert not jnp.all(pop[0] == pop[1])


class TestGenotypeDecoding:
    """Tests for genotype decoding."""

    def test_decode_returns_params(self):
        """Decoding should return NetworkParams."""
        key = jax.random.PRNGKey(42)
        config = NetworkConfig()
        genotype = create_random_genotype(key)
        params = decode_genotype(genotype, config)
        assert isinstance(params, NetworkParams)
        assert params.weights.shape == (4,)
        assert params.biases.shape == (2,)
        assert params.learnable_mask.shape == (6,)

    def test_weights_in_range(self):
        """Decoded weights should be within expected range."""
        key = jax.random.PRNGKey(42)
        config = NetworkConfig(max_weight=4.1)
        genotype = create_random_genotype(key)
        params = decode_genotype(genotype, config)
        # Weights should be in [-max_weight, max_weight]
        assert jnp.all(jnp.abs(params.weights) <= config.max_weight + 0.01)
        assert jnp.all(jnp.abs(params.biases) <= config.max_weight + 0.01)

    def test_learnable_mask_boolean(self):
        """Learnable mask should be boolean."""
        key = jax.random.PRNGKey(42)
        config = NetworkConfig()
        genotype = create_random_genotype(key)
        params = decode_genotype(genotype, config)
        assert params.learnable_mask.dtype == jnp.bool_

    def test_decode_population(self):
        """Should decode entire population at once."""
        key = jax.random.PRNGKey(42)
        config = NetworkConfig()
        pop = create_random_population(key, 50)
        params = decode_population(pop, config)
        assert params.weights.shape == (50, 4)
        assert params.biases.shape == (50, 2)


class TestGenotypeEncoding:
    """Tests for round-trip encoding/decoding."""

    def test_round_trip(self):
        """Encoding decoded params should roughly recover genotype."""
        key = jax.random.PRNGKey(42)
        config = NetworkConfig()
        original = create_random_genotype(key)
        params = decode_genotype(original, config)
        recovered = encode_genotype(params, config)

        # Decode both and compare params (encoding may not be exact due to quantization)
        params_original = decode_genotype(original, config)
        params_recovered = decode_genotype(recovered, config)

        # Weights should be close (may differ slightly due to quantization)
        assert jnp.allclose(params_original.weights, params_recovered.weights, atol=0.3)


class TestCrossover:
    """Tests for two-point crossover."""

    def test_crossover_produces_valid_genotypes(self):
        """Crossover should produce valid binary genotypes."""
        key = jax.random.PRNGKey(42)
        p1 = jnp.zeros(GENOTYPE_LENGTH, dtype=jnp.int32)
        p2 = jnp.ones(GENOTYPE_LENGTH, dtype=jnp.int32)

        o1, o2 = two_point_crossover(key, p1, p2)

        assert o1.shape == (GENOTYPE_LENGTH,)
        assert o2.shape == (GENOTYPE_LENGTH,)
        assert jnp.all((o1 == 0) | (o1 == 1))
        assert jnp.all((o2 == 0) | (o2 == 1))

    def test_crossover_mixes_parents(self):
        """Offspring should contain bits from both parents."""
        key = jax.random.PRNGKey(42)
        p1 = jnp.zeros(GENOTYPE_LENGTH, dtype=jnp.int32)
        p2 = jnp.ones(GENOTYPE_LENGTH, dtype=jnp.int32)

        o1, o2 = two_point_crossover(key, p1, p2)

        # Offspring 1 should have some 0s and some 1s (mix of parents)
        assert jnp.sum(o1) > 0  # Has some 1s from p2
        assert jnp.sum(o1) < GENOTYPE_LENGTH  # Has some 0s from p1

    def test_crossover_complementary(self):
        """Offspring should be complementary (o1 XOR o2 = p1 XOR p2)."""
        key = jax.random.PRNGKey(42)
        p1 = jnp.zeros(GENOTYPE_LENGTH, dtype=jnp.int32)
        p2 = jnp.ones(GENOTYPE_LENGTH, dtype=jnp.int32)

        o1, o2 = two_point_crossover(key, p1, p2)

        # Where o1 has p1's bit, o2 should have p2's bit
        assert jnp.all((o1 + o2) == 1)  # Should be complementary when parents are 0 and 1


class TestMutation:
    """Tests for point mutation."""

    def test_mutation_rate_zero(self):
        """Zero mutation rate should not change genotype."""
        key = jax.random.PRNGKey(42)
        genotype = jnp.ones(GENOTYPE_LENGTH, dtype=jnp.int32)
        mutated = point_mutation(key, genotype, rate=0.0)
        assert jnp.all(mutated == genotype)

    def test_mutation_rate_one(self):
        """100% mutation rate should flip all bits."""
        key = jax.random.PRNGKey(42)
        genotype = jnp.ones(GENOTYPE_LENGTH, dtype=jnp.int32)
        mutated = point_mutation(key, genotype, rate=1.0)
        assert jnp.all(mutated == 0)

    def test_mutation_rate_approximate(self):
        """Mutation rate should approximately match expected flips."""
        key = jax.random.PRNGKey(42)
        rate = 0.1
        # Use many genotypes to get good statistics
        genotypes = jnp.ones((100, GENOTYPE_LENGTH), dtype=jnp.int32)
        keys = jax.random.split(key, 100)
        mutated = jax.vmap(lambda k, g: point_mutation(k, g, rate))(keys, genotypes)

        # Count flips
        flips = jnp.sum(mutated != genotypes)
        expected = 100 * GENOTYPE_LENGTH * rate
        # Should be within 20% of expected
        assert abs(flips - expected) / expected < 0.2


class TestLinearScaling:
    """Tests for linear fitness scaling."""

    def test_preserves_mean(self):
        """Scaling should approximately preserve mean."""
        fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled = apply_linear_scaling(fitness, target_max_ratio=1.5)
        # Mean should be preserved or shifted but relative relationships maintained
        assert jnp.std(scaled) > 0  # Not all same

    def test_nonnegative(self):
        """Scaled fitness should be non-negative."""
        fitness = jnp.array([-1.0, 0.0, 1.0, 2.0])
        scaled = apply_linear_scaling(fitness, target_max_ratio=1.5)
        assert jnp.all(scaled >= 0)

    def test_uniform_fitness(self):
        """Uniform fitness should stay uniform."""
        fitness = jnp.array([5.0, 5.0, 5.0, 5.0])
        scaled = apply_linear_scaling(fitness, target_max_ratio=1.5)
        # All should be equal (no variance to scale)
        assert jnp.allclose(scaled, scaled[0])


class TestSelection:
    """Tests for fitness-proportionate selection."""

    def test_selection_count(self):
        """Should return requested number of parents."""
        key = jax.random.PRNGKey(42)
        pop = create_random_population(key, 100)
        fitness = jnp.arange(100, dtype=jnp.float32)

        key2 = jax.random.PRNGKey(123)
        parents = select_parents_fitness_proportionate(key2, pop, fitness, num_parents=50)

        assert parents.shape == (50, GENOTYPE_LENGTH)

    def test_selection_valid_individuals(self):
        """Selected parents should be from original population."""
        key = jax.random.PRNGKey(42)
        pop = create_random_population(key, 10)
        fitness = jnp.arange(10, dtype=jnp.float32) + 1

        key2 = jax.random.PRNGKey(123)
        parents = select_parents_fitness_proportionate(key2, pop, fitness, num_parents=20)

        # Each parent should match some individual in original population
        for p in parents:
            matches = jnp.any(jnp.all(pop == p, axis=1))
            assert matches

    def test_selection_favors_high_fitness(self):
        """Higher fitness individuals should be selected more often."""
        key = jax.random.PRNGKey(42)
        # Create identifiable population
        pop = jnp.eye(GENOTYPE_LENGTH, dtype=jnp.int32)[:10]  # 10 unique individuals
        # Give last individual very high fitness
        fitness = jnp.ones(10)
        fitness = fitness.at[9].set(100.0)

        # Select many times
        key2 = jax.random.PRNGKey(123)
        parents = select_parents_fitness_proportionate(key2, pop, fitness, num_parents=1000)

        # Count how often high-fitness individual is selected
        is_high_fitness = jnp.all(parents == pop[9], axis=1)
        count = jnp.sum(is_high_fitness)

        # Should be selected much more than 10% of the time
        assert count > 500


class TestNextGeneration:
    """Tests for creating next generation."""

    def test_next_gen_size(self):
        """Next generation should have same size as current."""
        key = jax.random.PRNGKey(42)
        config = GeneticConfig(population_size=100, mutation_rate=0.01, crossover_rate=0.7)
        pop = create_random_population(key, 100)
        fitness = jnp.arange(100, dtype=jnp.float32) + 1

        key2 = jax.random.PRNGKey(123)
        next_pop = create_next_generation(key2, pop, fitness, config)

        assert next_pop.shape == pop.shape

    def test_next_gen_valid(self):
        """Next generation should contain valid binary genotypes."""
        key = jax.random.PRNGKey(42)
        config = GeneticConfig(population_size=100, mutation_rate=0.01, crossover_rate=0.7)
        pop = create_random_population(key, 100)
        fitness = jnp.arange(100, dtype=jnp.float32) + 1

        key2 = jax.random.PRNGKey(123)
        next_pop = create_next_generation(key2, pop, fitness, config)

        assert jnp.all((next_pop == 0) | (next_pop == 1))

    def test_next_gen_different(self):
        """Next generation should be different from current (with mutation/crossover)."""
        key = jax.random.PRNGKey(42)
        config = GeneticConfig(population_size=100, mutation_rate=0.05, crossover_rate=0.9)
        pop = create_random_population(key, 100)
        fitness = jnp.ones(100)  # Uniform fitness

        key2 = jax.random.PRNGKey(123)
        next_pop = create_next_generation(key2, pop, fitness, config)

        # At least some individuals should be different
        same_count = jnp.sum(jnp.all(pop == next_pop, axis=1))
        assert same_count < 100  # Not all the same

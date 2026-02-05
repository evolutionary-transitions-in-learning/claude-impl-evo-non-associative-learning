"""Tests for genetics module."""

import jax
import jax.numpy as jnp
import pytest

from habituation_experiment_alice.config import NetworkConfig
from habituation_experiment_alice.genetics import (
    GENOTYPE_LENGTH_N1,
    compute_genotype_length,
    create_random_genotype,
    create_random_population,
    decode_genotype,
    encode_genotype,
    point_mutation,
    two_point_crossover,
)


class TestGenotypeConstants:
    def test_genotype_length(self):
        config = NetworkConfig(num_stimulus_channels=1)
        assert compute_genotype_length(config) == 81
        assert GENOTYPE_LENGTH_N1 == 81


class TestGenotypeCreation:
    def test_random_genotype_shape(self):
        key = jax.random.PRNGKey(0)
        genotype = create_random_genotype(key)
        assert genotype.shape == (81,)

    def test_random_genotype_binary(self):
        key = jax.random.PRNGKey(0)
        genotype = create_random_genotype(key)
        assert jnp.all((genotype == 0) | (genotype == 1))

    def test_random_population_shape(self):
        key = jax.random.PRNGKey(0)
        pop = create_random_population(key, 20)
        assert pop.shape == (20, 81)


class TestGenotypeDecode:
    def test_decode_shape(self):
        key = jax.random.PRNGKey(0)
        config = NetworkConfig()
        genotype = create_random_genotype(key)
        params = decode_genotype(genotype, config)
        assert params.weights.shape == (9,)
        assert params.biases.shape == (3,)
        assert params.learnable_mask.shape == (12,)

    def test_decoded_weights_bounded(self):
        key = jax.random.PRNGKey(42)
        config = NetworkConfig()
        genotype = create_random_genotype(key)
        params = decode_genotype(genotype, config)
        assert jnp.all(jnp.abs(params.weights) <= config.max_weight + 1e-5)
        assert jnp.all(jnp.abs(params.biases) <= config.max_weight + 1e-5)


class TestEncodeDecodeRoundTrip:
    def test_roundtrip(self):
        key = jax.random.PRNGKey(123)
        config = NetworkConfig()
        genotype = create_random_genotype(key)
        params = decode_genotype(genotype, config)
        re_encoded = encode_genotype(params, config)
        params2 = decode_genotype(re_encoded, config)
        assert jnp.allclose(params.weights, params2.weights, atol=1e-5)
        assert jnp.allclose(params.biases, params2.biases, atol=1e-5)


class TestCrossover:
    def test_offspring_shape(self):
        key = jax.random.PRNGKey(0)
        p1 = jnp.zeros(81, dtype=jnp.int32)
        p2 = jnp.ones(81, dtype=jnp.int32)
        o1, o2 = two_point_crossover(key, p1, p2)
        assert o1.shape == (81,)
        assert o2.shape == (81,)

    def test_offspring_binary(self):
        key = jax.random.PRNGKey(0)
        p1 = jnp.zeros(81, dtype=jnp.int32)
        p2 = jnp.ones(81, dtype=jnp.int32)
        o1, o2 = two_point_crossover(key, p1, p2)
        assert jnp.all((o1 == 0) | (o1 == 1))

    def test_offspring_has_bits_from_both(self):
        key = jax.random.PRNGKey(5)
        p1 = jnp.zeros(81, dtype=jnp.int32)
        p2 = jnp.ones(81, dtype=jnp.int32)
        o1, _ = two_point_crossover(key, p1, p2)
        # Should have some 0s and some 1s (very likely)
        assert jnp.sum(o1) > 0
        assert jnp.sum(o1) < 81


class TestMutation:
    def test_rate_zero_preserves(self):
        key = jax.random.PRNGKey(0)
        genotype = jnp.ones(81, dtype=jnp.int32)
        mutated = point_mutation(key, genotype, 0.0)
        assert jnp.all(mutated == genotype)

    def test_rate_one_flips_all(self):
        key = jax.random.PRNGKey(0)
        genotype = jnp.ones(81, dtype=jnp.int32)
        mutated = point_mutation(key, genotype, 1.0)
        assert jnp.all(mutated == 0)

    def test_rate_approximate(self):
        key = jax.random.PRNGKey(0)
        genotype = jnp.zeros(10000, dtype=jnp.int32)
        mutated = point_mutation(key, genotype, 0.01)
        flip_count = jnp.sum(mutated != genotype)
        # Expect ~100 flips, allow wide margin
        assert 50 < int(flip_count) < 200

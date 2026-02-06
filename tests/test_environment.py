"""Tests for environment generation."""

import jax
import jax.numpy as jnp
import pytest

from habituation_evolution.config import EnvironmentConfig, NetworkConfig, NoiseMode
from habituation_evolution.environment import (
    Environment,
    apply_continuous_noise,
    apply_discrete_noise,
    compute_accuracy_empirical,
    compute_clump_statistics,
    expand_to_clumps,
    generate_base_sequence,
    generate_clumpy_sequence,
    generate_environment,
    generate_environments_batch,
)


class TestBaseSequence:
    """Tests for base sequence generation."""

    def test_generates_correct_length(self):
        key = jax.random.PRNGKey(42)
        seq = generate_base_sequence(key, 100)
        assert seq.shape == (100,)

    def test_contains_only_binary_values(self):
        key = jax.random.PRNGKey(42)
        seq = generate_base_sequence(key, 1000)
        assert jnp.all((seq == 0) | (seq == 1))

    def test_approximately_balanced(self):
        """With enough samples, should be close to 50/50."""
        key = jax.random.PRNGKey(42)
        seq = generate_base_sequence(key, 10000)
        ratio = jnp.mean(seq)
        assert 0.45 < ratio < 0.55

    def test_different_keys_different_sequences(self):
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)
        seq1 = generate_base_sequence(key1, 100)
        seq2 = generate_base_sequence(key2, 100)
        assert not jnp.allclose(seq1, seq2)


class TestClumpExpansion:
    """Tests for clump expansion."""

    def test_expands_correctly(self):
        base = jnp.array([1, 0, 1])
        expanded = expand_to_clumps(base, 3)
        expected = jnp.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
        assert jnp.allclose(expanded, expected)

    def test_clump_scale_1(self):
        """Clump scale 1 should leave sequence unchanged."""
        base = jnp.array([1, 0, 1, 0, 1])
        expanded = expand_to_clumps(base, 1)
        assert jnp.allclose(expanded, base)

    def test_output_length(self):
        base = jnp.array([1, 0, 1, 0])
        expanded = expand_to_clumps(base, 5)
        assert len(expanded) == 4 * 5


class TestClumpySequence:
    """Tests for clumpy sequence generation."""

    def test_correct_length(self):
        key = jax.random.PRNGKey(42)
        seq = generate_clumpy_sequence(key, 1000, 10)
        assert len(seq) == 1000

    def test_minimum_clump_length(self):
        """Minimum clump should be >= clump_scale."""
        key = jax.random.PRNGKey(42)
        clump_scale = 5
        seq = generate_clumpy_sequence(key, 1000, clump_scale)
        stats = compute_clump_statistics(seq)
        assert stats["min"] >= clump_scale

    def test_clump_scale_1_random(self):
        """Clump scale 1 should produce random-like sequence."""
        key = jax.random.PRNGKey(42)
        seq = generate_clumpy_sequence(key, 1000, 1)
        stats = compute_clump_statistics(seq)
        # With clump scale 1, we expect small average clump size
        assert stats["mean"] < 5


class TestDiscreteNoise:
    """Tests for discrete noise application."""

    def test_perfect_accuracy(self):
        """100% accuracy should preserve sequence."""
        key = jax.random.PRNGKey(42)
        true_seq = jnp.array([1, 0, 1, 0, 1, 0, 1, 0])
        perceived = apply_discrete_noise(key, true_seq, 1.0)
        assert jnp.allclose(perceived, true_seq)

    def test_50_percent_accuracy(self):
        """50% accuracy should flip about half."""
        key = jax.random.PRNGKey(42)
        true_seq = jnp.ones(10000, dtype=jnp.int32)
        perceived = apply_discrete_noise(key, true_seq, 0.5)
        flip_rate = jnp.mean(perceived != true_seq)
        assert 0.45 < flip_rate < 0.55

    def test_accuracy_matches_parameter(self):
        """Empirical accuracy should match parameter."""
        key = jax.random.PRNGKey(42)
        true_seq = generate_base_sequence(key, 10000)
        for accuracy in [0.6, 0.75, 0.9]:
            key, subkey = jax.random.split(key)
            perceived = apply_discrete_noise(subkey, true_seq, accuracy)
            empirical = compute_accuracy_empirical(true_seq, perceived)
            assert abs(empirical - accuracy) < 0.05


class TestContinuousNoise:
    """Tests for continuous noise application."""

    def test_output_is_continuous(self):
        """Output should be continuous values, not binary."""
        key = jax.random.PRNGKey(42)
        true_seq = jnp.array([1, 0, 1, 0, 1])
        perceived = apply_continuous_noise(key, true_seq, 0.75, 0.5, 1.0, -1.0)
        # Should have values that aren't exactly 1.0 or -1.0
        assert not jnp.all((perceived == 1.0) | (perceived == -1.0))

    def test_perfect_accuracy_low_noise(self):
        """High accuracy should preserve sign."""
        key = jax.random.PRNGKey(42)
        true_seq = generate_base_sequence(key, 1000)
        perceived = apply_continuous_noise(key, true_seq, 0.99, 0.1, 1.0, -1.0)
        # Signs should mostly match
        perceived_binary = (perceived > 0).astype(jnp.int32)
        accuracy = jnp.mean(perceived_binary == true_seq)
        assert accuracy > 0.9


class TestEnvironmentGeneration:
    """Tests for full environment generation."""

    def test_returns_environment_tuple(self):
        key = jax.random.PRNGKey(42)
        env_config = EnvironmentConfig(clump_scale=10, sensory_accuracy=0.75, lifespan=100)
        net_config = NetworkConfig()
        env = generate_environment(key, env_config, net_config)
        assert isinstance(env, Environment)
        assert env.true_sequence.shape == (100,)
        assert env.perceived_sequence.shape == (100,)

    def test_discrete_mode(self):
        key = jax.random.PRNGKey(42)
        env_config = EnvironmentConfig(
            clump_scale=10, sensory_accuracy=0.75, lifespan=100, noise_mode=NoiseMode.DISCRETE
        )
        net_config = NetworkConfig(sweet_input=1.0, sour_input=-1.0)
        env = generate_environment(key, env_config, net_config)
        # Perceived should be either sweet or sour input values
        assert jnp.all((env.perceived_sequence == 1.0) | (env.perceived_sequence == -1.0))

    def test_continuous_mode(self):
        key = jax.random.PRNGKey(42)
        env_config = EnvironmentConfig(
            clump_scale=10,
            sensory_accuracy=0.75,
            lifespan=100,
            noise_mode=NoiseMode.CONTINUOUS,
            noise_std=0.5,
        )
        net_config = NetworkConfig()
        env = generate_environment(key, env_config, net_config)
        # Perceived should have continuous values
        unique_count = len(jnp.unique(env.perceived_sequence))
        assert unique_count > 2  # More than just two values


class TestBatchGeneration:
    """Tests for batch environment generation."""

    def test_batch_shape(self):
        key = jax.random.PRNGKey(42)
        env_config = EnvironmentConfig(clump_scale=10, sensory_accuracy=0.75, lifespan=100)
        net_config = NetworkConfig()
        envs = generate_environments_batch(key, env_config, net_config, batch_size=50)
        assert envs.true_sequence.shape == (50, 100)
        assert envs.perceived_sequence.shape == (50, 100)

    def test_batch_independence(self):
        """Each environment in batch should be different."""
        key = jax.random.PRNGKey(42)
        env_config = EnvironmentConfig(clump_scale=10, sensory_accuracy=0.75, lifespan=100)
        net_config = NetworkConfig()
        envs = generate_environments_batch(key, env_config, net_config, batch_size=10)
        # Check that first two environments are different
        assert not jnp.allclose(envs.true_sequence[0], envs.true_sequence[1])


class TestClumpStatistics:
    """Tests for clump statistics computation."""

    def test_known_sequence(self):
        seq = jnp.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0])
        stats = compute_clump_statistics(seq)
        assert stats["min"] == 1  # Single 0 at end
        assert stats["max"] == 4  # Four 1s in middle
        assert stats["num_clumps"] == 4

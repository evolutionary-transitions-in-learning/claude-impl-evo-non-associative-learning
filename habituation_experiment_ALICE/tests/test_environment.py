"""Tests for environment module."""

import jax
import jax.numpy as jnp
import pytest

from habituation_experiment_alice.config import SimulationConfig
from habituation_experiment_alice.environment import (
    assign_threat_types,
    generate_clumpy_threat_sequence,
    generate_pain_signal,
    generate_phase1_environment,
    generate_phase2_environment,
    generate_stimulus_signal,
)


class TestClumpySequence:
    def test_shape(self):
        key = jax.random.PRNGKey(0)
        seq = generate_clumpy_threat_sequence(key, 100, 10)
        assert seq.shape == (100,)

    def test_binary(self):
        key = jax.random.PRNGKey(0)
        seq = generate_clumpy_threat_sequence(key, 100, 10)
        assert jnp.all((seq == 0) | (seq == 1))

    def test_has_both_values(self):
        key = jax.random.PRNGKey(0)
        seq = generate_clumpy_threat_sequence(key, 1000, 10)
        assert jnp.sum(seq) > 0
        assert jnp.sum(seq) < 1000


class TestThreatTypes:
    def test_phase1_all_true(self):
        key = jax.random.PRNGKey(0)
        threat_present = jnp.ones(100)
        true_threat = assign_threat_types(key, threat_present, 1.0, 10, 100)
        assert jnp.allclose(true_threat, threat_present)

    def test_no_true_when_no_threat(self):
        key = jax.random.PRNGKey(0)
        threat_present = jnp.zeros(100)
        true_threat = assign_threat_types(key, threat_present, 1.0, 10, 100)
        assert jnp.all(true_threat == 0)


class TestPainSignal:
    def test_delay(self):
        true_threat = jnp.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=jnp.float32)
        pain = generate_pain_signal(true_threat, pain_delay=1, pain_magnitude=1.0)
        # Pain should be delayed by 1: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        assert pain.shape == (10,)
        assert float(pain[0]) == 0.0
        assert float(pain[1]) == 0.0
        assert float(pain[2]) == 0.0
        assert float(pain[3]) == 1.0
        assert float(pain[4]) == 1.0

    def test_delay_2(self):
        true_threat = jnp.array([1, 1, 0, 0, 0], dtype=jnp.float32)
        pain = generate_pain_signal(true_threat, pain_delay=2, pain_magnitude=1.0)
        assert float(pain[0]) == 0.0
        assert float(pain[1]) == 0.0
        assert float(pain[2]) == 1.0
        assert float(pain[3]) == 1.0

    def test_magnitude(self):
        true_threat = jnp.array([1, 0, 0, 0], dtype=jnp.float32)
        pain = generate_pain_signal(true_threat, pain_delay=1, pain_magnitude=2.5)
        assert float(pain[1]) == 2.5

    def test_first_steps_zero(self):
        true_threat = jnp.ones(10, dtype=jnp.float32)
        pain = generate_pain_signal(true_threat, pain_delay=3, pain_magnitude=1.0)
        assert jnp.all(pain[:3] == 0.0)


class TestStimulusSignal:
    def test_shape(self):
        threat = jnp.ones(50)
        signal = generate_stimulus_signal(threat, 1.0, 1)
        assert signal.shape == (50, 1)

    def test_zero_when_no_threat(self):
        threat = jnp.zeros(10)
        signal = generate_stimulus_signal(threat, 1.0, 1)
        assert jnp.all(signal == 0.0)

    def test_magnitude(self):
        threat = jnp.ones(5)
        signal = generate_stimulus_signal(threat, 2.5, 1)
        assert jnp.allclose(signal, 2.5)


class TestPhaseEnvironment:
    def test_phase1_all_true_threats(self):
        key = jax.random.PRNGKey(0)
        config = SimulationConfig()
        env = generate_phase1_environment(key, config)
        # Where threat is present, it should be a true threat
        mask = env.threat_present > 0
        assert jnp.all(env.true_threat[mask] > 0)

    def test_phase2_shapes(self):
        key = jax.random.PRNGKey(0)
        config = SimulationConfig()
        env = generate_phase2_environment(key, config)
        lt = config.environment.phase2_lifetime
        assert env.threat_present.shape == (lt,)
        assert env.true_threat.shape == (lt,)
        assert env.stimulus_signal.shape == (lt, 1)
        assert env.pain_signal.shape == (lt,)

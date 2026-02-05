"""Tests for health dynamics module."""

import jax
import jax.numpy as jnp
import pytest

from habituation_experiment_alice.config import HealthConfig
from habituation_experiment_alice.health import compute_health_delta, simulate_health


class TestHealthDelta:
    def test_passive_decay_only(self):
        config = HealthConfig()
        delta = compute_health_delta(jnp.array(0.0), jnp.array(0.0), config)
        assert abs(float(delta) - (-0.1)) < 1e-6

    def test_full_eating_no_threat(self):
        config = HealthConfig()
        delta = compute_health_delta(jnp.array(-1.0), jnp.array(0.0), config)
        # -0.1 + 1.0 = 0.9
        assert abs(float(delta) - 0.9) < 1e-6

    def test_full_withdrawal_true_threat(self):
        config = HealthConfig()
        delta = compute_health_delta(jnp.array(1.0), jnp.array(1.0), config)
        # -0.1 + 0 - (1 - 1.0) * 5.0 = -0.1
        assert abs(float(delta) - (-0.1)) < 1e-6

    def test_no_protection_true_threat(self):
        config = HealthConfig()
        delta = compute_health_delta(jnp.array(0.0), jnp.array(1.0), config)
        # -0.1 + 0 - (1 - 0) * 5.0 = -5.1
        assert abs(float(delta) - (-5.1)) < 1e-6

    def test_full_eating_true_threat(self):
        config = HealthConfig()
        delta = compute_health_delta(jnp.array(-1.0), jnp.array(1.0), config)
        # -0.1 + 1.0 - (1 - 0) * 5.0 = -4.1
        assert abs(float(delta) - (-4.1)) < 1e-6

    def test_partial_withdrawal(self):
        config = HealthConfig()
        delta = compute_health_delta(jnp.array(0.5), jnp.array(1.0), config)
        # -0.1 + 0 - (1 - 0.5) * 5.0 = -0.1 - 2.5 = -2.6
        assert abs(float(delta) - (-2.6)) < 1e-6


class TestSimulateHealth:
    def test_passive_decay_death(self):
        config = HealthConfig(starting_health=10.0, passive_decay=1.0)
        outputs = jnp.zeros(20)
        threats = jnp.zeros(20)
        result = simulate_health(outputs, threats, config)
        # Health: 10, 9, 8, ..., 1, 0, 0, 0...
        assert float(result.final_health) == 0.0
        assert float(result.health_trajectory[0]) == 9.0  # after first step
        assert float(result.health_trajectory[9]) == 0.0

    def test_eating_sustains(self):
        config = HealthConfig(starting_health=20.0)
        outputs = -jnp.ones(100)  # always eating
        threats = jnp.zeros(100)  # no threats
        result = simulate_health(outputs, threats, config)
        # delta = -0.1 + 1.0 = 0.9 per step, health grows
        assert float(result.final_health) > 20.0

    def test_death_stays_dead(self):
        config = HealthConfig(starting_health=5.0)
        outputs = jnp.zeros(20)
        threats = jnp.ones(20)  # continuous true threat, no protection
        result = simulate_health(outputs, threats, config)
        # delta = -5.1 per step, dies after ~1 step
        assert float(result.final_health) == 0.0
        # Should be dead early
        assert float(result.alive_mask[-1]) == 0.0

    def test_full_withdrawal_survives_threats(self):
        config = HealthConfig(starting_health=20.0)
        outputs = jnp.ones(50)  # always withdrawing
        threats = jnp.ones(50)  # continuous true threat
        result = simulate_health(outputs, threats, config)
        # delta = -0.1 per step (just passive decay, threat fully blocked)
        # 20 - 50*0.1 = 15
        assert abs(float(result.final_health) - 15.0) < 1e-4
        assert float(result.alive_mask[-1]) == 1.0

"""Tests for neural network module."""

import jax
import jax.numpy as jnp
import pytest

from habituation_experiment_alice.config import LearningConfig, NetworkConfig
from habituation_experiment_alice.network import (
    CONN_PAIN_OUT,
    CONN_STIM_OUT,
    NUM_CONNECTIONS_N1,
    NUM_NEURONS_N1,
    NUM_PARAMS_N1,
    NetworkParams,
    NetworkState,
    apply_hebbian_learning,
    init_network_params,
    init_network_state,
    network_step,
    run_network_phase,
)


class TestNetworkState:
    def test_init_shapes(self):
        state = init_network_state()
        assert state.activations.shape == (3,)
        assert jnp.all(state.activations == 0.0)

    def test_init_custom_size(self):
        state = init_network_state(num_neurons=5)
        assert state.activations.shape == (5,)


class TestNetworkParams:
    def test_init_defaults(self):
        params = init_network_params()
        assert params.weights.shape == (9,)
        assert params.biases.shape == (3,)
        assert params.learnable_mask.shape == (12,)

    def test_init_custom(self):
        w = jnp.ones(9)
        b = jnp.ones(3)
        m = jnp.ones(12, dtype=jnp.bool_)
        params = init_network_params(weights=w, biases=b, learnable_mask=m)
        assert jnp.all(params.weights == 1.0)
        assert jnp.all(params.biases == 1.0)


class TestNetworkStep:
    def test_zero_weights_zero_output(self):
        state = init_network_state()
        params = init_network_params()
        new_state, output = network_step(state, jnp.array(0.0), jnp.array(0.0), params)
        assert abs(float(output)) < 1e-6

    def test_positive_stimulus_positive_forward_weight(self):
        state = init_network_state()
        weights = jnp.zeros(9)
        weights = weights.at[CONN_STIM_OUT].set(2.0)
        params = init_network_params(weights=weights)
        _, output = network_step(state, jnp.array(1.0), jnp.array(0.0), params)
        assert float(output) > 0.0

    def test_pain_propagates(self):
        state = init_network_state()
        weights = jnp.zeros(9)
        weights = weights.at[CONN_PAIN_OUT].set(2.0)
        params = init_network_params(weights=weights)
        _, output = network_step(state, jnp.array(0.0), jnp.array(1.0), params)
        assert float(output) > 0.0

    def test_output_bounded(self):
        state = init_network_state()
        weights = jnp.ones(9) * 10.0
        params = init_network_params(weights=weights)
        _, output = network_step(state, jnp.array(100.0), jnp.array(100.0), params)
        assert -1.0 <= float(output) <= 1.0


class TestHebbianLearning:
    def test_learning_disabled(self):
        state = NetworkState(activations=jnp.array([1.0, 1.0, 1.0]))
        params = init_network_params(
            weights=jnp.zeros(9),
            learnable_mask=jnp.ones(12, dtype=jnp.bool_),
        )
        config = LearningConfig(enabled=False)
        new_params = apply_hebbian_learning(state, params, config)
        assert jnp.allclose(new_params.weights, params.weights)

    def test_learning_updates_learnable(self):
        state = NetworkState(activations=jnp.array([1.0, 1.0, 1.0]))
        mask = jnp.zeros(12, dtype=jnp.bool_)
        mask = mask.at[0].set(True)  # Only connection 0 is learnable
        params = init_network_params(
            weights=jnp.zeros(9),
            learnable_mask=mask,
        )
        config = LearningConfig(enabled=True, learning_rate=0.1)
        new_params = apply_hebbian_learning(state, params, config)
        # Connection 0 (S->O) should change, others should not
        assert float(new_params.weights[0]) != 0.0
        assert jnp.all(new_params.weights[1:] == 0.0)


class TestRunNetworkPhase:
    def test_output_shape(self):
        net_config = NetworkConfig()
        learn_config = LearningConfig()
        params = init_network_params()
        stimulus = jnp.zeros(100)
        pain = jnp.zeros(100)
        outputs, final_params = run_network_phase(
            params, stimulus, pain, net_config, learn_config
        )
        assert outputs.shape == (100,)

    def test_nonzero_input_produces_nonzero_output(self):
        net_config = NetworkConfig()
        learn_config = LearningConfig(enabled=False)
        weights = jnp.zeros(9)
        weights = weights.at[CONN_STIM_OUT].set(1.0)
        params = init_network_params(weights=weights)
        stimulus = jnp.ones(10)
        pain = jnp.zeros(10)
        outputs, _ = run_network_phase(params, stimulus, pain, net_config, learn_config)
        assert jnp.any(outputs != 0.0)

"""Tests for neural network implementation."""

import jax
import jax.numpy as jnp
import pytest

from habituation_evolution.config import LearningConfig, NetworkConfig, HebbianRule
from habituation_evolution.network import (
    NetworkParams,
    NetworkState,
    WEIGHT_FORWARD,
    WEIGHT_RECURRENT,
    WEIGHT_SELF_SENSORY,
    WEIGHT_SELF_MOTOR,
    get_eating_decision,
    init_network_params,
    init_network_state,
    motor_activation_scaled_sigmoid,
    motor_activation_tanh,
    network_step,
    run_network_lifespan,
    sensory_activation_fn,
    apply_hebbian_learning,
)


class TestActivationFunctions:
    """Tests for activation functions."""

    def test_sensory_linear(self):
        """Sensory activation should be linear (identity)."""
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert jnp.allclose(sensory_activation_fn(x), x)

    def test_motor_tanh_range(self):
        """Tanh should output in [-1, 1]."""
        x = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        y = motor_activation_tanh(x)
        assert jnp.all(y >= -1.0)
        assert jnp.all(y <= 1.0)

    def test_motor_tanh_zero(self):
        """Tanh(0) should be 0."""
        assert jnp.isclose(motor_activation_tanh(jnp.array(0.0)), 0.0)

    def test_motor_scaled_sigmoid_range(self):
        """Scaled sigmoid should output in [-1, 1]."""
        x = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        y = motor_activation_scaled_sigmoid(x)
        assert jnp.all(y >= -1.0)
        assert jnp.all(y <= 1.0)

    def test_motor_scaled_sigmoid_zero(self):
        """Scaled sigmoid(0) should be 0."""
        assert jnp.isclose(motor_activation_scaled_sigmoid(jnp.array(0.0)), 0.0)


class TestNetworkInitialization:
    """Tests for network initialization."""

    def test_init_state_zeros(self):
        state = init_network_state()
        assert jnp.isclose(state.sensory_activation, 0.0)
        assert jnp.isclose(state.motor_activation, 0.0)

    def test_init_state_batched(self):
        state = init_network_state(batch_size=10)
        assert state.sensory_activation.shape == (10,)
        assert state.motor_activation.shape == (10,)
        assert jnp.all(state.sensory_activation == 0.0)

    def test_init_params_defaults(self):
        params = init_network_params()
        assert params.weights.shape == (4,)
        assert params.biases.shape == (2,)
        assert params.learnable_mask.shape == (6,)
        assert jnp.all(params.weights == 0.0)
        assert jnp.all(params.biases == 0.0)
        assert jnp.all(~params.learnable_mask)

    def test_init_params_custom(self):
        weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        biases = jnp.array([0.5, -0.5])
        learnable = jnp.array([True, False, True, False, True, False])
        params = init_network_params(weights, biases, learnable)
        assert jnp.allclose(params.weights, weights)
        assert jnp.allclose(params.biases, biases)


class TestNetworkStep:
    """Tests for single network step."""

    def test_zero_weights_zero_output(self):
        """With zero weights, output should be near zero."""
        config = NetworkConfig()
        params = init_network_params()
        state = init_network_state()
        sensory_input = jnp.array(1.0)

        new_state, motor_output = network_step(state, sensory_input, params, config)

        # With zero forward weight and zero bias, motor should be ~0
        assert jnp.isclose(motor_output, 0.0, atol=0.01)

    def test_positive_forward_weight(self):
        """Positive forward weight should produce positive output for positive input."""
        config = NetworkConfig()
        weights = jnp.zeros(4).at[WEIGHT_FORWARD].set(2.0)
        params = init_network_params(weights=weights)
        state = init_network_state()
        sensory_input = jnp.array(1.0)  # Sweet (food)

        new_state, motor_output = network_step(state, sensory_input, params, config)

        # tanh(2.0 * 1.0) > 0
        assert motor_output > 0

    def test_negative_forward_weight(self):
        """Negative forward weight should produce negative output for positive input."""
        config = NetworkConfig()
        weights = jnp.zeros(4).at[WEIGHT_FORWARD].set(-2.0)
        params = init_network_params(weights=weights)
        state = init_network_state()
        sensory_input = jnp.array(1.0)

        new_state, motor_output = network_step(state, sensory_input, params, config)

        assert motor_output < 0

    def test_recurrent_connection(self):
        """Recurrent connection should influence next step."""
        config = NetworkConfig()
        weights = jnp.zeros(4)
        weights = weights.at[WEIGHT_FORWARD].set(1.0)
        weights = weights.at[WEIGHT_RECURRENT].set(0.5)
        params = init_network_params(weights=weights)

        # First step
        state = init_network_state()
        state1, motor1 = network_step(state, jnp.array(1.0), params, config)

        # Second step - recurrent should add to input
        state2, motor2 = network_step(state1, jnp.array(0.0), params, config)

        # With zero input but positive recurrent from previous motor,
        # we should still get some positive sensory activation
        assert state2.sensory_activation > 0

    def test_self_recurrent_sensory(self):
        """Self-recurrent on sensory should accumulate."""
        config = NetworkConfig()
        weights = jnp.zeros(4).at[WEIGHT_SELF_SENSORY].set(0.9)
        params = init_network_params(weights=weights)

        state = init_network_state()
        # Feed constant input and watch activation grow
        activations = []
        for _ in range(5):
            state, _ = network_step(state, jnp.array(1.0), params, config)
            activations.append(float(state.sensory_activation))

        # Activations should be increasing (accumulating)
        assert activations[-1] > activations[0]


class TestEatingDecision:
    """Tests for eating decision logic."""

    def test_positive_eats(self):
        assert get_eating_decision(jnp.array(0.5)) == True

    def test_negative_doesnt_eat(self):
        assert get_eating_decision(jnp.array(-0.5)) == False

    def test_zero_doesnt_eat(self):
        """Boundary case: exactly 0 should not eat."""
        assert get_eating_decision(jnp.array(0.0)) == False

    def test_batch_decisions(self):
        motor = jnp.array([0.5, -0.5, 0.1, -0.1])
        decisions = get_eating_decision(motor)
        expected = jnp.array([True, False, True, False])
        assert jnp.all(decisions == expected)


class TestHebbianLearning:
    """Tests for Hebbian learning."""

    def test_learning_disabled(self):
        """When disabled, params should not change."""
        config = LearningConfig(enabled=False)
        weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        params = init_network_params(
            weights=weights,
            learnable_mask=jnp.array([True, True, True, True, True, True]),
        )
        state = NetworkState(
            sensory_activation=jnp.array(1.0),
            motor_activation=jnp.array(0.5),
        )

        new_params = apply_hebbian_learning(state, params, config)
        assert jnp.allclose(new_params.weights, params.weights)

    def test_learning_updates_learnable(self):
        """Learning should update learnable weights."""
        config = LearningConfig(enabled=True, rule=HebbianRule.BASIC, learning_rate=0.1)
        weights = jnp.zeros(4)
        learnable = jnp.array([True, False, True, False, False, False])
        params = init_network_params(weights=weights, learnable_mask=learnable)
        state = NetworkState(
            sensory_activation=jnp.array(1.0),
            motor_activation=jnp.array(0.5),
        )

        new_params = apply_hebbian_learning(state, params, config)

        # Weight 0 (forward) should change: Δw = 0.1 * 1.0 * 0.5 = 0.05
        assert not jnp.isclose(new_params.weights[0], 0.0)
        # Weight 1 (non-learnable) should not change
        assert jnp.isclose(new_params.weights[1], 0.0)

    def test_basic_hebbian_rule(self):
        """Basic Hebbian: Δw = η * pre * post."""
        config = LearningConfig(enabled=True, rule=HebbianRule.BASIC, learning_rate=0.1)
        weights = jnp.zeros(4)
        learnable = jnp.ones(6, dtype=jnp.bool_)
        params = init_network_params(weights=weights, learnable_mask=learnable)

        # sensory=1.0, motor=0.5
        # Forward weight: pre=sensory=1.0, post=motor=0.5
        # Expected Δw = 0.1 * 1.0 * 0.5 = 0.05
        state = NetworkState(
            sensory_activation=jnp.array(1.0),
            motor_activation=jnp.array(0.5),
        )

        new_params = apply_hebbian_learning(state, params, config)
        assert jnp.isclose(new_params.weights[WEIGHT_FORWARD], 0.05)


class TestLifespanSimulation:
    """Tests for running network over lifespan."""

    def test_output_shapes(self):
        """Outputs should have correct shapes."""
        config_net = NetworkConfig()
        config_learn = LearningConfig(enabled=False)
        params = init_network_params(
            weights=jnp.array([1.0, 0.0, 0.0, 0.0])
        )
        inputs = jnp.ones(100)

        motor_outputs, decisions, final_params = run_network_lifespan(
            params, inputs, config_net, config_learn
        )

        assert motor_outputs.shape == (100,)
        assert decisions.shape == (100,)

    def test_always_eat_network(self):
        """Network with strong positive bias should always eat."""
        config_net = NetworkConfig()
        config_learn = LearningConfig(enabled=False)
        params = init_network_params(biases=jnp.array([0.0, 5.0]))  # Strong motor bias
        inputs = jnp.zeros(100)  # Neutral input

        motor_outputs, decisions, _ = run_network_lifespan(
            params, inputs, config_net, config_learn
        )

        # Should always decide to eat
        assert jnp.all(decisions)

    def test_never_eat_network(self):
        """Network with strong negative bias should never eat."""
        config_net = NetworkConfig()
        config_learn = LearningConfig(enabled=False)
        params = init_network_params(biases=jnp.array([0.0, -5.0]))
        inputs = jnp.zeros(100)

        motor_outputs, decisions, _ = run_network_lifespan(
            params, inputs, config_net, config_learn
        )

        # Should never decide to eat
        assert not jnp.any(decisions)

    def test_sensory_following_network(self):
        """Network with forward weight should follow sensory input."""
        config_net = NetworkConfig()
        config_learn = LearningConfig(enabled=False)
        params = init_network_params(weights=jnp.array([3.0, 0.0, 0.0, 0.0]))

        # Alternating positive/negative inputs
        inputs = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0])

        motor_outputs, decisions, _ = run_network_lifespan(
            params, inputs, config_net, config_learn
        )

        # Should mostly follow input sign
        expected_decisions = inputs > 0
        # Allow some slack due to activation function saturation
        assert jnp.mean(decisions == expected_decisions) > 0.8

"""Neural network implementation for the ALICE threat discrimination simulation.

The network consists of N+2 neurons (for N stimulus channels):
- N stimulus input neurons (linear activation)
- 1 pain input neuron (linear activation)
- 1 output neuron (tanh activation, output in [-1, +1])

For the default N=1 case, there are 3 neurons with full connectivity
(9 weights + 3 biases = 12 parameters).

Output interpretation:
  +1 = full withdrawal (max protection, no eating)
  -1 = full eating (max health gain, no protection)
   0 = neutral (no eating, no protection)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .config import HebbianRule, LearningConfig, NetworkConfig


# ============================================================================
# Network Data Structures
# ============================================================================


class NetworkParams(NamedTuple):
    """Network parameters (weights and biases).

    For N=1 (3 neurons):
        weights: shape (9,) - full connectivity
        biases: shape (3,) - one per neuron
        learnable_mask: shape (12,) - which params are learnable (9 weights + 3 biases)
    """

    weights: Array
    biases: Array
    learnable_mask: Array


class NetworkState(NamedTuple):
    """Network activation state.

    activations: shape (num_neurons,) - one activation per neuron
    For N=1: [stimulus_activation, pain_activation, output_activation]
    """

    activations: Array


# ============================================================================
# Connection index constants for N=1 (3 neurons: S=0, P=1, O=2)
# ============================================================================

# Forward connections (input -> output)
CONN_STIM_OUT = 0   # stimulus -> output
CONN_PAIN_OUT = 1   # pain -> output

# Recurrent connections (output -> input, 1-step delay)
CONN_OUT_STIM = 2   # output -> stimulus
CONN_OUT_PAIN = 3   # output -> pain

# Self-recurrent connections (1-step delay)
CONN_STIM_STIM = 4  # stimulus -> stimulus
CONN_PAIN_PAIN = 5  # pain -> pain
CONN_OUT_OUT = 6     # output -> output

# Cross-input connections (1-step delay)
CONN_STIM_PAIN = 7  # stimulus -> pain
CONN_PAIN_STIM = 8  # pain -> stimulus

# Neuron indices
NEURON_STIM = 0
NEURON_PAIN = 1
NEURON_OUTPUT = 2

# Counts for N=1
NUM_CONNECTIONS_N1 = 9
NUM_NEURONS_N1 = 3
NUM_PARAMS_N1 = 12  # 9 weights + 3 biases

# Pre/post neuron index for each connection (source, target)
# Used for Hebbian learning
CONNECTION_MAP = [
    (NEURON_STIM, NEURON_OUTPUT),   # 0: S -> O
    (NEURON_PAIN, NEURON_OUTPUT),   # 1: P -> O
    (NEURON_OUTPUT, NEURON_STIM),   # 2: O -> S
    (NEURON_OUTPUT, NEURON_PAIN),   # 3: O -> P
    (NEURON_STIM, NEURON_STIM),     # 4: S -> S
    (NEURON_PAIN, NEURON_PAIN),     # 5: P -> P
    (NEURON_OUTPUT, NEURON_OUTPUT), # 6: O -> O
    (NEURON_STIM, NEURON_PAIN),     # 7: S -> P
    (NEURON_PAIN, NEURON_STIM),     # 8: P -> S
]


# ============================================================================
# Network Initialization
# ============================================================================


def init_network_state(num_neurons: int = NUM_NEURONS_N1) -> NetworkState:
    """Initialize network state with zero activations."""
    return NetworkState(activations=jnp.zeros(num_neurons))


def init_network_params(
    weights: Array | None = None,
    biases: Array | None = None,
    learnable_mask: Array | None = None,
    num_connections: int = NUM_CONNECTIONS_N1,
    num_neurons: int = NUM_NEURONS_N1,
) -> NetworkParams:
    """Initialize network parameters."""
    if weights is None:
        weights = jnp.zeros(num_connections)
    if biases is None:
        biases = jnp.zeros(num_neurons)
    if learnable_mask is None:
        learnable_mask = jnp.zeros(num_connections + num_neurons, dtype=jnp.bool_)

    return NetworkParams(weights=weights, biases=biases, learnable_mask=learnable_mask)


# ============================================================================
# Network Forward Pass (N=1 specialization)
# ============================================================================


def network_step(
    state: NetworkState,
    stimulus_input: Array,
    pain_input: Array,
    params: NetworkParams,
) -> tuple[NetworkState, Array]:
    """Execute single timestep of network update.

    Update order:
    1. Compute input neurons (stimulus, pain) from external inputs + recurrent + bias
    2. Compute output neuron from fresh input activations + self-recurrent + bias
    3. Return new state and output value

    Args:
        state: Current network state (previous activations)
        stimulus_input: External stimulus input (scalar for N=1)
        pain_input: External pain input (scalar)
        params: Network parameters

    Returns:
        Tuple of (new_state, output_value in [-1, +1])
    """
    # Previous activations
    prev_s = state.activations[NEURON_STIM]
    prev_p = state.activations[NEURON_PAIN]
    prev_o = state.activations[NEURON_OUTPUT]

    w = params.weights
    b = params.biases

    # New stimulus activation (linear)
    new_s = (
        stimulus_input
        + w[CONN_OUT_STIM] * prev_o
        + w[CONN_STIM_STIM] * prev_s
        + w[CONN_PAIN_STIM] * prev_p
        + b[NEURON_STIM]
    )

    # New pain activation (linear)
    new_p = (
        pain_input
        + w[CONN_OUT_PAIN] * prev_o
        + w[CONN_PAIN_PAIN] * prev_p
        + w[CONN_STIM_PAIN] * prev_s
        + b[NEURON_PAIN]
    )

    # New output activation (tanh, uses fresh input values)
    new_o = jnp.tanh(
        w[CONN_STIM_OUT] * new_s
        + w[CONN_PAIN_OUT] * new_p
        + w[CONN_OUT_OUT] * prev_o
        + b[NEURON_OUTPUT]
    )

    new_state = NetworkState(
        activations=jnp.array([new_s, new_p, new_o])
    )

    return new_state, new_o


# ============================================================================
# Hebbian Learning
# ============================================================================


def apply_hebbian_basic(
    pre: Array, post: Array, weight: Array, learning_rate: float
) -> Array:
    """Basic Hebbian: dw = eta * pre * post."""
    return weight + learning_rate * pre * post


def apply_hebbian_oja(
    pre: Array, post: Array, weight: Array, learning_rate: float
) -> Array:
    """Oja's rule: dw = eta * post * (pre - post * w)."""
    return weight + learning_rate * post * (pre - post * weight)


def apply_hebbian_normalized(
    pre: Array, post: Array, weight: Array, learning_rate: float, weights_norm: Array
) -> Array:
    """Normalized Hebbian: dw = eta * pre * post / ||w||."""
    norm = jnp.maximum(weights_norm, 1e-8)
    return weight + learning_rate * pre * post / norm


def apply_hebbian_learning(
    state: NetworkState,
    params: NetworkParams,
    config: LearningConfig,
) -> NetworkParams:
    """Apply Hebbian learning to learnable connections.

    Args:
        state: Current network state (activations after forward pass)
        params: Current network parameters
        config: Learning configuration

    Returns:
        Updated network parameters
    """
    if not config.enabled:
        return params

    activations = state.activations
    lr = config.learning_rate

    # Build pre/post activation arrays for all 9 connections
    pre_indices = jnp.array([src for src, _ in CONNECTION_MAP])
    post_indices = jnp.array([tgt for _, tgt in CONNECTION_MAP])

    pre_activations = activations[pre_indices]
    post_activations = activations[post_indices]

    # Get learning function
    if config.rule == HebbianRule.BASIC:
        learn_fn = lambda pre, post, w: apply_hebbian_basic(pre, post, w, lr)
    elif config.rule == HebbianRule.OJA:
        learn_fn = lambda pre, post, w: apply_hebbian_oja(pre, post, w, lr)
    else:  # NORMALIZED
        weights_norm = jnp.linalg.norm(params.weights)
        learn_fn = lambda pre, post, w: apply_hebbian_normalized(
            pre, post, w, lr, weights_norm
        )

    # Update weights where learnable
    num_conns = params.weights.shape[0]
    new_weights = jnp.where(
        params.learnable_mask[:num_conns],
        jax.vmap(learn_fn)(pre_activations, post_activations, params.weights),
        params.weights,
    )

    # Update biases where learnable (bias = connection from constant 1.0)
    num_neurons = params.biases.shape[0]
    bias_pre = jnp.ones(num_neurons)
    bias_post = activations[:num_neurons]

    new_biases = jnp.where(
        params.learnable_mask[num_conns:],
        jax.vmap(learn_fn)(bias_pre, bias_post, params.biases),
        params.biases,
    )

    # Clip to prevent explosion
    clip = config.weight_clip
    new_weights = jnp.clip(new_weights, -clip, clip)
    new_biases = jnp.clip(new_biases, -clip, clip)

    return NetworkParams(
        weights=new_weights,
        biases=new_biases,
        learnable_mask=params.learnable_mask,
    )


# ============================================================================
# Full Phase Simulation
# ============================================================================


def run_network_phase(
    params: NetworkParams,
    stimulus_inputs: Array,
    pain_inputs: Array,
    net_config: NetworkConfig,
    learn_config: LearningConfig,
) -> tuple[Array, NetworkParams]:
    """Run network over an entire evaluation phase using lax.scan.

    Args:
        params: Initial network parameters
        stimulus_inputs: Stimulus input sequence, shape (lifetime,) for N=1
        pain_inputs: Pain input sequence, shape (lifetime,)
        net_config: Network configuration
        learn_config: Learning configuration

    Returns:
        Tuple of:
        - outputs: Motor outputs for each timestep, shape (lifetime,)
        - final_params: Network parameters after learning
    """

    def step_fn(carry, inputs):
        state, params = carry
        stimulus, pain = inputs

        # Network forward pass
        new_state, output = network_step(state, stimulus, pain, params)

        # Apply Hebbian learning
        new_params = apply_hebbian_learning(new_state, params, learn_config)

        return (new_state, new_params), output

    # Initialize state
    initial_state = init_network_state(net_config.num_neurons)

    # Run through all timesteps
    (final_state, final_params), outputs = jax.lax.scan(
        step_fn, (initial_state, params), (stimulus_inputs, pain_inputs)
    )

    return outputs, final_params

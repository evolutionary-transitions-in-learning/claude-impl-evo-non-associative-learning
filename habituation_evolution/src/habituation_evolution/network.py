"""Neural network implementation for the habituation/sensitization simulation.

The network consists of two units:
1. Sensory unit: receives smell input (sweet/sour), linear activation
2. Motor unit: determines eating behavior, logistic activation [-1, +1]

Connections:
- Forward: sensory -> motor (immediate)
- Recurrent: motor -> sensory (1 timestep delay)
- Self-recurrent: sensory -> sensory (1 timestep delay)
- Self-recurrent: motor -> motor (1 timestep delay)

Plus biases on both units.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .config import HebbianRule, LearningConfig, MotorActivation, NetworkConfig


# ============================================================================
# Network Data Structures
# ============================================================================


class NetworkParams(NamedTuple):
    """Network parameters (weights and biases).

    Weight indices:
        0: forward (sensory -> motor)
        1: recurrent (motor -> sensory)
        2: self_sensory (sensory -> sensory)
        3: self_motor (motor -> motor)

    Bias indices:
        0: sensory bias
        1: motor bias
    """

    weights: Array  # Shape: (4,)
    biases: Array  # Shape: (2,)
    learnable_mask: Array  # Shape: (6,) - which params are learnable (4 weights + 2 biases)


class NetworkState(NamedTuple):
    """Network activation state.

    Attributes:
        sensory_activation: Current sensory unit activation
        motor_activation: Current motor unit activation
    """

    sensory_activation: Array  # Shape: () for single, (batch,) for batched
    motor_activation: Array  # Shape: () for single, (batch,) for batched


# Weight/bias index constants
WEIGHT_FORWARD = 0  # sensory -> motor
WEIGHT_RECURRENT = 1  # motor -> sensory (feedback)
WEIGHT_SELF_SENSORY = 2  # sensory -> sensory
WEIGHT_SELF_MOTOR = 3  # motor -> motor

BIAS_SENSORY = 0
BIAS_MOTOR = 1


# ============================================================================
# Activation Functions
# ============================================================================


def sensory_activation_fn(x: Array) -> Array:
    """Sensory unit activation function (linear).

    Args:
        x: Input to sensory unit

    Returns:
        Activation value (unchanged)
    """
    return x


def motor_activation_tanh(x: Array) -> Array:
    """Motor unit activation using tanh.

    Output range: [-1, +1]

    Args:
        x: Input to motor unit

    Returns:
        Activation value in [-1, +1]
    """
    return jnp.tanh(x)


def motor_activation_scaled_sigmoid(x: Array) -> Array:
    """Motor unit activation using scaled sigmoid.

    Output range: [-1, +1]
    Formula: 2 * sigmoid(x) - 1

    Args:
        x: Input to motor unit

    Returns:
        Activation value in [-1, +1]
    """
    return 2.0 * jax.nn.sigmoid(x) - 1.0


def get_motor_activation_fn(config: NetworkConfig):
    """Get the motor activation function based on config.

    Args:
        config: Network configuration

    Returns:
        Activation function
    """
    if config.motor_activation == MotorActivation.TANH:
        return motor_activation_tanh
    else:
        return motor_activation_scaled_sigmoid


# ============================================================================
# Network Initialization
# ============================================================================


def init_network_state(batch_size: int | None = None) -> NetworkState:
    """Initialize network state with zero activations.

    Args:
        batch_size: If provided, create batched state

    Returns:
        NetworkState with zero activations
    """
    if batch_size is None:
        return NetworkState(
            sensory_activation=jnp.array(0.0),
            motor_activation=jnp.array(0.0),
        )
    else:
        return NetworkState(
            sensory_activation=jnp.zeros(batch_size),
            motor_activation=jnp.zeros(batch_size),
        )


def init_network_params(
    weights: Array | None = None,
    biases: Array | None = None,
    learnable_mask: Array | None = None,
) -> NetworkParams:
    """Initialize network parameters.

    Args:
        weights: Initial weights (4,), defaults to zeros
        biases: Initial biases (2,), defaults to zeros
        learnable_mask: Which params are learnable (6,), defaults to all False

    Returns:
        NetworkParams
    """
    if weights is None:
        weights = jnp.zeros(4)
    if biases is None:
        biases = jnp.zeros(2)
    if learnable_mask is None:
        learnable_mask = jnp.zeros(6, dtype=jnp.bool_)

    return NetworkParams(weights=weights, biases=biases, learnable_mask=learnable_mask)


# ============================================================================
# Network Forward Pass
# ============================================================================


def network_step(
    state: NetworkState,
    sensory_input: Array,
    params: NetworkParams,
    config: NetworkConfig,
) -> tuple[NetworkState, Array]:
    """Execute single timestep of network update.

    The update order:
    1. Compute new sensory activation from input + recurrent + self-recurrent + bias
    2. Compute new motor activation from sensory + self-recurrent + bias
    3. Return new state and motor output

    Args:
        state: Current network state (previous activations)
        sensory_input: External sensory input for this timestep
        params: Network parameters
        config: Network configuration

    Returns:
        Tuple of (new_state, motor_output)
    """
    motor_fn = get_motor_activation_fn(config)

    # Unpack previous activations
    prev_sensory = state.sensory_activation
    prev_motor = state.motor_activation

    # Unpack weights and biases
    w_forward = params.weights[WEIGHT_FORWARD]
    w_recurrent = params.weights[WEIGHT_RECURRENT]
    w_self_sensory = params.weights[WEIGHT_SELF_SENSORY]
    w_self_motor = params.weights[WEIGHT_SELF_MOTOR]
    b_sensory = params.biases[BIAS_SENSORY]
    b_motor = params.biases[BIAS_MOTOR]

    # Compute new sensory activation
    # Input: external input + feedback from motor + self-recurrent + bias
    sensory_input_total = (
        sensory_input
        + w_recurrent * prev_motor
        + w_self_sensory * prev_sensory
        + b_sensory
    )
    new_sensory = sensory_activation_fn(sensory_input_total)

    # Compute new motor activation
    # Input: forward from sensory + self-recurrent + bias
    motor_input_total = (
        w_forward * new_sensory + w_self_motor * prev_motor + b_motor
    )
    new_motor = motor_fn(motor_input_total)

    new_state = NetworkState(
        sensory_activation=new_sensory,
        motor_activation=new_motor,
    )

    return new_state, new_motor


def get_eating_decision(motor_activation: Array) -> Array:
    """Convert motor activation to binary eat/don't-eat decision.

    Args:
        motor_activation: Motor unit output in [-1, +1]

    Returns:
        Boolean array: True = eat, False = don't eat
    """
    return motor_activation > 0.0


# ============================================================================
# Hebbian Learning
# ============================================================================


def apply_hebbian_basic(
    pre: Array, post: Array, weight: Array, learning_rate: float
) -> Array:
    """Basic Hebbian learning rule: Δw = η * pre * post.

    Args:
        pre: Pre-synaptic activation
        post: Post-synaptic activation
        weight: Current weight value
        learning_rate: Learning rate η

    Returns:
        Updated weight
    """
    delta = learning_rate * pre * post
    return weight + delta


def apply_hebbian_oja(
    pre: Array, post: Array, weight: Array, learning_rate: float
) -> Array:
    """Oja's rule: Δw = η * post * (pre - post * w).

    This rule is self-normalizing.

    Args:
        pre: Pre-synaptic activation
        post: Post-synaptic activation
        weight: Current weight value
        learning_rate: Learning rate η

    Returns:
        Updated weight
    """
    delta = learning_rate * post * (pre - post * weight)
    return weight + delta


def apply_hebbian_normalized(
    pre: Array, post: Array, weight: Array, learning_rate: float, weights_norm: Array
) -> Array:
    """Normalized Hebbian: Δw = η * pre * post / ||w||.

    Args:
        pre: Pre-synaptic activation
        post: Post-synaptic activation
        weight: Current weight value
        learning_rate: Learning rate η
        weights_norm: L2 norm of all weights

    Returns:
        Updated weight
    """
    # Avoid division by zero
    norm = jnp.maximum(weights_norm, 1e-8)
    delta = learning_rate * pre * post / norm
    return weight + delta


def apply_hebbian_learning(
    state: NetworkState,
    params: NetworkParams,
    config: LearningConfig,
) -> NetworkParams:
    """Apply Hebbian learning to learnable connections.

    The learning is applied based on current activations:
    - Forward weight: sensory (pre) -> motor (post)
    - Recurrent weight: motor (pre) -> sensory (post)
    - Self-sensory: sensory (pre) -> sensory (post)
    - Self-motor: motor (pre) -> motor (post)
    - Biases: treated as connections from constant input of 1.0

    Args:
        state: Current network state
        params: Current network parameters
        config: Learning configuration

    Returns:
        Updated network parameters
    """
    if not config.enabled:
        return params

    sensory = state.sensory_activation
    motor = state.motor_activation
    lr = config.learning_rate

    # Get learning function based on rule
    if config.rule == HebbianRule.BASIC:
        learn_fn = lambda pre, post, w: apply_hebbian_basic(pre, post, w, lr)
    elif config.rule == HebbianRule.OJA:
        learn_fn = lambda pre, post, w: apply_hebbian_oja(pre, post, w, lr)
    else:  # NORMALIZED
        weights_norm = jnp.linalg.norm(params.weights)
        learn_fn = lambda pre, post, w: apply_hebbian_normalized(pre, post, w, lr, weights_norm)

    # Define pre/post for each connection
    # Weight 0: forward (sensory -> motor)
    # Weight 1: recurrent (motor -> sensory)
    # Weight 2: self_sensory (sensory -> sensory)
    # Weight 3: self_motor (motor -> motor)
    pre_activations = jnp.array([sensory, motor, sensory, motor])
    post_activations = jnp.array([motor, sensory, sensory, motor])

    # Update weights where learnable
    new_weights = jnp.where(
        params.learnable_mask[:4],
        jax.vmap(learn_fn)(pre_activations, post_activations, params.weights),
        params.weights,
    )

    # Update biases where learnable (bias = connection from constant 1.0)
    # Bias 0: sensory bias (pre=1.0, post=sensory)
    # Bias 1: motor bias (pre=1.0, post=motor)
    bias_pre = jnp.array([1.0, 1.0])
    bias_post = jnp.array([sensory, motor])

    new_biases = jnp.where(
        params.learnable_mask[4:],
        jax.vmap(learn_fn)(bias_pre, bias_post, params.biases),
        params.biases,
    )

    return NetworkParams(
        weights=new_weights,
        biases=new_biases,
        learnable_mask=params.learnable_mask,
    )


# ============================================================================
# Full Lifespan Simulation
# ============================================================================


def run_network_lifespan(
    params: NetworkParams,
    sensory_inputs: Array,
    net_config: NetworkConfig,
    learn_config: LearningConfig,
) -> tuple[Array, Array, NetworkParams]:
    """Run network over entire lifespan.

    Args:
        params: Initial network parameters
        sensory_inputs: Sensory input sequence, shape (lifespan,)
        net_config: Network configuration
        learn_config: Learning configuration

    Returns:
        Tuple of:
        - motor_outputs: Motor activations for each timestep, shape (lifespan,)
        - decisions: Eating decisions for each timestep, shape (lifespan,)
        - final_params: Network parameters after learning
    """

    def step_fn(carry, sensory_input):
        """Single step function for lax.scan."""
        state, params = carry

        # Network forward pass
        new_state, motor_output = network_step(state, sensory_input, params, net_config)

        # Apply Hebbian learning
        new_params = apply_hebbian_learning(new_state, params, learn_config)

        # Get eating decision
        decision = get_eating_decision(motor_output)

        return (new_state, new_params), (motor_output, decision)

    # Initialize state
    initial_state = init_network_state()

    # Run through all timesteps
    (final_state, final_params), (motor_outputs, decisions) = jax.lax.scan(
        step_fn, (initial_state, params), sensory_inputs
    )

    return motor_outputs, decisions, final_params


# Batched version using vmap
def run_network_lifespan_batch(
    params_batch: NetworkParams,
    sensory_inputs_batch: Array,
    net_config: NetworkConfig,
    learn_config: LearningConfig,
) -> tuple[Array, Array, NetworkParams]:
    """Run networks over entire lifespan for a batch of creatures.

    Args:
        params_batch: Network parameters for each creature, weights shape (batch, 4)
        sensory_inputs_batch: Sensory inputs for each creature, shape (batch, lifespan)
        net_config: Network configuration
        learn_config: Learning configuration

    Returns:
        Tuple of:
        - motor_outputs: shape (batch, lifespan)
        - decisions: shape (batch, lifespan)
        - final_params: parameters after learning, weights shape (batch, 4)
    """
    # vmap over the batch dimension
    batch_run = jax.vmap(
        lambda p, s: run_network_lifespan(p, s, net_config, learn_config),
        in_axes=(NetworkParams(0, 0, 0), 0),
        out_axes=(0, 0, NetworkParams(0, 0, 0)),
    )

    return batch_run(params_batch, sensory_inputs_batch)

"""Genetic algorithm components for the ALICE threat discrimination simulation.

This module handles:
- Genotype encoding/decoding (81 bits for N=1)
- Genetic operators (crossover, mutation)
- Population management

Genotype Structure (81 bits for N=1, 3 neurons):
- 9 connections x 7 bits each = 63 bits
  - Bit 0: connection present (1) or absent (0)
  - Bit 1: sign (1 = positive, 0 = negative)
  - Bits 2-5: magnitude index (0-15)
  - Bit 6: learnable flag (1 = learnable, 0 = fixed)
- 3 biases x 6 bits each = 18 bits
  - Bit 0: sign (1 = positive, 0 = negative)
  - Bits 1-4: magnitude index (0-15)
  - Bit 5: learnable flag (1 = learnable, 0 = fixed)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from .config import NetworkConfig
from .network import NetworkParams


# ============================================================================
# Genotype Constants
# ============================================================================

BITS_PER_CONNECTION = 7
BITS_PER_BIAS = 6

# Bit offsets within connection encoding
CONN_BIT_PRESENT = 0
CONN_BIT_SIGN = 1
CONN_BITS_MAG_START = 2
CONN_BITS_MAG_END = 6
CONN_BIT_LEARNABLE = 6

# Bit offsets within bias encoding
BIAS_BIT_SIGN = 0
BIAS_BITS_MAG_START = 1
BIAS_BITS_MAG_END = 5
BIAS_BIT_LEARNABLE = 5


def compute_genotype_length(config: NetworkConfig) -> int:
    """Compute genotype length from network config."""
    return config.num_connections * BITS_PER_CONNECTION + config.num_biases * BITS_PER_BIAS


# Default for N=1: 9*7 + 3*6 = 63 + 18 = 81
GENOTYPE_LENGTH_N1 = 81


# ============================================================================
# Genotype Creation
# ============================================================================


def create_random_genotype(key: PRNGKey, genotype_length: int = GENOTYPE_LENGTH_N1) -> Array:
    """Generate a random binary genotype."""
    return jax.random.bernoulli(key, p=0.5, shape=(genotype_length,)).astype(jnp.int32)


def create_random_population(
    key: PRNGKey, pop_size: int, genotype_length: int = GENOTYPE_LENGTH_N1
) -> Array:
    """Generate a population of random genotypes."""
    keys = jax.random.split(key, pop_size)
    return jax.vmap(lambda k: create_random_genotype(k, genotype_length))(keys)


# ============================================================================
# Genotype Decoding
# ============================================================================


def bits_to_int(bits: Array) -> Array:
    """Convert binary array to integer."""
    powers = 2 ** jnp.arange(len(bits))
    return jnp.sum(bits * powers)


def decode_magnitude(magnitude_bits: Array, config: NetworkConfig) -> Array:
    """Decode magnitude bits to weight value."""
    magnitudes = config.get_weight_magnitudes()
    index = bits_to_int(magnitude_bits)
    index = jnp.clip(index, 0, len(magnitudes) - 1)
    return magnitudes[index]


def decode_connection(conn_bits: Array, config: NetworkConfig) -> tuple[Array, Array]:
    """Decode connection bits to weight value and learnable flag."""
    present = conn_bits[CONN_BIT_PRESENT]
    sign = conn_bits[CONN_BIT_SIGN]
    mag_bits = conn_bits[CONN_BITS_MAG_START:CONN_BITS_MAG_END]
    learnable = conn_bits[CONN_BIT_LEARNABLE]

    magnitude = decode_magnitude(mag_bits, config)
    sign_multiplier = 2.0 * sign - 1.0
    weight = present * sign_multiplier * magnitude

    return weight, learnable.astype(jnp.bool_)


def decode_bias(bias_bits: Array, config: NetworkConfig) -> tuple[Array, Array]:
    """Decode bias bits to bias value and learnable flag."""
    sign = bias_bits[BIAS_BIT_SIGN]
    mag_bits = bias_bits[BIAS_BITS_MAG_START:BIAS_BITS_MAG_END]
    learnable = bias_bits[BIAS_BIT_LEARNABLE]

    magnitude = decode_magnitude(mag_bits, config)
    sign_multiplier = 2.0 * sign - 1.0
    bias = sign_multiplier * magnitude

    return bias, learnable.astype(jnp.bool_)


def decode_genotype(genotype: Array, config: NetworkConfig) -> NetworkParams:
    """Convert binary genotype to network parameters.

    For N=1: 9 connections (7 bits each) + 3 biases (6 bits each) = 81 bits.

    Returns:
        NetworkParams with decoded weights, biases, and learnable mask
    """
    num_conns = config.num_connections
    num_biases = config.num_biases

    weights = []
    biases = []
    learnable_mask = []

    # Decode connections
    offset = 0
    for i in range(num_conns):
        conn_bits = genotype[offset : offset + BITS_PER_CONNECTION]
        weight, learnable = decode_connection(conn_bits, config)
        weights.append(weight)
        learnable_mask.append(learnable)
        offset += BITS_PER_CONNECTION

    # Decode biases
    for i in range(num_biases):
        bias_bits = genotype[offset : offset + BITS_PER_BIAS]
        bias, learnable = decode_bias(bias_bits, config)
        biases.append(bias)
        learnable_mask.append(learnable)
        offset += BITS_PER_BIAS

    return NetworkParams(
        weights=jnp.array(weights),
        biases=jnp.array(biases),
        learnable_mask=jnp.array(learnable_mask),
    )


def decode_population(population: Array, config: NetworkConfig) -> NetworkParams:
    """Decode entire population of genotypes to network parameters."""
    return jax.vmap(lambda g: decode_genotype(g, config))(population)


# ============================================================================
# Genotype Encoding (for testing/analysis)
# ============================================================================


def int_to_bits(value: int, num_bits: int) -> Array:
    """Convert integer to binary array."""
    return jnp.array([(value >> i) & 1 for i in range(num_bits)], dtype=jnp.int32)


def encode_connection(
    weight: float, learnable: bool, config: NetworkConfig
) -> Array:
    """Encode weight and learnable flag to connection bits."""
    magnitudes = config.get_weight_magnitudes()

    present = jnp.array(abs(weight) > 1e-8, dtype=jnp.int32)
    sign = jnp.array(weight >= 0, dtype=jnp.int32)
    magnitude = jnp.abs(jnp.array(weight))
    mag_index = jnp.argmin(jnp.abs(magnitudes - magnitude))

    bits = jnp.zeros(BITS_PER_CONNECTION, dtype=jnp.int32)
    bits = bits.at[CONN_BIT_PRESENT].set(present)
    bits = bits.at[CONN_BIT_SIGN].set(sign)
    bits = bits.at[CONN_BITS_MAG_START:CONN_BITS_MAG_END].set(int_to_bits(mag_index, 4))
    bits = bits.at[CONN_BIT_LEARNABLE].set(jnp.array(learnable, dtype=jnp.int32))

    return bits


def encode_bias(bias: float, learnable: bool, config: NetworkConfig) -> Array:
    """Encode bias and learnable flag to bias bits."""
    magnitudes = config.get_weight_magnitudes()

    sign = jnp.array(bias >= 0, dtype=jnp.int32)
    magnitude = jnp.abs(jnp.array(bias))
    mag_index = jnp.argmin(jnp.abs(magnitudes - magnitude))

    bits = jnp.zeros(BITS_PER_BIAS, dtype=jnp.int32)
    bits = bits.at[BIAS_BIT_SIGN].set(sign)
    bits = bits.at[BIAS_BITS_MAG_START:BIAS_BITS_MAG_END].set(int_to_bits(mag_index, 4))
    bits = bits.at[BIAS_BIT_LEARNABLE].set(jnp.array(learnable, dtype=jnp.int32))

    return bits


def encode_genotype(params: NetworkParams, config: NetworkConfig) -> Array:
    """Convert network parameters back to genotype."""
    num_conns = config.num_connections
    parts = []

    for i in range(num_conns):
        conn_bits = encode_connection(
            float(params.weights[i]), bool(params.learnable_mask[i]), config
        )
        parts.append(conn_bits)

    for i in range(config.num_biases):
        bias_bits = encode_bias(
            float(params.biases[i]), bool(params.learnable_mask[num_conns + i]), config
        )
        parts.append(bias_bits)

    return jnp.concatenate(parts)


# ============================================================================
# Genetic Operators
# ============================================================================


def two_point_crossover(
    key: PRNGKey, parent1: Array, parent2: Array
) -> tuple[Array, Array]:
    """Two-point crossover producing two offspring."""
    genotype_length = parent1.shape[0]
    key1, key2 = jax.random.split(key)
    point1 = jax.random.randint(key1, (), 0, genotype_length)
    point2 = jax.random.randint(key2, (), 0, genotype_length)

    # Ensure point1 <= point2
    point1, point2 = jnp.minimum(point1, point2), jnp.maximum(point1, point2)

    # Create mask for middle section
    indices = jnp.arange(genotype_length)
    middle_mask = (indices >= point1) & (indices < point2)

    offspring1 = jnp.where(middle_mask, parent2, parent1)
    offspring2 = jnp.where(middle_mask, parent1, parent2)

    return offspring1, offspring2


def point_mutation(key: PRNGKey, genotype: Array, rate: float) -> Array:
    """Apply point mutation with given per-bit rate."""
    flip_mask = jax.random.bernoulli(key, p=rate, shape=genotype.shape)
    return jnp.where(flip_mask, 1 - genotype, genotype)


# ============================================================================
# Population Analysis
# ============================================================================


def compute_genotype_diversity(population: Array) -> float:
    """Compute mean pairwise normalized Hamming distance.

    Returns a value in [0, 1] where 0 = all identical, 0.5 = random.
    """
    import numpy as np

    pop_np = np.array(population)
    pop_size, geno_len = pop_np.shape
    if pop_size < 2:
        return 0.0

    # Pairwise XOR: count differing bits
    diffs = pop_np[:, None, :] != pop_np[None, :, :]
    hamming = np.sum(diffs, axis=-1)  # (pop_size, pop_size)

    # Mean of upper triangle (exclude diagonal)
    mask = np.triu(np.ones((pop_size, pop_size), dtype=bool), k=1)
    mean_distance = np.sum(hamming[mask]) / (np.sum(mask) * geno_len)
    return float(mean_distance)


def compute_pop_learnable_fractions(population: Array, config: NetworkConfig) -> "np.ndarray":
    """Compute fraction of population with each parameter set to learnable.

    Returns array of shape (num_params,) with values in [0, 1].
    """
    import numpy as np

    pop_np = np.array(population)
    num_conns = config.num_connections
    num_biases = config.num_biases

    fractions = []
    offset = 0
    for _ in range(num_conns):
        learnable_bits = pop_np[:, offset + CONN_BIT_LEARNABLE]
        fractions.append(np.mean(learnable_bits))
        offset += BITS_PER_CONNECTION

    for _ in range(num_biases):
        learnable_bits = pop_np[:, offset + BIAS_BIT_LEARNABLE]
        fractions.append(np.mean(learnable_bits))
        offset += BITS_PER_BIAS

    return np.array(fractions)

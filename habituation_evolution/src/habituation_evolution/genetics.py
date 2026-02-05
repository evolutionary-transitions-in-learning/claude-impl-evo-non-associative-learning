"""Genetic algorithm components for the habituation/sensitization simulation.

This module handles:
- Genotype encoding/decoding
- Genetic operators (selection, crossover, mutation)
- Population management

Genotype Structure (40 bits total):
- 4 connections × 7 bits each = 28 bits
  - Bit 0: connection present (1) or absent (0)
  - Bit 1: sign (1 = positive, 0 = negative)
  - Bits 2-5: magnitude index (0-15)
  - Bit 6: learnable flag (1 = learnable, 0 = fixed)
- 2 biases × 6 bits each = 12 bits
  - Bit 0: sign (1 = positive, 0 = negative)
  - Bits 1-4: magnitude index (0-15)
  - Bit 5: learnable flag (1 = learnable, 0 = fixed)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from .config import GeneticConfig, NetworkConfig
from .network import NetworkParams


# ============================================================================
# Genotype Constants
# ============================================================================

# Bits per connection: present(1) + sign(1) + magnitude(4) + learnable(1) = 7
BITS_PER_CONNECTION = 7
NUM_CONNECTIONS = 4

# Bits per bias: sign(1) + magnitude(4) + learnable(1) = 6
BITS_PER_BIAS = 6
NUM_BIASES = 2

# Total genotype length
GENOTYPE_LENGTH = NUM_CONNECTIONS * BITS_PER_CONNECTION + NUM_BIASES * BITS_PER_BIAS  # 40

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


# ============================================================================
# Genotype Creation
# ============================================================================


def create_random_genotype(key: PRNGKey) -> Array:
    """Generate a random binary genotype.

    Args:
        key: JAX random key

    Returns:
        Binary array of shape (GENOTYPE_LENGTH,)
    """
    return jax.random.bernoulli(key, p=0.5, shape=(GENOTYPE_LENGTH,)).astype(jnp.int32)


def create_random_population(key: PRNGKey, pop_size: int) -> Array:
    """Generate a population of random genotypes.

    Args:
        key: JAX random key
        pop_size: Number of individuals in population

    Returns:
        Binary array of shape (pop_size, GENOTYPE_LENGTH)
    """
    keys = jax.random.split(key, pop_size)
    return jax.vmap(create_random_genotype)(keys)


# ============================================================================
# Genotype Decoding
# ============================================================================


def bits_to_int(bits: Array) -> Array:
    """Convert binary array to integer.

    Args:
        bits: Binary array

    Returns:
        Integer value
    """
    powers = 2 ** jnp.arange(len(bits))
    return jnp.sum(bits * powers)


def decode_magnitude(magnitude_bits: Array, config: NetworkConfig) -> Array:
    """Decode magnitude bits to weight value.

    Args:
        magnitude_bits: 4 bits representing magnitude index
        config: Network configuration

    Returns:
        Weight magnitude value
    """
    magnitudes = config.get_weight_magnitudes()
    index = bits_to_int(magnitude_bits)
    # Clip index to valid range
    index = jnp.clip(index, 0, len(magnitudes) - 1)
    return magnitudes[index]


def decode_connection(conn_bits: Array, config: NetworkConfig) -> tuple[Array, Array]:
    """Decode connection bits to weight value and learnable flag.

    Args:
        conn_bits: 7 bits for one connection
        config: Network configuration

    Returns:
        Tuple of (weight_value, is_learnable)
    """
    present = conn_bits[CONN_BIT_PRESENT]
    sign = conn_bits[CONN_BIT_SIGN]
    mag_bits = conn_bits[CONN_BITS_MAG_START:CONN_BITS_MAG_END]
    learnable = conn_bits[CONN_BIT_LEARNABLE]

    # Decode magnitude
    magnitude = decode_magnitude(mag_bits, config)

    # Apply sign: 1 -> +1, 0 -> -1
    sign_multiplier = 2.0 * sign - 1.0

    # Apply present flag: if not present, weight is 0
    weight = present * sign_multiplier * magnitude

    return weight, learnable.astype(jnp.bool_)


def decode_bias(bias_bits: Array, config: NetworkConfig) -> tuple[Array, Array]:
    """Decode bias bits to bias value and learnable flag.

    Args:
        bias_bits: 6 bits for one bias
        config: Network configuration

    Returns:
        Tuple of (bias_value, is_learnable)
    """
    sign = bias_bits[BIAS_BIT_SIGN]
    mag_bits = bias_bits[BIAS_BITS_MAG_START:BIAS_BITS_MAG_END]
    learnable = bias_bits[BIAS_BIT_LEARNABLE]

    # Decode magnitude
    magnitude = decode_magnitude(mag_bits, config)

    # Apply sign
    sign_multiplier = 2.0 * sign - 1.0
    bias = sign_multiplier * magnitude

    return bias, learnable.astype(jnp.bool_)


def decode_genotype(genotype: Array, config: NetworkConfig) -> NetworkParams:
    """Convert binary genotype to network parameters.

    Args:
        genotype: Binary array of shape (GENOTYPE_LENGTH,)
        config: Network configuration

    Returns:
        NetworkParams with decoded weights, biases, and learnable mask
    """
    weights = []
    biases = []
    learnable_mask = []

    # Decode connections
    offset = 0
    for i in range(NUM_CONNECTIONS):
        conn_bits = genotype[offset : offset + BITS_PER_CONNECTION]
        weight, learnable = decode_connection(conn_bits, config)
        weights.append(weight)
        learnable_mask.append(learnable)
        offset += BITS_PER_CONNECTION

    # Decode biases
    for i in range(NUM_BIASES):
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
    """Decode entire population of genotypes to network parameters.

    Args:
        population: Binary array of shape (pop_size, GENOTYPE_LENGTH)
        config: Network configuration

    Returns:
        NetworkParams with batched arrays:
        - weights: shape (pop_size, 4)
        - biases: shape (pop_size, 2)
        - learnable_mask: shape (pop_size, 6)
    """
    return jax.vmap(lambda g: decode_genotype(g, config))(population)


# ============================================================================
# Genotype Encoding (for testing/analysis)
# ============================================================================


def int_to_bits(value: int, num_bits: int) -> Array:
    """Convert integer to binary array.

    Args:
        value: Integer value
        num_bits: Number of bits in output

    Returns:
        Binary array of shape (num_bits,)
    """
    return jnp.array([(value >> i) & 1 for i in range(num_bits)], dtype=jnp.int32)


def encode_connection(
    weight: float, learnable: bool, config: NetworkConfig
) -> Array:
    """Encode weight and learnable flag to connection bits.

    Args:
        weight: Weight value
        learnable: Whether connection is learnable
        config: Network configuration

    Returns:
        Binary array of shape (BITS_PER_CONNECTION,)
    """
    magnitudes = config.get_weight_magnitudes()

    # Determine if connection is present (non-zero weight)
    present = jnp.abs(weight) > 1e-8

    # Determine sign
    sign = weight >= 0

    # Find closest magnitude
    magnitude = jnp.abs(weight)
    mag_index = jnp.argmin(jnp.abs(magnitudes - magnitude))

    # Encode to bits
    bits = jnp.zeros(BITS_PER_CONNECTION, dtype=jnp.int32)
    bits = bits.at[CONN_BIT_PRESENT].set(present.astype(jnp.int32))
    bits = bits.at[CONN_BIT_SIGN].set(sign.astype(jnp.int32))
    bits = bits.at[CONN_BITS_MAG_START:CONN_BITS_MAG_END].set(int_to_bits(mag_index, 4))
    bits = bits.at[CONN_BIT_LEARNABLE].set(jnp.array(learnable, dtype=jnp.int32))

    return bits


def encode_bias(bias: float, learnable: bool, config: NetworkConfig) -> Array:
    """Encode bias and learnable flag to bias bits.

    Args:
        bias: Bias value
        learnable: Whether bias is learnable
        config: Network configuration

    Returns:
        Binary array of shape (BITS_PER_BIAS,)
    """
    magnitudes = config.get_weight_magnitudes()

    # Determine sign
    sign = bias >= 0

    # Find closest magnitude
    magnitude = jnp.abs(bias)
    mag_index = jnp.argmin(jnp.abs(magnitudes - magnitude))

    # Encode to bits
    bits = jnp.zeros(BITS_PER_BIAS, dtype=jnp.int32)
    bits = bits.at[BIAS_BIT_SIGN].set(sign.astype(jnp.int32))
    bits = bits.at[BIAS_BITS_MAG_START:BIAS_BITS_MAG_END].set(int_to_bits(mag_index, 4))
    bits = bits.at[BIAS_BIT_LEARNABLE].set(jnp.array(learnable, dtype=jnp.int32))

    return bits


def encode_genotype(params: NetworkParams, config: NetworkConfig) -> Array:
    """Convert network parameters back to genotype.

    Args:
        params: Network parameters
        config: Network configuration

    Returns:
        Binary array of shape (GENOTYPE_LENGTH,)
    """
    parts = []

    # Encode connections
    for i in range(NUM_CONNECTIONS):
        conn_bits = encode_connection(
            float(params.weights[i]), bool(params.learnable_mask[i]), config
        )
        parts.append(conn_bits)

    # Encode biases
    for i in range(NUM_BIASES):
        bias_bits = encode_bias(
            float(params.biases[i]), bool(params.learnable_mask[4 + i]), config
        )
        parts.append(bias_bits)

    return jnp.concatenate(parts)


# ============================================================================
# Genetic Operators
# ============================================================================


def two_point_crossover(
    key: PRNGKey, parent1: Array, parent2: Array
) -> tuple[Array, Array]:
    """Two-point crossover producing two offspring.

    Args:
        key: JAX random key
        parent1: First parent genotype
        parent2: Second parent genotype

    Returns:
        Tuple of two offspring genotypes
    """
    # Select two crossover points
    key1, key2 = jax.random.split(key)
    point1 = jax.random.randint(key1, (), 0, GENOTYPE_LENGTH)
    point2 = jax.random.randint(key2, (), 0, GENOTYPE_LENGTH)

    # Ensure point1 <= point2
    point1, point2 = jnp.minimum(point1, point2), jnp.maximum(point1, point2)

    # Create mask for middle section
    indices = jnp.arange(GENOTYPE_LENGTH)
    middle_mask = (indices >= point1) & (indices < point2)

    # Create offspring
    offspring1 = jnp.where(middle_mask, parent2, parent1)
    offspring2 = jnp.where(middle_mask, parent1, parent2)

    return offspring1, offspring2


def point_mutation(key: PRNGKey, genotype: Array, rate: float) -> Array:
    """Apply point mutation with given per-bit rate.

    Args:
        key: JAX random key
        genotype: Binary genotype
        rate: Probability of flipping each bit

    Returns:
        Mutated genotype
    """
    # Generate mutation mask
    flip_mask = jax.random.bernoulli(key, p=rate, shape=genotype.shape)

    # XOR to flip selected bits
    mutated = jnp.where(flip_mask, 1 - genotype, genotype)

    return mutated


def apply_linear_scaling(fitness_scores: Array, target_max_ratio: float) -> Array:
    """Apply linear fitness scaling.

    First shifts fitness to be non-negative, then scales so that
    max fitness = target_max_ratio * mean fitness. This ensures
    correct selection pressure regardless of the raw fitness sign.

    Args:
        fitness_scores: Raw fitness scores
        target_max_ratio: Target ratio of max to mean fitness

    Returns:
        Scaled fitness scores (all non-negative)
    """
    # First shift to non-negative so scaling math works correctly
    min_f = jnp.min(fitness_scores)
    shifted = fitness_scores - min_f + 1e-8  # All positive now

    mean_f = jnp.mean(shifted)
    max_f = jnp.max(shifted)

    # Target: max' = target_max_ratio * mean'
    target_max = target_max_ratio * mean_f

    # Linear scaling: f' = a * f + b
    # Constraints: mean' = mean_f, max' = target_max
    range_f = max_f - mean_f
    a = jnp.where(
        range_f > 1e-8,
        (target_max - mean_f) / range_f,
        1.0,
    )
    b = mean_f * (1.0 - a)

    scaled = a * shifted + b

    # Ensure non-negative (safety check)
    min_scaled = jnp.min(scaled)
    scaled = jnp.where(min_scaled < 0, scaled - min_scaled, scaled)

    return scaled


def select_parents_fitness_proportionate(
    key: PRNGKey, population: Array, fitness_scores: Array, num_parents: int
) -> Array:
    """Stochastic fitness-proportionate selection.

    Args:
        key: JAX random key
        population: Population of genotypes, shape (pop_size, genotype_length)
        fitness_scores: Fitness scores, shape (pop_size,)
        num_parents: Number of parents to select

    Returns:
        Selected parent genotypes, shape (num_parents, genotype_length)
    """
    # Normalize fitness to probabilities
    # Ensure all positive
    min_fitness = jnp.min(fitness_scores)
    shifted_fitness = fitness_scores - min_fitness + 1e-8

    probs = shifted_fitness / jnp.sum(shifted_fitness)

    # Sample indices with replacement
    indices = jax.random.choice(
        key, len(population), shape=(num_parents,), p=probs, replace=True
    )

    return population[indices]


def create_next_generation(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    config: GeneticConfig,
) -> Array:
    """Create the next generation through selection, crossover, and mutation.

    Args:
        key: JAX random key
        population: Current population, shape (pop_size, genotype_length)
        fitness_scores: Fitness scores, shape (pop_size,)
        config: Genetic algorithm configuration

    Returns:
        New population, shape (pop_size, genotype_length)
    """
    key_select, key_crossover, key_mutate = jax.random.split(key, 3)
    pop_size = len(population)

    # Apply linear fitness scaling if enabled
    if config.scaling_enabled:
        scaled_fitness = apply_linear_scaling(fitness_scores, config.scaling_target_max)
    else:
        scaled_fitness = fitness_scores

    # Select parents (we need pop_size parents for pop_size offspring)
    parents = select_parents_fitness_proportionate(
        key_select, population, scaled_fitness, pop_size
    )

    # Apply crossover
    # Pair up parents and apply crossover with probability crossover_rate
    crossover_keys = jax.random.split(key_crossover, pop_size // 2)
    do_crossover = jax.random.bernoulli(
        key_crossover, p=config.crossover_rate, shape=(pop_size // 2,)
    )

    def crossover_pair(key, do_cross, p1, p2):
        """Apply crossover to a pair if selected."""
        o1, o2 = two_point_crossover(key, p1, p2)
        # Return original parents if no crossover
        return jnp.where(do_cross, o1, p1), jnp.where(do_cross, o2, p2)

    # Reshape parents into pairs
    parents1 = parents[::2]
    parents2 = parents[1::2]

    offspring1, offspring2 = jax.vmap(crossover_pair)(
        crossover_keys, do_crossover, parents1, parents2
    )

    # Interleave offspring back together
    offspring = jnp.empty_like(population)
    offspring = offspring.at[::2].set(offspring1)
    offspring = offspring.at[1::2].set(offspring2)

    # Apply mutation
    mutation_keys = jax.random.split(key_mutate, pop_size)
    offspring = jax.vmap(lambda k, g: point_mutation(k, g, config.mutation_rate))(
        mutation_keys, offspring
    )

    return offspring


# ============================================================================
# Elitism (optional extension)
# ============================================================================


def create_next_generation_with_elitism(
    key: PRNGKey,
    population: Array,
    fitness_scores: Array,
    config: GeneticConfig,
    elite_count: int = 1,
) -> Array:
    """Create next generation with elitism (preserving best individuals).

    Args:
        key: JAX random key
        population: Current population
        fitness_scores: Fitness scores
        config: Genetic algorithm configuration
        elite_count: Number of elite individuals to preserve

    Returns:
        New population with elite individuals preserved
    """
    # Get indices of elite individuals
    elite_indices = jnp.argsort(fitness_scores)[-elite_count:]
    elites = population[elite_indices]

    # Create rest of population through normal GA
    key_ga = key
    rest_size = len(population) - elite_count

    # Temporarily adjust for generating fewer offspring
    new_offspring = create_next_generation(key_ga, population, fitness_scores, config)

    # Replace last elite_count with elites
    new_population = new_offspring.at[-elite_count:].set(elites)

    return new_population

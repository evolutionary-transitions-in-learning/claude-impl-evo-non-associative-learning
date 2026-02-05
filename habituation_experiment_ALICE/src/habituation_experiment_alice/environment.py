"""Environment generation for the ALICE threat discrimination simulation.

This module handles:
- Generating clumpy threat sequences (threat present/absent)
- Assigning true/false threat types (with clump coherence)
- Generating stimulus signals (identical for true/false threats)
- Generating delayed pain signals (from true threats only)
- Phase 1 (all true threats) and Phase 2 (mixed) environments
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from .config import SimulationConfig


class PhaseEnvironment(NamedTuple):
    """Environment data for one phase of evaluation.

    Attributes:
        threat_present: Whether a threat stimulus is present. Shape: (lifetime,)
        true_threat: Whether the threat is a TRUE threat. Shape: (lifetime,)
        stimulus_signal: What the organism perceives. Shape: (lifetime, num_channels)
        pain_signal: Delayed pain from true threats. Shape: (lifetime,)
    """

    threat_present: Array
    true_threat: Array
    stimulus_signal: Array
    pain_signal: Array


# ============================================================================
# Clumpy Sequence Generation
# ============================================================================


def generate_clumpy_threat_sequence(
    key: PRNGKey, lifetime: int, clump_scale: int
) -> Array:
    """Generate a clumpy threat-present/absent binary sequence.

    1. Create base random sequence of length ceil(lifetime / clump_scale)
    2. Expand each element to clump_scale consecutive copies
    3. Truncate to exactly lifetime

    Minimum clump length = clump_scale, mean ~ 2 * clump_scale.

    Returns:
        Binary array shape (lifetime,), 1=threat present, 0=no threat
    """
    base_length = (lifetime + clump_scale - 1) // clump_scale
    base_seq = jax.random.bernoulli(key, p=0.5, shape=(base_length,)).astype(jnp.float32)
    expanded = jnp.repeat(base_seq, clump_scale)
    return expanded[:lifetime]


# ============================================================================
# Threat Type Assignment
# ============================================================================


def assign_threat_types(
    key: PRNGKey,
    threat_present: Array,
    true_threat_ratio: float,
    clump_scale: int,
    lifetime: int,
) -> Array:
    """Assign true/false labels to threat clumps.

    Each clump is independently labeled as true or false threat to maintain
    clump coherence (all timesteps in a clump have the same type).

    Args:
        key: JAX random key
        threat_present: Binary threat-present sequence, shape (lifetime,)
        true_threat_ratio: Probability that a threat clump is a true threat
        clump_scale: Clump scale (same as used for sequence generation)
        lifetime: Total lifetime length

    Returns:
        Binary array shape (lifetime,), 1 where true threat AND present, 0 otherwise.
    """
    base_length = (lifetime + clump_scale - 1) // clump_scale
    base_is_true = jax.random.bernoulli(
        key, p=true_threat_ratio, shape=(base_length,)
    ).astype(jnp.float32)
    is_true_expanded = jnp.repeat(base_is_true, clump_scale)[:lifetime]
    return threat_present * is_true_expanded


# ============================================================================
# Signal Generation
# ============================================================================


def generate_pain_signal(
    true_threat: Array,
    pain_delay: int,
    pain_magnitude: float,
) -> Array:
    """Generate pain signal as delayed version of true_threat indicator.

    pain[t] = pain_magnitude * true_threat[t - delay]
    For t < delay: pain[t] = 0

    Args:
        true_threat: Binary true-threat sequence, shape (lifetime,)
        pain_delay: Number of timesteps to delay pain
        pain_magnitude: Pain signal intensity

    Returns:
        Pain signal array, shape (lifetime,)
    """
    # Shift true_threat forward by pain_delay timesteps
    padded = jnp.concatenate([jnp.zeros(pain_delay), true_threat[:-pain_delay]])
    return padded * pain_magnitude


def generate_stimulus_signal(
    threat_present: Array,
    stimulus_magnitude: float,
    num_channels: int,
) -> Array:
    """Generate stimulus signal.

    Stimulus is present whenever any threat is present, regardless of
    whether the threat is true or false (identical sensory profile).

    Args:
        threat_present: Binary threat-present sequence, shape (lifetime,)
        stimulus_magnitude: Signal magnitude when threat present
        num_channels: Number of stimulus channels

    Returns:
        Stimulus signal, shape (lifetime, num_channels)
    """
    # For N=1 channel, broadcast to (lifetime, 1)
    return threat_present[:, None] * stimulus_magnitude * jnp.ones(num_channels)


# ============================================================================
# Phase Environment Generation
# ============================================================================


def generate_phase_environment(
    key: PRNGKey,
    lifetime: int,
    clump_scale: int,
    true_threat_ratio: float,
    pain_delay: int,
    pain_magnitude: float,
    stimulus_magnitude: float,
    num_channels: int,
) -> PhaseEnvironment:
    """Generate complete environment for one evaluation phase."""
    key_threat, key_assign = jax.random.split(key)

    threat_present = generate_clumpy_threat_sequence(key_threat, lifetime, clump_scale)
    true_threat = assign_threat_types(
        key_assign, threat_present, true_threat_ratio, clump_scale, lifetime
    )
    stimulus_signal = generate_stimulus_signal(
        threat_present, stimulus_magnitude, num_channels
    )
    pain_signal = generate_pain_signal(true_threat, pain_delay, pain_magnitude)

    return PhaseEnvironment(
        threat_present=threat_present,
        true_threat=true_threat,
        stimulus_signal=stimulus_signal,
        pain_signal=pain_signal,
    )


def generate_phase1_environment(key: PRNGKey, config: SimulationConfig) -> PhaseEnvironment:
    """Generate Phase 1 environment: all threats are true, with gaps for eating."""
    return generate_phase_environment(
        key,
        lifetime=config.environment.phase1_lifetime,
        clump_scale=config.environment.clump_scale,
        true_threat_ratio=1.0,  # ALL threats are true in phase 1
        pain_delay=config.pain.delay,
        pain_magnitude=config.pain.magnitude,
        stimulus_magnitude=config.environment.stimulus_magnitude,
        num_channels=config.environment.num_stimulus_channels,
    )


def generate_phase2_environment(key: PRNGKey, config: SimulationConfig) -> PhaseEnvironment:
    """Generate Phase 2 environment: mix of true and false threats."""
    return generate_phase_environment(
        key,
        lifetime=config.environment.phase2_lifetime,
        clump_scale=config.environment.clump_scale,
        true_threat_ratio=config.environment.true_false_ratio,
        pain_delay=config.pain.delay,
        pain_magnitude=config.pain.magnitude,
        stimulus_magnitude=config.environment.stimulus_magnitude,
        num_channels=config.environment.num_stimulus_channels,
    )

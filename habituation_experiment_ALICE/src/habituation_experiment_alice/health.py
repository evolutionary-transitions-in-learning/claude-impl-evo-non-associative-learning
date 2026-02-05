"""Health dynamics for the ALICE threat discrimination simulation.

Health model:
- Passive decay: health decreases each timestep
- Eating: when output < 0, health increases proportional to |output|
- Threat damage: when true threat present, damage proportional to (1 - protection)
  where protection = max(0, output)
- Death: health <= 0, organism dies and stays dead

Output interpretation:
  +1 = full withdrawal (max protection=1.0, no eating)
  -1 = full eating (eating_amount=1.0, no protection)
   0 = neutral (no eating, no protection)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .config import HealthConfig


class HealthResult(NamedTuple):
    """Result of health simulation over a lifetime.

    Attributes:
        health_trajectory: Health at each timestep, shape (lifetime,)
        alive_mask: 1 where alive, 0 after death, shape (lifetime,)
        final_health: Scalar, health at end (0 if dead)
    """

    health_trajectory: Array
    alive_mask: Array
    final_health: Array


def compute_health_delta(
    output: Array,
    true_threat_present: Array,
    config: HealthConfig,
) -> Array:
    """Compute health change for a single timestep.

    Args:
        output: Motor output in [-1, +1]
        true_threat_present: 0 or 1
        config: Health configuration

    Returns:
        Scalar health delta
    """
    # Passive decay (always)
    delta = -config.passive_decay

    # Eating: when output < 0, eating amount = |output|
    eating_amount = jnp.maximum(-output, 0.0)
    delta = delta + eating_amount * config.eating_gain_rate

    # Damage from true threats: protection = max(0, output)
    protection = jnp.maximum(output, 0.0)
    damage = (1.0 - protection) * config.threat_damage
    delta = delta - true_threat_present * damage

    return delta


def simulate_health(
    outputs: Array,
    true_threats: Array,
    config: HealthConfig,
) -> HealthResult:
    """Simulate health over a lifetime.

    Uses jax.lax.scan for efficiency. Once health <= 0, the organism
    is dead and no further health changes occur.

    Args:
        outputs: Motor outputs, shape (lifetime,)
        true_threats: True threat indicators, shape (lifetime,)
        config: Health configuration

    Returns:
        HealthResult with trajectory, alive mask, and final health
    """

    def step_fn(health, t_data):
        output, true_threat = t_data
        alive = health > 0.0

        delta = compute_health_delta(output, true_threat, config)
        # Only apply delta if alive
        new_health = jnp.where(alive, health + delta, 0.0)
        # Clamp to non-negative
        new_health = jnp.maximum(new_health, 0.0)

        return new_health, (new_health, alive.astype(jnp.float32))

    initial_health = config.starting_health
    final_health, (health_trajectory, alive_mask) = jax.lax.scan(
        step_fn, initial_health, (outputs, true_threats)
    )

    return HealthResult(
        health_trajectory=health_trajectory,
        alive_mask=alive_mask,
        final_health=final_health,
    )

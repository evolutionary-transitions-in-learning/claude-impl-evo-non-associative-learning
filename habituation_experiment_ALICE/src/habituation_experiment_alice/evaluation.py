"""Two-phase evaluation pipeline for the ALICE threat discrimination simulation.

Phase 1 (Survival Test):
- All threats are true positives
- Agents that die (health <= 0) get fitness based on survival time

Phase 2 (Discrimination Test):
- Mix of true and false threats
- Network parameters decoded fresh (same genotype)
- Fitness = 1.0 + final health at end of phase 2

Both phases are always computed (for JIT shape consistency),
with phase 2 result masked if the agent died in phase 1.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from .config import SimulationConfig
from .environment import (
    PhaseEnvironment,
    generate_phase1_environment,
    generate_phase2_environment,
)
from .genetics import decode_genotype
from .health import HealthResult, simulate_health
from .network import NetworkParams, run_network_phase


class PhaseResult(NamedTuple):
    """Result from evaluating one phase."""

    outputs: Array
    health_result: HealthResult
    survived: Array  # boolean scalar


class EvaluationResult(NamedTuple):
    """Result from full two-phase evaluation."""

    phase1_result: PhaseResult
    phase2_result: PhaseResult
    fitness: Array  # scalar
    survived_phase1: Array  # boolean scalar


class PopulationEvalSummary(NamedTuple):
    """Summary statistics from evaluating an entire population."""

    fitness: Array              # (pop_size,)
    survived_phase1: Array      # (pop_size,) bool
    phase1_survival_time: Array  # (pop_size,) float
    phase2_final_health: Array  # (pop_size,) float


class AgentTrace(NamedTuple):
    """Full behavioral trace for a single agent evaluation."""

    # Phase 1 signals
    phase1_outputs: Array
    phase1_health: Array
    phase1_alive: Array
    phase1_stimulus: Array
    phase1_pain: Array
    phase1_true_threat: Array
    phase1_threat_present: Array
    # Phase 2 signals
    phase2_outputs: Array
    phase2_health: Array
    phase2_alive: Array
    phase2_stimulus: Array
    phase2_pain: Array
    phase2_true_threat: Array
    phase2_threat_present: Array
    # Network weights
    weights_initial: Array
    biases_initial: Array
    weights_after_phase1: Array
    biases_after_phase1: Array
    weights_after_phase2: Array
    biases_after_phase2: Array
    learnable_mask: Array
    # Summary
    fitness: Array
    survived_phase1: Array


def evaluate_phase(
    params: NetworkParams,
    environment: PhaseEnvironment,
    config: SimulationConfig,
) -> PhaseResult:
    """Evaluate one phase: run network, simulate health.

    Args:
        params: Network parameters (fresh from genotype decode)
        environment: Phase environment data
        config: Simulation configuration

    Returns:
        PhaseResult with outputs, health result, and survival flag
    """
    # For N=1, stimulus_signal is shape (lifetime, 1); extract channel 0
    stimulus_inputs = environment.stimulus_signal[:, 0]
    pain_inputs = environment.pain_signal

    # Run network over phase
    outputs, final_params = run_network_phase(
        params,
        stimulus_inputs,
        pain_inputs,
        config.network,
        config.learning,
    )

    # Simulate health
    health_result = simulate_health(
        outputs, environment.true_threat, config.health
    )

    survived = health_result.final_health > 0.0

    return PhaseResult(
        outputs=outputs,
        health_result=health_result,
        survived=survived,
    )


def evaluate_creature(
    genotype: Array,
    phase1_env: PhaseEnvironment,
    phase2_env: PhaseEnvironment,
    config: SimulationConfig,
) -> EvaluationResult:
    """Full two-phase evaluation of a single creature.

    Phase 1: Survival test with all true threats.
    Phase 2: Discrimination test with mixed threats.
    Network parameters reset between phases (both use fresh genotype decode).

    Args:
        genotype: Binary genotype array
        phase1_env: Phase 1 environment
        phase2_env: Phase 2 environment
        config: Simulation configuration

    Returns:
        EvaluationResult with fitness and phase details
    """
    # Decode genotype to network parameters
    params = decode_genotype(genotype, config.network)

    # Phase 1: survival test
    phase1 = evaluate_phase(params, phase1_env, config)

    # Phase 2: discrimination test (fresh params from same genotype)
    phase2 = evaluate_phase(params, phase2_env, config)

    # Fitness structure (3 tiers to ensure monotonic selection pressure):
    #
    # Tier 1 (died in phase 1): 0 < fitness < 1.0
    #   = survival_fraction * 0.99 (gradient toward surviving longer)
    #
    # Tier 2 (survived phase 1, died in phase 2): fitness = 1.0
    #   = guaranteed reward for passing the survival test
    #
    # Tier 3 (survived both): fitness > 1.0
    #   = 1.0 + phase2_final_health (main optimization target)
    #
    # This ensures: die_early < die_late < survive_p1 < survive_p1+good_p2

    phase1_lifetime = phase1.outputs.shape[0]
    survival_time = jnp.sum(phase1.health_result.alive_mask)
    survival_fraction = survival_time / phase1_lifetime

    phase1_bootstrap_fitness = survival_fraction * 0.99  # tier 1: [0, 0.99)
    phase2_fitness = 1.0 + phase2.health_result.final_health  # tier 2+3: >= 1.0

    fitness = jnp.where(
        phase1.survived,
        phase2_fitness,
        phase1_bootstrap_fitness,
    )

    return EvaluationResult(
        phase1_result=phase1,
        phase2_result=phase2,
        fitness=fitness,
        survived_phase1=phase1.survived,
    )


def _extract_population_summary(results: EvaluationResult) -> PopulationEvalSummary:
    """Extract PopulationEvalSummary from vmapped EvaluationResult."""
    phase1_survival_time = jnp.sum(
        results.phase1_result.health_result.alive_mask, axis=1
    )
    return PopulationEvalSummary(
        fitness=results.fitness,
        survived_phase1=results.survived_phase1,
        phase1_survival_time=phase1_survival_time,
        phase2_final_health=results.phase2_result.health_result.final_health,
    )


def evaluate_population(
    key: PRNGKey,
    population: Array,
    config: SimulationConfig,
) -> PopulationEvalSummary:
    """Evaluate entire population.

    When config.environment.shared_environment is True (default), all
    creatures face the same threat patterns for fair comparison. When
    False, each creature gets independently randomized threats.

    Args:
        key: JAX random key
        population: Population genotypes, shape (pop_size, genotype_length)
        config: Simulation configuration

    Returns:
        PopulationEvalSummary with fitness and phase-specific stats
    """
    if config.environment.shared_environment:
        key_p1, key_p2 = jax.random.split(key)
        phase1_env = generate_phase1_environment(key_p1, config)
        phase2_env = generate_phase2_environment(key_p2, config)

        eval_fn = lambda genotype: evaluate_creature(
            genotype, phase1_env, phase2_env, config
        )
        results = jax.vmap(eval_fn)(population)
    else:
        pop_size = population.shape[0]
        keys = jax.random.split(key, pop_size)

        def eval_one(genotype, creature_key):
            k1, k2 = jax.random.split(creature_key)
            p1_env = generate_phase1_environment(k1, config)
            p2_env = generate_phase2_environment(k2, config)
            return evaluate_creature(genotype, p1_env, p2_env, config)

        results = jax.vmap(eval_one)(population, keys)

    return _extract_population_summary(results)


def evaluate_agent_detailed(
    genotype: Array,
    key: PRNGKey,
    config: SimulationConfig,
) -> AgentTrace:
    """Evaluate a single agent and return full behavioral traces.

    Used for checkpoint trace generation. Generates fresh environments
    using the provided key and captures all signals for visualization.

    Args:
        genotype: Binary genotype array
        key: JAX random key for environment generation
        config: Simulation configuration

    Returns:
        AgentTrace with complete behavioral data for both phases
    """
    key_p1, key_p2 = jax.random.split(key)

    phase1_env = generate_phase1_environment(key_p1, config)
    phase2_env = generate_phase2_environment(key_p2, config)

    params = decode_genotype(genotype, config.network)

    # Phase 1
    s1 = phase1_env.stimulus_signal[:, 0]
    p1 = phase1_env.pain_signal
    outputs1, params_after_p1 = run_network_phase(
        params, s1, p1, config.network, config.learning
    )
    health1 = simulate_health(outputs1, phase1_env.true_threat, config.health)

    # Phase 2 (same initial params)
    s2 = phase2_env.stimulus_signal[:, 0]
    p2 = phase2_env.pain_signal
    outputs2, params_after_p2 = run_network_phase(
        params, s2, p2, config.network, config.learning
    )
    health2 = simulate_health(outputs2, phase2_env.true_threat, config.health)

    # Fitness (same logic as evaluate_creature)
    phase1_lifetime = outputs1.shape[0]
    survival_time = jnp.sum(health1.alive_mask)
    survival_fraction = survival_time / phase1_lifetime
    survived_p1 = health1.final_health > 0.0

    phase1_bootstrap_fitness = survival_fraction * 0.99
    phase2_fitness = 1.0 + health2.final_health
    fitness = jnp.where(survived_p1, phase2_fitness, phase1_bootstrap_fitness)

    return AgentTrace(
        phase1_outputs=outputs1,
        phase1_health=health1.health_trajectory,
        phase1_alive=health1.alive_mask,
        phase1_stimulus=s1,
        phase1_pain=p1,
        phase1_true_threat=phase1_env.true_threat,
        phase1_threat_present=phase1_env.threat_present,
        phase2_outputs=outputs2,
        phase2_health=health2.health_trajectory,
        phase2_alive=health2.alive_mask,
        phase2_stimulus=s2,
        phase2_pain=p2,
        phase2_true_threat=phase2_env.true_threat,
        phase2_threat_present=phase2_env.threat_present,
        weights_initial=params.weights,
        biases_initial=params.biases,
        weights_after_phase1=params_after_p1.weights,
        biases_after_phase1=params_after_p1.biases,
        weights_after_phase2=params_after_p2.weights,
        biases_after_phase2=params_after_p2.biases,
        learnable_mask=params.learnable_mask,
        fitness=fitness,
        survived_phase1=survived_p1,
    )

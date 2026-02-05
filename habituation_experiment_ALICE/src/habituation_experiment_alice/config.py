"""Configuration dataclasses for the ALICE threat discrimination simulation.

This module defines all configuration parameters for:
- Environment (threat sequences, lifetimes, clumping)
- Pain channel (delay, magnitude)
- Health dynamics (decay, eating, damage)
- Neural network architecture
- Hebbian learning
- Genetic algorithm (tournament selection)
- Top-level simulation orchestration
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class WeightSpacing(str, Enum):
    """Weight magnitude spacing method."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


class HebbianRule(str, Enum):
    """Hebbian learning rule variant."""

    BASIC = "basic"  # dw = eta * pre * post
    OJA = "oja"  # dw = eta * post * (pre - post * w)
    NORMALIZED = "normalized"  # dw = eta * pre * post / ||w||


@dataclass
class EnvironmentConfig:
    """Configuration for the simulated threat environment."""

    clump_scale: int = 10
    phase1_lifetime: int = 500
    phase2_lifetime: int = 1000
    true_false_ratio: float = 0.5  # fraction of threats that are TRUE in phase 2
    stimulus_magnitude: float = 1.0
    num_stimulus_channels: int = 1
    shared_environment: bool = True  # all creatures face same threats per generation

    def __post_init__(self):
        if self.clump_scale < 1:
            raise ValueError(f"clump_scale must be >= 1, got {self.clump_scale}")
        if not 0.0 <= self.true_false_ratio <= 1.0:
            raise ValueError(
                f"true_false_ratio must be in [0, 1], got {self.true_false_ratio}"
            )
        if self.phase1_lifetime < 1:
            raise ValueError(f"phase1_lifetime must be >= 1, got {self.phase1_lifetime}")
        if self.phase2_lifetime < 1:
            raise ValueError(f"phase2_lifetime must be >= 1, got {self.phase2_lifetime}")
        if self.num_stimulus_channels < 1:
            raise ValueError(
                f"num_stimulus_channels must be >= 1, got {self.num_stimulus_channels}"
            )


@dataclass
class PainConfig:
    """Configuration for the pain channel."""

    delay: int = 1  # timesteps before pain signal appears
    magnitude: float = 1.0  # pain signal intensity

    def __post_init__(self):
        if self.delay < 1:
            raise ValueError(f"pain delay must be >= 1, got {self.delay}")
        if self.magnitude < 0:
            raise ValueError(f"pain magnitude must be >= 0, got {self.magnitude}")


@dataclass
class HealthConfig:
    """Configuration for health dynamics."""

    starting_health: float = 20.0
    passive_decay: float = 0.1  # health lost per timestep
    eating_gain_rate: float = 1.0  # health gained per timestep at full eating (output=-1)
    threat_damage: float = 5.0  # damage per timestep from unprotected true threat

    def __post_init__(self):
        if self.starting_health <= 0:
            raise ValueError(f"starting_health must be > 0, got {self.starting_health}")
        if self.passive_decay < 0:
            raise ValueError(f"passive_decay must be >= 0, got {self.passive_decay}")
        if self.eating_gain_rate < 0:
            raise ValueError(f"eating_gain_rate must be >= 0, got {self.eating_gain_rate}")
        if self.threat_damage < 0:
            raise ValueError(f"threat_damage must be >= 0, got {self.threat_damage}")


@dataclass
class NetworkConfig:
    """Configuration for the neural network architecture."""

    num_stimulus_channels: int = 1
    num_weight_magnitudes: int = 16
    max_weight: float = 4.1
    weight_spacing: WeightSpacing = WeightSpacing.LINEAR

    def __post_init__(self):
        if isinstance(self.weight_spacing, str):
            self.weight_spacing = WeightSpacing(self.weight_spacing)

    def get_weight_magnitudes(self):
        """Return array of possible weight magnitudes."""
        import jax.numpy as jnp

        n = self.num_weight_magnitudes
        if self.weight_spacing == WeightSpacing.LINEAR:
            return jnp.linspace(0.0, self.max_weight, n)
        else:  # LOGARITHMIC
            magnitudes = jnp.logspace(-2, jnp.log10(self.max_weight), n - 1)
            return jnp.concatenate([jnp.array([0.0]), magnitudes])

    @property
    def num_neurons(self) -> int:
        """Total neurons: N stimulus + 1 pain + 1 output."""
        return self.num_stimulus_channels + 2

    @property
    def num_connections(self) -> int:
        """Full connectivity: n^2 directed connections."""
        n = self.num_neurons
        return n * n

    @property
    def num_biases(self) -> int:
        """One bias per neuron."""
        return self.num_neurons

    @property
    def num_params(self) -> int:
        """Total parameters (connections + biases)."""
        return self.num_connections + self.num_biases


@dataclass
class LearningConfig:
    """Configuration for Hebbian learning."""

    enabled: bool = True
    rule: HebbianRule = HebbianRule.BASIC
    learning_rate: float = 0.01
    weight_clip: float = 4.1

    def __post_init__(self):
        if isinstance(self.rule, str):
            self.rule = HebbianRule(self.rule)


@dataclass
class GeneticConfig:
    """Configuration for the genetic algorithm."""

    population_size: int = 100
    max_generations: int = 2000
    mutation_rate: float = 0.01  # per bit
    crossover_rate: float = 0.7

    def __post_init__(self):
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {self.mutation_rate}")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {self.crossover_rate}")
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")


@dataclass
class SimulationConfig:
    """Top-level configuration combining all sub-configs."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    pain: PainConfig = field(default_factory=PainConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    seed: int = 42
    num_runs: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Create configuration from a dictionary."""
        return cls(
            environment=EnvironmentConfig(**data.get("environment", {})),
            network=NetworkConfig(**data.get("network", {})),
            pain=PainConfig(**data.get("pain", {})),
            health=HealthConfig(**data.get("health", {})),
            learning=LearningConfig(**data.get("learning", {})),
            genetic=GeneticConfig(**data.get("genetic", {})),
            seed=data.get("simulation", {}).get("seed", 42),
            num_runs=data.get("simulation", {}).get("num_runs", 5),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            "environment": {
                "clump_scale": self.environment.clump_scale,
                "phase1_lifetime": self.environment.phase1_lifetime,
                "phase2_lifetime": self.environment.phase2_lifetime,
                "true_false_ratio": self.environment.true_false_ratio,
                "stimulus_magnitude": self.environment.stimulus_magnitude,
                "num_stimulus_channels": self.environment.num_stimulus_channels,
                "shared_environment": self.environment.shared_environment,
            },
            "network": {
                "num_stimulus_channels": self.network.num_stimulus_channels,
                "num_weight_magnitudes": self.network.num_weight_magnitudes,
                "max_weight": self.network.max_weight,
                "weight_spacing": self.network.weight_spacing.value,
            },
            "pain": {
                "delay": self.pain.delay,
                "magnitude": self.pain.magnitude,
            },
            "health": {
                "starting_health": self.health.starting_health,
                "passive_decay": self.health.passive_decay,
                "eating_gain_rate": self.health.eating_gain_rate,
                "threat_damage": self.health.threat_damage,
            },
            "learning": {
                "enabled": self.learning.enabled,
                "rule": self.learning.rule.value,
                "learning_rate": self.learning.learning_rate,
                "weight_clip": self.learning.weight_clip,
            },
            "genetic": {
                "population_size": self.genetic.population_size,
                "max_generations": self.genetic.max_generations,
                "mutation_rate": self.genetic.mutation_rate,
                "crossover_rate": self.genetic.crossover_rate,
            },
            "simulation": {
                "seed": self.seed,
                "num_runs": self.num_runs,
            },
        }

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def with_updates(self, **kwargs) -> "SimulationConfig":
        """Create a new config with updated values.

        Supports dotted notation: with_updates(**{"environment.clump_scale": 20})
        """
        import copy

        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            elif "." in key:
                parts = key.split(".")
                obj = new_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        return new_config

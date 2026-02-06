"""Configuration dataclasses for the habituation/sensitization evolution simulation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml


class NoiseMode(str, Enum):
    """Sensory noise model selection."""

    DISCRETE = "discrete"  # Binary flip with probability (1 - accuracy)
    CONTINUOUS = "continuous"  # Gaussian noise added to continuous signal


class WeightSpacing(str, Enum):
    """Weight magnitude spacing method."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


class MotorActivation(str, Enum):
    """Motor unit activation function."""

    TANH = "tanh"
    SCALED_SIGMOID = "scaled_sigmoid"


class HebbianRule(str, Enum):
    """Hebbian learning rule variant."""

    BASIC = "basic"  # Δw = η * pre * post
    OJA = "oja"  # Δw = η * post * (pre - post * w)
    NORMALIZED = "normalized"  # Δw = η * pre * post / ||w||


@dataclass
class EnvironmentConfig:
    """Configuration for the simulated environment."""

    clump_scale: int = 10
    sensory_accuracy: float = 0.75
    lifespan: int = 1000
    noise_mode: NoiseMode = NoiseMode.DISCRETE
    # For continuous noise mode: standard deviation of Gaussian noise
    noise_std: float = 0.5

    def __post_init__(self):
        if isinstance(self.noise_mode, str):
            self.noise_mode = NoiseMode(self.noise_mode)
        if not 0.5 <= self.sensory_accuracy <= 1.0:
            raise ValueError(f"sensory_accuracy must be in [0.5, 1.0], got {self.sensory_accuracy}")
        if self.clump_scale < 1:
            raise ValueError(f"clump_scale must be >= 1, got {self.clump_scale}")
        if self.lifespan < 1:
            raise ValueError(f"lifespan must be >= 1, got {self.lifespan}")


@dataclass
class NetworkConfig:
    """Configuration for the neural network architecture."""

    # Weight configuration
    num_weight_magnitudes: int = 16
    max_weight: float = 4.1
    weight_spacing: WeightSpacing = WeightSpacing.LINEAR

    # Input encoding (for discrete mode)
    sweet_input: float = 1.0  # food smell
    sour_input: float = -1.0  # poison smell

    # Activation function
    motor_activation: MotorActivation = MotorActivation.TANH

    def __post_init__(self):
        if isinstance(self.weight_spacing, str):
            self.weight_spacing = WeightSpacing(self.weight_spacing)
        if isinstance(self.motor_activation, str):
            self.motor_activation = MotorActivation(self.motor_activation)

    def get_weight_magnitudes(self):
        """Return array of possible weight magnitudes."""
        import jax.numpy as jnp

        n = self.num_weight_magnitudes
        if self.weight_spacing == WeightSpacing.LINEAR:
            return jnp.linspace(0.0, self.max_weight, n)
        else:  # LOGARITHMIC
            # Use log spacing, but include 0 explicitly
            magnitudes = jnp.logspace(-2, jnp.log10(self.max_weight), n - 1)
            return jnp.concatenate([jnp.array([0.0]), magnitudes])


@dataclass
class LearningConfig:
    """Configuration for Hebbian learning."""

    enabled: bool = True
    rule: HebbianRule = HebbianRule.BASIC
    learning_rate: float = 0.01
    weight_clip: float = 4.1  # Clip weights to [-clip, clip] to prevent explosion

    def __post_init__(self):
        if isinstance(self.rule, str):
            self.rule = HebbianRule(self.rule)


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""

    eat_food_reward: float = 1.0
    eat_poison_penalty: float = -1.0
    no_eat_cost: float = 0.0
    # Threshold for considering cluster-tracking evolved
    success_threshold_ratio: float = 0.7


@dataclass
class GeneticConfig:
    """Configuration for the genetic algorithm."""

    population_size: int = 100
    max_generations: int = 2000
    mutation_rate: float = 0.01  # per bit
    crossover_rate: float = 0.7
    # Linear fitness scaling
    scaling_enabled: bool = True
    scaling_target_max: float = 1.5  # target max as multiple of mean

    def __post_init__(self):
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {self.mutation_rate}")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {self.crossover_rate}")


@dataclass
class ExperimentConfig:
    """Configuration for running the full experimental grid."""

    clump_scales: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 40, 80])
    sensory_accuracies: list[float] = field(
        default_factory=lambda: [0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
    )
    runs_per_condition: int = 5


@dataclass
class SimulationConfig:
    """Top-level configuration combining all sub-configs."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    seed: int = 42
    num_runs: int = 5
    use_jit: bool = True

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
            learning=LearningConfig(**data.get("learning", {})),
            fitness=FitnessConfig(**data.get("fitness", {})),
            genetic=GeneticConfig(**data.get("genetic", {})),
            experiment=ExperimentConfig(**data.get("experiment", {})),
            seed=data.get("simulation", {}).get("seed", 42),
            num_runs=data.get("simulation", {}).get("num_runs", 5),
            use_jit=data.get("simulation", {}).get("use_jit", True),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            "environment": {
                "clump_scale": self.environment.clump_scale,
                "sensory_accuracy": self.environment.sensory_accuracy,
                "lifespan": self.environment.lifespan,
                "noise_mode": self.environment.noise_mode.value,
                "noise_std": self.environment.noise_std,
            },
            "network": {
                "num_weight_magnitudes": self.network.num_weight_magnitudes,
                "max_weight": self.network.max_weight,
                "weight_spacing": self.network.weight_spacing.value,
                "sweet_input": self.network.sweet_input,
                "sour_input": self.network.sour_input,
                "motor_activation": self.network.motor_activation.value,
            },
            "learning": {
                "enabled": self.learning.enabled,
                "rule": self.learning.rule.value,
                "learning_rate": self.learning.learning_rate,
            },
            "fitness": {
                "eat_food_reward": self.fitness.eat_food_reward,
                "eat_poison_penalty": self.fitness.eat_poison_penalty,
                "no_eat_cost": self.fitness.no_eat_cost,
                "success_threshold_ratio": self.fitness.success_threshold_ratio,
            },
            "genetic": {
                "population_size": self.genetic.population_size,
                "max_generations": self.genetic.max_generations,
                "mutation_rate": self.genetic.mutation_rate,
                "crossover_rate": self.genetic.crossover_rate,
                "scaling_enabled": self.genetic.scaling_enabled,
                "scaling_target_max": self.genetic.scaling_target_max,
            },
            "experiment": {
                "clump_scales": self.experiment.clump_scales,
                "sensory_accuracies": self.experiment.sensory_accuracies,
                "runs_per_condition": self.experiment.runs_per_condition,
            },
            "simulation": {
                "seed": self.seed,
                "num_runs": self.num_runs,
                "use_jit": self.use_jit,
            },
        }

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def with_updates(self, **kwargs) -> "SimulationConfig":
        """Create a new config with updated values."""
        import copy

        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            elif "." in key:
                # Handle nested updates like "environment.clump_scale"
                parts = key.split(".")
                obj = new_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        return new_config

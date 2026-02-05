"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from habituation_experiment_alice.config import (
    EnvironmentConfig,
    GeneticConfig,
    HealthConfig,
    LearningConfig,
    NetworkConfig,
    PainConfig,
    SimulationConfig,
)


class TestEnvironmentConfig:
    def test_defaults(self):
        cfg = EnvironmentConfig()
        assert cfg.clump_scale == 10
        assert cfg.phase1_lifetime == 500
        assert cfg.phase2_lifetime == 1000
        assert cfg.true_false_ratio == 0.5

    def test_invalid_clump_scale(self):
        with pytest.raises(ValueError):
            EnvironmentConfig(clump_scale=0)

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            EnvironmentConfig(true_false_ratio=1.5)


class TestHealthConfig:
    def test_defaults(self):
        cfg = HealthConfig()
        assert cfg.starting_health == 20.0
        assert cfg.passive_decay == 0.1
        assert cfg.eating_gain_rate == 1.0
        assert cfg.threat_damage == 5.0

    def test_invalid_starting_health(self):
        with pytest.raises(ValueError):
            HealthConfig(starting_health=0)


class TestNetworkConfig:
    def test_num_neurons(self):
        cfg = NetworkConfig(num_stimulus_channels=1)
        assert cfg.num_neurons == 3

    def test_num_connections(self):
        cfg = NetworkConfig(num_stimulus_channels=1)
        assert cfg.num_connections == 9

    def test_num_biases(self):
        cfg = NetworkConfig(num_stimulus_channels=1)
        assert cfg.num_biases == 3

    def test_num_params(self):
        cfg = NetworkConfig(num_stimulus_channels=1)
        assert cfg.num_params == 12

    def test_weight_magnitudes(self):
        cfg = NetworkConfig()
        mags = cfg.get_weight_magnitudes()
        assert len(mags) == 16
        assert float(mags[0]) == 0.0
        assert abs(float(mags[-1]) - 4.1) < 1e-5


class TestGeneticConfig:
    def test_invalid_mutation_rate(self):
        with pytest.raises(ValueError):
            GeneticConfig(mutation_rate=1.5)

    def test_invalid_pop_size(self):
        with pytest.raises(ValueError):
            GeneticConfig(population_size=1)


class TestSimulationConfig:
    def test_yaml_roundtrip(self):
        config = SimulationConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            config.save_yaml(f.name)
            loaded = SimulationConfig.from_yaml(f.name)

        assert loaded.environment.clump_scale == config.environment.clump_scale
        assert loaded.health.starting_health == config.health.starting_health
        assert loaded.genetic.population_size == config.genetic.population_size

    def test_with_updates(self):
        config = SimulationConfig()
        updated = config.with_updates(**{"environment.clump_scale": 20})
        assert updated.environment.clump_scale == 20
        assert config.environment.clump_scale == 10  # original unchanged

    def test_from_dict(self):
        data = {"health": {"starting_health": 50.0}, "genetic": {"population_size": 50}}
        config = SimulationConfig.from_dict(data)
        assert config.health.starting_health == 50.0
        assert config.genetic.population_size == 50

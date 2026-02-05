"""Second round sweep for binary+CTRNN with more aggressive settings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax

from habituation_experiment_alice.config import SimulationConfig
from habituation_experiment_alice.simulation import run_simulation


def run_sweep(name, overrides, generations=200, pop_size=50):
    """Run a single config and return summary."""
    config = SimulationConfig.from_dict(overrides)
    config = config.with_updates(
        **{"genetic.max_generations": generations, "genetic.population_size": pop_size}
    )
    key = jax.random.PRNGKey(config.seed)
    result = run_simulation(key, config, verbose=False)
    best = float(result.history.best_fitness[-1])
    mean = float(result.history.mean_fitness[-1])
    max_best = float(result.history.best_fitness.max())
    p1_passed = max_best >= 1.0
    return name, best, mean, max_best, p1_passed


def make_config(network_mode="simple", genotype_mode="binary", **kwargs):
    d = {
        "environment": {
            "clump_scale": 10,
            "phase1_lifetime": kwargs.get("phase1_lifetime", 500),
            "phase2_lifetime": 1000,
            "true_false_ratio": 0.5,
            "stimulus_magnitude": 1.0,
            "num_stimulus_channels": 1,
            "shared_environment": True,
        },
        "network": {
            "num_stimulus_channels": 1,
            "num_weight_magnitudes": 16,
            "max_weight": kwargs.get("max_weight", 4.1),
            "weight_spacing": "linear",
            "network_mode": network_mode,
            "tau_min": kwargs.get("tau_min", 0.5),
            "tau_max": kwargs.get("tau_max", 10.0),
            "num_tau_levels": 32,
        },
        "pain": {"delay": 2, "magnitude": 1.0},
        "health": {
            "starting_health": kwargs.get("starting_health", 20.0),
            "passive_decay": kwargs.get("passive_decay", 0.2),
            "eating_gain_rate": 1.0,
            "threat_damage": kwargs.get("threat_damage", 5.0),
        },
        "learning": {
            "enabled": True,
            "rule": "basic",
            "learning_rate": kwargs.get("learning_rate", 0.01),
            "weight_clip": kwargs.get("max_weight", 4.1),
        },
        "genetic": {
            "population_size": 50,
            "max_generations": 200,
            "mutation_rate": kwargs.get("mutation_rate", 0.05),
            "crossover_rate": kwargs.get("crossover_rate", 0.7),
            "tournament_size": kwargs.get("tournament_size", 2),
            "genotype_mode": genotype_mode,
            "mutation_std": kwargs.get("mutation_std", 0.1),
        },
        "simulation": {"seed": kwargs.get("seed", 99), "num_runs": 1},
    }
    return d


if __name__ == "__main__":
    configs = []

    # Binary CTRNN: try larger pop, longer, easier environment
    print("=" * 70)
    print("Binary + CTRNN sweep (round 2) - radical approaches")
    print("=" * 70)

    # Larger pop (100) + longer (500 gens)
    configs.append(("bc_tau2_pop100_500g", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_max=2.0, mutation_rate=0.05), 500, 100))

    # Lower passive decay (proven to help cont+simple)
    configs.append(("bc_tau2_pd01_mr10", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_max=2.0, mutation_rate=0.10, passive_decay=0.1), 300, 50))

    # Higher starting health
    configs.append(("bc_tau2_h40_mr10", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_max=2.0, mutation_rate=0.10, starting_health=40.0), 300, 50))

    # Very high mutation rate to force exploration
    configs.append(("bc_tau2_mr30_pop100", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_max=2.0, mutation_rate=0.30), 300, 100))

    # tau close to 1.0 (nearly instant neurons)
    configs.append(("bc_tau1_pd01_mr10", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_min=0.8, tau_max=1.2, mutation_rate=0.10, passive_decay=0.1), 300, 50))

    # Different seeds to test variance
    configs.append(("bc_tau2_pd01_mr10_s42", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_max=2.0, mutation_rate=0.10, passive_decay=0.1, seed=42), 300, 50))

    configs.append(("bc_tau2_pd01_mr10_s7", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_max=2.0, mutation_rate=0.10, passive_decay=0.1, seed=7), 300, 50))

    # Shorter phase1 (easier survival)
    configs.append(("bc_tau2_p1_250_mr10", make_config(
        network_mode="ctrnn", genotype_mode="binary",
        tau_max=2.0, mutation_rate=0.10, phase1_lifetime=250), 300, 50))

    # Also try continuous+SIMPLE refinement with the winning formula
    print("\nAlso running cont+simple refinements...")
    configs.append(("cs_mw2_pd01_mr15_std05_t2", make_config(
        network_mode="simple", genotype_mode="continuous",
        max_weight=2.0, passive_decay=0.1, mutation_rate=0.15,
        mutation_std=0.5, tournament_size=2), 200, 50))

    configs.append(("cs_mw2_pd01_mr10_std03_t2", make_config(
        network_mode="simple", genotype_mode="continuous",
        max_weight=2.0, passive_decay=0.1, mutation_rate=0.10,
        mutation_std=0.3, tournament_size=2), 200, 50))

    results = []
    for item in configs:
        name, cfg = item[0], item[1]
        gens = item[2] if len(item) > 2 else 200
        pop = item[3] if len(item) > 3 else 50
        print(f"  Running {name} ({gens}g, pop{pop})...", end="", flush=True)
        name, best, mean, max_best, p1 = run_sweep(name, cfg, generations=gens, pop_size=pop)
        marker = "YES" if p1 else "no"
        print(f"  best={best:7.2f}  mean={mean:5.2f}  max_best={max_best:7.2f}  p1={marker}")
        results.append((name, best, mean, max_best, p1))

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by max best fitness)")
    print("=" * 70)
    results.sort(key=lambda x: -x[3])
    for name, best, mean, max_best, p1 in results:
        marker = "YES" if p1 else "no"
        print(f"  {name:35s}  best={best:7.2f}  mean={mean:5.2f}  max_best={max_best:7.2f}  p1={marker}")

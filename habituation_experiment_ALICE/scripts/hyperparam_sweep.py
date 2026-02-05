"""Hyperparameter sweep for finding working configs for each mode combination.

Usage:
    python scripts/hyperparam_sweep.py
"""
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
    # Check if any generation had survivors (phase 1 passed)
    max_best = float(result.history.best_fitness.max())
    p1_passed = max_best >= 1.0
    return name, best, mean, max_best, p1_passed


def make_config(network_mode="simple", genotype_mode="binary", **kwargs):
    """Build config dict with overrides."""
    d = {
        "environment": {
            "clump_scale": 10,
            "phase1_lifetime": 500,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="all",
                       choices=["all", "cont_simple", "bin_ctrnn", "cont_ctrnn"])
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--pop", type=int, default=50)
    args = parser.parse_args()

    configs = []

    # =========================================================================
    # Continuous + SIMPLE sweeps
    # Key insight: dense large-weight networks need aggressive mutation
    # =========================================================================
    if args.sweep in ("all", "cont_simple"):
        # Baseline that barely worked: t3, mr=0.2, std=0.5
        # Try: smaller max_weight to reduce initial weight magnitudes
        configs.append(("cs_mw2_mr20_std05", make_config(
            network_mode="simple", genotype_mode="continuous",
            max_weight=2.0, mutation_rate=0.20, mutation_std=0.5,
            tournament_size=3)))

        configs.append(("cs_mw1_mr20_std03", make_config(
            network_mode="simple", genotype_mode="continuous",
            max_weight=1.0, mutation_rate=0.20, mutation_std=0.3,
            tournament_size=3)))

        # Higher mutation with default max_weight
        configs.append(("cs_mr30_std10_t3", make_config(
            network_mode="simple", genotype_mode="continuous",
            mutation_rate=0.30, mutation_std=1.0, tournament_size=3)))

        configs.append(("cs_mr50_std20_t3", make_config(
            network_mode="simple", genotype_mode="continuous",
            mutation_rate=0.50, mutation_std=2.0, tournament_size=3)))

        # Very aggressive: high mutation, smaller weights, larger pop
        configs.append(("cs_mw2_mr30_std10_t3", make_config(
            network_mode="simple", genotype_mode="continuous",
            max_weight=2.0, mutation_rate=0.30, mutation_std=1.0,
            tournament_size=3)))

        # Try higher starting health to give more time to learn
        configs.append(("cs_mw2_h40_mr20_std05", make_config(
            network_mode="simple", genotype_mode="continuous",
            max_weight=2.0, mutation_rate=0.20, mutation_std=0.5,
            tournament_size=3, starting_health=40.0)))

        # Try lower passive decay
        configs.append(("cs_mw2_pd01_mr20_std05", make_config(
            network_mode="simple", genotype_mode="continuous",
            max_weight=2.0, mutation_rate=0.20, mutation_std=0.5,
            tournament_size=3, passive_decay=0.1)))

        # Try no crossover (pure mutation)
        configs.append(("cs_mw2_nocross_mr30_std10", make_config(
            network_mode="simple", genotype_mode="continuous",
            max_weight=2.0, mutation_rate=0.30, mutation_std=1.0,
            tournament_size=3, crossover_rate=0.0)))

    # =========================================================================
    # Binary + CTRNN sweeps
    # Key insight: binary encoding needs bit flips to explore tau space
    # =========================================================================
    if args.sweep in ("all", "bin_ctrnn"):
        # Constrained tau + higher mutation
        configs.append(("bc_tau2_mr15", make_config(
            network_mode="ctrnn", genotype_mode="binary",
            tau_max=2.0, mutation_rate=0.15)))

        configs.append(("bc_tau2_mr20", make_config(
            network_mode="ctrnn", genotype_mode="binary",
            tau_max=2.0, mutation_rate=0.20)))

        configs.append(("bc_tau2_mr10_t3", make_config(
            network_mode="ctrnn", genotype_mode="binary",
            tau_max=2.0, mutation_rate=0.10, tournament_size=3)))

        configs.append(("bc_tau15_mr15", make_config(
            network_mode="ctrnn", genotype_mode="binary",
            tau_max=1.5, mutation_rate=0.15)))

        configs.append(("bc_tau2_mr20_t3", make_config(
            network_mode="ctrnn", genotype_mode="binary",
            tau_max=2.0, mutation_rate=0.20, tournament_size=3)))

        # Even more extreme
        configs.append(("bc_tau2_mr25", make_config(
            network_mode="ctrnn", genotype_mode="binary",
            tau_max=2.0, mutation_rate=0.25)))

        # Try tau_max=1.0 (very fast neurons, close to simple mode)
        configs.append(("bc_tau1_mr15", make_config(
            network_mode="ctrnn", genotype_mode="binary",
            tau_max=1.0, mutation_rate=0.15)))

    # =========================================================================
    # CTRNN + Continuous verification (extend known good config)
    # =========================================================================
    if args.sweep in ("all", "cont_ctrnn"):
        # Proven: tau_max=2.0, mutation_std=1.0, mutation_rate=0.1
        configs.append(("cc_tau2_std10_mr10", make_config(
            network_mode="ctrnn", genotype_mode="continuous",
            tau_max=2.0, mutation_rate=0.10, mutation_std=1.0)))

        # Try with tournament_size=3
        configs.append(("cc_tau2_std10_mr10_t3", make_config(
            network_mode="ctrnn", genotype_mode="continuous",
            tau_max=2.0, mutation_rate=0.10, mutation_std=1.0,
            tournament_size=3)))

    # Run all configs
    print(f"Running {len(configs)} configs ({args.gens} gens, pop {args.pop})")
    print("=" * 70)

    results = []
    for name, cfg in configs:
        print(f"  Running {name}...", end="", flush=True)
        name, best, mean, max_best, p1 = run_sweep(
            name, cfg, generations=args.gens, pop_size=args.pop
        )
        marker = "YES" if p1 else "no"
        print(f"  best={best:7.2f}  mean={mean:5.2f}  max_best={max_best:7.2f}  p1={marker}")
        results.append((name, best, mean, max_best, p1))

    # Summary sorted by max_best
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by max best fitness)")
    print("=" * 70)
    results.sort(key=lambda x: -x[3])
    for name, best, mean, max_best, p1 in results:
        marker = "YES" if p1 else "no"
        print(f"  {name:35s}  best={best:7.2f}  mean={mean:5.2f}  max_best={max_best:7.2f}  p1={marker}")

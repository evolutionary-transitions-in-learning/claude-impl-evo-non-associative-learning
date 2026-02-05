"""Entry point for running ALICE threat discrimination experiments.

Usage:
    python scripts/run_experiment.py --config config/default.yaml
    python scripts/run_experiment.py --config config/default.yaml --name my_test
    python scripts/run_experiment.py --generations 500 --pop-size 50 --num-runs 3

Results are saved to runs/<name>_<timestamp>/ with:
    config.yaml     - config used
    data/           - per-run history, genotypes, fitness
    viz/            - (empty, populated by visualize.py)
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax

from habituation_experiment_alice.config import SimulationConfig
from habituation_experiment_alice.io import (
    create_experiment_dir,
    save_experiment_summary,
    save_run_data,
)
from habituation_experiment_alice.simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Run ALICE threat discrimination evolution experiment"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file (uses defaults if not provided)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name (defaults to config filename or 'default')"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Max generations (overrides config)"
    )
    parser.add_argument(
        "--pop-size", type=int, default=None,
        help="Population size (overrides config)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=None,
        help="Number of independent runs"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = SimulationConfig.from_yaml(args.config)
    else:
        config = SimulationConfig()

    # Apply overrides
    updates = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.generations is not None:
        updates["genetic.max_generations"] = args.generations
    if args.pop_size is not None:
        updates["genetic.population_size"] = args.pop_size
    if args.num_runs is not None:
        updates["num_runs"] = args.num_runs

    if updates:
        config = config.with_updates(**updates)

    num_runs = config.num_runs
    verbose = not args.quiet

    # Create experiment directory
    project_root = Path(__file__).parent.parent
    experiment_dir = create_experiment_dir(
        project_root, name=args.name, config_path=args.config
    )

    if verbose:
        print("ALICE Threat Discrimination Evolution")
        print("=" * 50)
        print(f"Experiment dir: {experiment_dir}")
        print(f"Configuration:")
        print(f"  Population size: {config.genetic.population_size}")
        print(f"  Max generations: {config.genetic.max_generations}")
        print(f"  Phase 1 lifetime: {config.environment.phase1_lifetime}")
        print(f"  Phase 2 lifetime: {config.environment.phase2_lifetime}")
        print(f"  Clump scale: {config.environment.clump_scale}")
        print(f"  True/False ratio: {config.environment.true_false_ratio}")
        print(f"  Pain delay: {config.pain.delay}")
        print(f"  Starting health: {config.health.starting_health}")
        print(f"  Seed: {config.seed}")
        print(f"  Runs: {num_runs}")

    key = jax.random.PRNGKey(config.seed)
    keys = jax.random.split(key, num_runs)
    results = []

    for i, run_key in enumerate(keys):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Run {i+1}/{num_runs}")
            print(f"{'='*50}")

        result = run_simulation(run_key, config, verbose=verbose)
        results.append(result)

        # Save run data immediately (so partial results survive crashes)
        save_run_data(experiment_dir, i, result)
        if verbose:
            print(f"  Run {i} data saved to {experiment_dir / 'data' / f'run_{i:03d}'}")

    # Save summary and config
    save_experiment_summary(experiment_dir, results, config)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Experiment complete!")
        print(f"Results: {experiment_dir}")
        print(f"  Data:   {experiment_dir / 'data'}")
        print(f"  Viz:    {experiment_dir / 'viz'}  (run gen_all_viz.py to populate)")


if __name__ == "__main__":
    main()

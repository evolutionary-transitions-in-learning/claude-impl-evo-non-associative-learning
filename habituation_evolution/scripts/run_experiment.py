#!/usr/bin/env python3
"""
Run the full experimental grid to reproduce results from Todd & Miller.

This script collects data for:
1. Main experiment: 6 clump-scales × 7 accuracies × N runs
2. Lifespan hypothesis test: clump-scale 80, 70% accuracy, 4000 timesteps
3. Network architecture analysis for successful runs

Data is saved incrementally to allow resumption and separate visualization.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from habituation_evolution import (
    SimulationConfig,
    run_simulation,
    decode_genotype,
    compute_optimal_fitness,
    compute_sensory_only_fitness,
    compute_success_threshold,
)


@dataclass
class RunResult:
    """Results from a single simulation run."""

    # Condition parameters
    clump_scale: int
    sensory_accuracy: float
    lifespan: int
    run_id: int

    # Outcome
    generations_to_success: int
    success: bool

    # Final statistics
    final_mean_fitness: float
    final_best_fitness: float
    final_std_fitness: float

    # Fitness history (sampled every N generations to save space)
    fitness_history_generations: list
    fitness_history_mean: list
    fitness_history_best: list

    # Best evolved network
    best_genotype: list  # Binary genotype as list
    best_weights: list   # Decoded weights
    best_biases: list    # Decoded biases
    best_learnable: list # Learnable mask

    # Metadata
    seed: int
    timestamp: str
    runtime_seconds: float


@dataclass
class ExperimentConfig:
    """Configuration for the full experiment."""

    # Main experiment grid
    clump_scales: list
    sensory_accuracies: list
    runs_per_condition: int

    # Simulation parameters
    population_size: int
    max_generations: int
    lifespan: int
    mutation_rate: float
    crossover_rate: float

    # Lifespan hypothesis test
    run_lifespan_test: bool
    lifespan_test_lifespan: int
    lifespan_test_runs: int

    # Data collection
    history_sample_interval: int  # Save fitness every N generations

    # Output
    output_dir: str
    experiment_name: str


def get_default_experiment_config() -> ExperimentConfig:
    """Get default experiment configuration matching the paper."""
    return ExperimentConfig(
        # Main experiment - matches paper
        clump_scales=[1, 5, 10, 20, 40, 80],
        sensory_accuracies=[0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
        runs_per_condition=5,  # Paper used 5-10

        # Simulation parameters
        population_size=100,
        max_generations=2000,
        lifespan=1000,
        mutation_rate=0.01,
        crossover_rate=0.7,

        # Lifespan hypothesis test
        run_lifespan_test=True,
        lifespan_test_lifespan=4000,
        lifespan_test_runs=10,

        # Data collection
        history_sample_interval=10,  # Every 10 generations

        # Output
        output_dir="data/experiments",
        experiment_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )


def run_single_condition(
    key: jax.random.PRNGKey,
    clump_scale: int,
    sensory_accuracy: float,
    lifespan: int,
    run_id: int,
    exp_config: ExperimentConfig,
    verbose: bool = True,
) -> RunResult:
    """Run a single experimental condition."""

    start_time = time.time()
    # Generate a seed for logging purposes (use key bits directly)
    seed = int(jax.random.key_data(key)[0]) % (2**31)

    # Create simulation config
    sim_config = SimulationConfig()
    sim_config.environment.clump_scale = clump_scale
    sim_config.environment.sensory_accuracy = sensory_accuracy
    sim_config.environment.lifespan = lifespan
    sim_config.genetic.population_size = exp_config.population_size
    sim_config.genetic.max_generations = exp_config.max_generations
    sim_config.genetic.mutation_rate = exp_config.mutation_rate
    sim_config.genetic.crossover_rate = exp_config.crossover_rate
    sim_config.seed = seed

    # Compute success threshold for progress display
    threshold = compute_success_threshold(sim_config)

    if verbose:
        print(f"  Running: clump={clump_scale}, acc={sensory_accuracy:.0%}, "
              f"life={lifespan}, run={run_id} (threshold={threshold:.1f})")

    # Progress callback for generation updates
    def progress_callback(gen, max_gen, mean_fit, best_fit):
        if gen % 100 == 0 or gen == max_gen:  # Update every 100 generations
            print(f"\r    Gen {gen:4d}/{max_gen} | mean={mean_fit:.1f} best={best_fit:.1f}", end="", flush=True)

    # Run simulation
    result = run_simulation(
        key, sim_config, verbose=False,
        progress_callback=progress_callback if verbose else None
    )

    runtime = time.time() - start_time

    # Save full fitness history (every generation) for post-hoc threshold analysis
    history_gens = list(range(len(result.history.mean_fitness)))

    # Decode best genotype
    best_genotype = result.final_state.best_genotype
    best_params = decode_genotype(best_genotype, sim_config.network)

    return RunResult(
        clump_scale=clump_scale,
        sensory_accuracy=sensory_accuracy,
        lifespan=lifespan,
        run_id=run_id,
        generations_to_success=result.generations_to_success,
        success=result.success,
        final_mean_fitness=float(result.final_state.mean_fitness),
        final_best_fitness=float(result.final_state.best_fitness),
        final_std_fitness=float(jnp.std(result.final_state.fitness_scores)),
        fitness_history_generations=history_gens,
        fitness_history_mean=[float(result.history.mean_fitness[i]) for i in history_gens],
        fitness_history_best=[float(result.history.best_fitness[i]) for i in history_gens],
        best_genotype=[int(x) for x in best_genotype],
        best_weights=[float(x) for x in best_params.weights],
        best_biases=[float(x) for x in best_params.biases],
        best_learnable=[bool(x) for x in best_params.learnable_mask],
        seed=seed,
        timestamp=datetime.now().isoformat(),
        runtime_seconds=runtime,
    )


def save_result(result: RunResult, output_dir: Path):
    """Save a single result to JSON file."""
    filename = (f"result_clump{result.clump_scale}_"
                f"acc{int(result.sensory_accuracy*100)}_"
                f"life{result.lifespan}_"
                f"run{result.run_id}.json")

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(asdict(result), f, indent=2)


def load_existing_results(output_dir: Path) -> set:
    """Load set of already-completed condition keys."""
    completed = set()
    if output_dir.exists():
        for filepath in output_dir.glob("result_*.json"):
            with open(filepath) as f:
                data = json.load(f)
                key = (data['clump_scale'], data['sensory_accuracy'],
                       data['lifespan'], data['run_id'])
                completed.add(key)
    return completed


def run_main_experiment(
    exp_config: ExperimentConfig,
    resume: bool = True,
    verbose: bool = True,
):
    """Run the main experimental grid."""

    output_dir = Path(exp_config.output_dir) / exp_config.experiment_name / "main"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config_path = output_dir.parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(exp_config), f, indent=2)

    # Check for existing results if resuming
    completed = load_existing_results(output_dir) if resume else set()

    # Build list of conditions to run
    conditions = []
    for clump_scale in exp_config.clump_scales:
        for accuracy in exp_config.sensory_accuracies:
            for run_id in range(exp_config.runs_per_condition):
                key = (clump_scale, accuracy, exp_config.lifespan, run_id)
                if key not in completed:
                    conditions.append(key)

    total = len(exp_config.clump_scales) * len(exp_config.sensory_accuracies) * exp_config.runs_per_condition
    print(f"Main experiment: {len(conditions)} conditions to run ({total - len(conditions)} already completed)")

    # Run conditions
    master_key = jax.random.PRNGKey(42)

    for i, (clump_scale, accuracy, lifespan, run_id) in enumerate(conditions):
        master_key, run_key = jax.random.split(master_key)

        if verbose:
            print(f"[{i+1}/{len(conditions)}]", end=" ")

        result = run_single_condition(
            run_key, clump_scale, accuracy, lifespan, run_id, exp_config, verbose
        )

        save_result(result, output_dir)

        if verbose:
            status = f"converged@gen{result.generations_to_success}" if result.success else "no convergence"
            print(f"\n    -> {status} in {result.runtime_seconds:.1f}s")

    print(f"Main experiment complete. Results saved to {output_dir}")


def run_lifespan_test(
    exp_config: ExperimentConfig,
    resume: bool = True,
    verbose: bool = True,
):
    """Run the lifespan hypothesis test."""

    if not exp_config.run_lifespan_test:
        print("Lifespan test disabled in config")
        return

    output_dir = Path(exp_config.output_dir) / exp_config.experiment_name / "lifespan_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed parameters for this test
    clump_scale = 80
    accuracy = 0.70
    lifespan = exp_config.lifespan_test_lifespan

    # Check existing
    completed = load_existing_results(output_dir) if resume else set()

    conditions = []
    for run_id in range(exp_config.lifespan_test_runs):
        key = (clump_scale, accuracy, lifespan, run_id)
        if key not in completed:
            conditions.append(key)

    print(f"Lifespan test: {len(conditions)} runs to complete")

    master_key = jax.random.PRNGKey(123)

    for i, (cs, acc, life, run_id) in enumerate(conditions):
        master_key, run_key = jax.random.split(master_key)

        if verbose:
            print(f"[{i+1}/{len(conditions)}]", end=" ")

        result = run_single_condition(
            run_key, cs, acc, life, run_id, exp_config, verbose
        )

        save_result(result, output_dir)

        if verbose:
            status = f"converged@gen{result.generations_to_success}" if result.success else "no convergence"
            print(f"\n    -> {status} in {result.runtime_seconds:.1f}s")

    print(f"Lifespan test complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run habituation/sensitization evolution experiments"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment config JSON (uses defaults if not provided)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs per condition"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming"
    )
    parser.add_argument(
        "--main-only", action="store_true",
        help="Only run main experiment (skip lifespan test)"
    )
    parser.add_argument(
        "--lifespan-only", action="store_true",
        help="Only run lifespan test"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with reduced parameters"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
            exp_config = ExperimentConfig(**config_dict)
    else:
        exp_config = get_default_experiment_config()

    # Apply command-line overrides
    exp_config.output_dir = args.output_dir
    if args.name:
        exp_config.experiment_name = args.name
    exp_config.runs_per_condition = args.runs

    if args.quick:
        print("Running in quick test mode...")
        exp_config.clump_scales = [1, 10, 40]
        exp_config.sensory_accuracies = [0.60, 0.75, 0.90]
        exp_config.runs_per_condition = 2
        exp_config.max_generations = 200
        exp_config.population_size = 50
        exp_config.lifespan_test_runs = 2

    if args.main_only:
        exp_config.run_lifespan_test = False

    resume = not args.no_resume
    verbose = not args.quiet

    print(f"Experiment: {exp_config.experiment_name}")
    print(f"Output: {exp_config.output_dir}")
    print(f"Grid: {len(exp_config.clump_scales)} clump-scales × "
          f"{len(exp_config.sensory_accuracies)} accuracies × "
          f"{exp_config.runs_per_condition} runs")
    print()

    # Run experiments
    if not args.lifespan_only:
        run_main_experiment(exp_config, resume=resume, verbose=verbose)
        print()

    if exp_config.run_lifespan_test and not args.main_only:
        run_lifespan_test(exp_config, resume=resume, verbose=verbose)


if __name__ == "__main__":
    main()

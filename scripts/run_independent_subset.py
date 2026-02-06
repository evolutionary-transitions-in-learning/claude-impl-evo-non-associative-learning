#!/usr/bin/env python3
"""
Run a strategic subset of conditions with INDEPENDENT environments.

This tests whether independent environments (as described in the paper)
reproduce the expected patterns, particularly:
- The anomaly at clump=80 (high clump + low accuracy = slow evolution)
- U-shaped curves within clump-scale rows
- Calibration against paper's known data points

Strategic subset:
- clump=80 at all 7 accuracies × 3 runs = 21 runs (check anomaly/U-shape)
- clump=20 at acc=70% × 3 runs = 3 runs (calibration: paper says 61 gens)
- clump=10 at acc=70% × 3 runs = 3 runs (control comparison)
Total: 27 runs

Expected runtime: ~11-12 minutes
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from habituation_evolution import (
    SimulationConfig,
    run_simulation,
    decode_genotype,
    compute_optimal_fitness,
    compute_sensory_only_fitness,
    compute_success_threshold,
)


def run_single(key, clump_scale, accuracy, lifespan, run_id, max_generations, pop_size):
    """Run a single simulation and return result dict."""
    start = time.time()
    seed = int(jax.random.key_data(key)[0]) % (2**31)

    sim_config = SimulationConfig()
    sim_config.environment.clump_scale = clump_scale
    sim_config.environment.sensory_accuracy = accuracy
    sim_config.environment.lifespan = lifespan
    sim_config.genetic.population_size = pop_size
    sim_config.genetic.max_generations = max_generations
    sim_config.genetic.mutation_rate = 0.01
    sim_config.genetic.crossover_rate = 0.7
    sim_config.seed = seed

    threshold = compute_success_threshold(sim_config)

    def progress_cb(gen, max_gen, mean_fit, best_fit):
        if gen % 200 == 0 or gen == max_gen:
            print(f"\r    Gen {gen:4d}/{max_gen} | mean={mean_fit:.1f} best={best_fit:.1f}", end="", flush=True)

    result = run_simulation(key, sim_config, verbose=False, progress_callback=progress_cb)

    runtime = time.time() - start

    # Save full fitness history
    history_gens = list(range(len(result.history.mean_fitness)))

    # Decode best genotype
    best_genotype = result.final_state.best_genotype
    best_params = decode_genotype(best_genotype, sim_config.network)

    return {
        "clump_scale": clump_scale,
        "sensory_accuracy": accuracy,
        "lifespan": lifespan,
        "run_id": run_id,
        "generations_to_success": result.generations_to_success,
        "success": result.success,
        "final_mean_fitness": float(result.final_state.mean_fitness),
        "final_best_fitness": float(result.final_state.best_fitness),
        "final_std_fitness": float(jnp.std(result.final_state.fitness_scores)),
        "fitness_history_generations": history_gens,
        "fitness_history_mean": [float(result.history.mean_fitness[i]) for i in history_gens],
        "fitness_history_best": [float(result.history.best_fitness[i]) for i in history_gens],
        "best_genotype": [int(x) for x in best_genotype],
        "best_weights": [float(x) for x in best_params.weights],
        "best_biases": [float(x) for x in best_params.biases],
        "best_learnable": [bool(x) for x in best_params.learnable_mask],
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": runtime,
    }


def main():
    output_dir = Path("data/experiments/independent_env_subset/main")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strategic conditions
    accuracies = [0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
    runs_per = 3
    max_generations = 2000
    pop_size = 100
    lifespan = 1000

    conditions = []
    # clump=80, all accuracies (the anomaly row)
    for acc in accuracies:
        for run_id in range(runs_per):
            conditions.append((80, acc, run_id))
    # clump=20, acc=70% (calibration point: paper says 61 gens)
    for run_id in range(runs_per):
        conditions.append((20, 0.70, run_id))
    # clump=10, acc=70% (control)
    for run_id in range(runs_per):
        conditions.append((10, 0.70, run_id))

    # Check for already completed
    completed = set()
    for f in output_dir.glob("result_*.json"):
        with open(f) as fh:
            d = json.load(fh)
            completed.add((d['clump_scale'], d['sensory_accuracy'], d['run_id']))

    remaining = [(cs, acc, rid) for cs, acc, rid in conditions if (cs, acc, rid) not in completed]

    print(f"Independent environments strategic subset")
    print(f"Total conditions: {len(conditions)}, already done: {len(conditions) - len(remaining)}, remaining: {len(remaining)}")
    print(f"Max generations: {max_generations}, Pop size: {pop_size}, Lifespan: {lifespan}")
    print()

    master_key = jax.random.PRNGKey(999)
    total_start = time.time()

    for i, (clump_scale, accuracy, run_id) in enumerate(remaining):
        master_key, run_key = jax.random.split(master_key)

        print(f"[{i+1}/{len(remaining)}] clump={clump_scale}, acc={accuracy:.0%}, run={run_id}")

        result = run_single(run_key, clump_scale, accuracy, lifespan, run_id, max_generations, pop_size)

        # Save
        filename = f"result_clump{clump_scale}_acc{int(accuracy*100)}_life{lifespan}_run{run_id}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)

        status = f"converged@gen{result['generations_to_success']}" if result['success'] else "no convergence"
        elapsed = time.time() - total_start
        print(f"\n    -> {status} | final_mean={result['final_mean_fitness']:.1f} | {result['runtime_seconds']:.1f}s | total elapsed: {elapsed:.0f}s")

    total_time = time.time() - total_start
    print(f"\nAll done in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Quick summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_results = []
    for f in sorted(output_dir.glob("result_*.json")):
        with open(f) as fh:
            all_results.append(json.load(fh))

    # Group by condition
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in all_results:
        by_condition[(r['clump_scale'], r['sensory_accuracy'])].append(r)

    print(f"\n{'Clump':>6} {'Acc':>6} {'MeanFinal':>10} {'BestFinal':>10} {'Converged':>10}")
    print("-" * 50)
    for (cs, acc), runs in sorted(by_condition.items()):
        mean_final = sum(r['final_mean_fitness'] for r in runs) / len(runs)
        best_final = sum(r['final_best_fitness'] for r in runs) / len(runs)
        n_conv = sum(1 for r in runs if r['success'])
        print(f"{cs:>6} {acc:>6.0%} {mean_final:>10.1f} {best_final:>10.1f} {n_conv:>5}/{len(runs)}")


if __name__ == "__main__":
    main()

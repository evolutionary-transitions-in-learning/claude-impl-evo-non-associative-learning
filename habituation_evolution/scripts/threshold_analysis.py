#!/usr/bin/env python3
"""
Post-hoc threshold analysis for finding the best success_threshold_ratio.

This script loads full mean fitness histories from simulation results and
generates heatmaps for different threshold ratios to find which one best
matches the published data from Todd & Miller.

Usage:
    python scripts/threshold_analysis.py data/experiments/paper_reproduction
    python scripts/threshold_analysis.py data/experiments/paper_reproduction --ratios 0.3 0.5 0.7 0.9
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(experiment_dir: Path) -> list[dict]:
    """Load all main experiment result JSONs."""
    results = []
    main_dir = experiment_dir / "main"
    for f in sorted(main_dir.glob("result_*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def compute_threshold(accuracy: float, ratio: float, lifespan: int = 1000) -> float:
    """Compute the success threshold for a given accuracy and ratio.

    threshold = sensory_only + ratio * (optimal - sensory_only)

    Where:
        optimal = lifespan / 2  (eat all food, avoid all poison)
        sensory_only = lifespan * (accuracy - 0.5)  (eat when smell says food)
    """
    optimal = lifespan / 2.0
    sensory_only = lifespan * (accuracy - 0.5)
    return sensory_only + ratio * (optimal - sensory_only)


def find_convergence_generation(mean_fitness_history: list[float], threshold: float) -> int | None:
    """Find the first generation where mean fitness >= threshold.

    Returns:
        Generation number, or None if never crossed.
    """
    for gen, fitness in enumerate(mean_fitness_history):
        if fitness >= threshold:
            return gen
    return None


def analyze_ratio(results: list[dict], ratio: float) -> dict:
    """Analyze results for a given threshold ratio.

    Returns dict mapping (clump_scale, accuracy) -> mean_generations_to_success
    """
    # Group by condition
    conditions = {}
    for r in results:
        key = (r['clump_scale'], r['sensory_accuracy'])
        if key not in conditions:
            conditions[key] = []

        lifespan = r['lifespan']
        threshold = compute_threshold(r['sensory_accuracy'], ratio, lifespan)

        # Use full history (generation indices match list indices)
        mean_history = r['fitness_history_mean']
        gen_indices = r['fitness_history_generations']

        # Find convergence
        conv_gen = None
        for i, gen in enumerate(gen_indices):
            if mean_history[i] >= threshold:
                conv_gen = gen
                break

        # If not converged, use max_generations (2000)
        conditions[key].append(conv_gen if conv_gen is not None else 2000)

    # Compute means
    return {k: np.mean(v) for k, v in conditions.items()}


def make_heatmap(condition_means: dict, ratio: float, ax, clump_scales: list, accuracies: list):
    """Plot a heatmap on the given axes."""
    grid = np.full((len(clump_scales), len(accuracies)), np.nan)
    for i, cs in enumerate(clump_scales):
        for j, acc in enumerate(accuracies):
            key = (cs, acc)
            if key in condition_means:
                grid[i, j] = condition_means[key]

    sns.heatmap(
        grid,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=[f'{int(a*100)}%' for a in accuracies],
        yticklabels=[str(cs) for cs in clump_scales],
        ax=ax,
        vmin=0,
        vmax=2000,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Mean Generations'},
    )
    ax.set_xlabel('Sensory Accuracy')
    ax.set_ylabel('Clump Scale')
    ax.set_title(f'Threshold Ratio = {ratio:.2f}')


def main():
    parser = argparse.ArgumentParser(description="Post-hoc threshold analysis")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "--ratios", nargs='+', type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Threshold ratios to analyze"
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    results = load_results(experiment_dir)
    print(f"Loaded {len(results)} results from {experiment_dir / 'main'}")

    if not results:
        print("No results found!")
        return

    # Extract unique conditions
    clump_scales = sorted(set(r['clump_scale'] for r in results))
    accuracies = sorted(set(r['sensory_accuracy'] for r in results))

    ratios = args.ratios

    # Create figure with subplots
    n_ratios = len(ratios)
    cols = min(3, n_ratios)
    rows = (n_ratios + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n_ratios == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ratio in enumerate(ratios):
        print(f"\nRatio = {ratio:.2f}")
        condition_means = analyze_ratio(results, ratio)

        # Print table
        print(f"{'Clump':>6}", end="")
        for acc in accuracies:
            print(f"  {int(acc*100):>5}%", end="")
        print()
        for cs in clump_scales:
            print(f"{cs:>6}", end="")
            for acc in accuracies:
                val = condition_means.get((cs, acc), float('nan'))
                print(f"  {val:>6.0f}", end="")
            print()

        make_heatmap(condition_means, ratio, axes[i], clump_scales, accuracies)

    # Hide unused subplots
    for j in range(n_ratios, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Mean Generations to Evolve Cluster-Tracking\nby Success Threshold Ratio', fontsize=16, y=1.02)
    plt.tight_layout()

    output_dir = experiment_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "threshold_ratio_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to {output_path}")

    # Also generate individual high-res heatmaps for each ratio
    for ratio in ratios:
        condition_means = analyze_ratio(results, ratio)
        fig_single, ax_single = plt.subplots(figsize=(10, 7))
        make_heatmap(condition_means, ratio, ax_single, clump_scales, accuracies)
        ax_single.set_title(f'Mean Generations to Evolve Cluster-Tracking\n(threshold ratio = {ratio:.2f}, 2000 = no convergence)')
        plt.tight_layout()
        single_path = output_dir / f"heatmap_ratio_{ratio:.2f}.png"
        fig_single.savefig(single_path, dpi=150, bbox_inches='tight')
        plt.close(fig_single)

    print(f"Saved individual heatmaps to {output_dir}/heatmap_ratio_*.png")


if __name__ == "__main__":
    main()

"""Generate visualizations from experiment data.

Usage:
    python scripts/visualize.py runs/<experiment_name>/

Reads from data/ subfolder and writes plots to viz/ subfolder.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from habituation_experiment_alice.io import load_experiment


def plot_fitness_history(histories, summary, viz_dir):
    """Plot mean and best fitness over generations for all runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean fitness
    ax = axes[0]
    for i, h in enumerate(histories):
        gens = np.arange(len(h["mean_fitness"]))
        ax.plot(gens, h["mean_fitness"], alpha=0.6, label=f"Run {i}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    ax.set_title("Mean Fitness Over Generations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Best fitness
    ax = axes[1]
    for i, h in enumerate(histories):
        gens = np.arange(len(h["best_fitness"]))
        ax.plot(gens, h["best_fitness"], alpha=0.6, label=f"Run {i}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Over Generations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / "fitness_history.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_fitness_summary(histories, summary, viz_dir):
    """Plot mean fitness with std shading (averaged across runs)."""
    # Find the shortest history (all should be same length)
    min_len = min(len(h["mean_fitness"]) for h in histories)
    gens = np.arange(min_len)

    # Stack and compute stats across runs
    mean_stack = np.stack([h["mean_fitness"][:min_len] for h in histories])
    best_stack = np.stack([h["best_fitness"][:min_len] for h in histories])

    mean_of_means = np.mean(mean_stack, axis=0)
    std_of_means = np.std(mean_stack, axis=0)
    mean_of_best = np.mean(best_stack, axis=0)
    std_of_best = np.std(best_stack, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gens, mean_of_means, color="steelblue", label="Mean fitness (avg)")
    ax.fill_between(
        gens,
        mean_of_means - std_of_means,
        mean_of_means + std_of_means,
        color="steelblue", alpha=0.2,
    )

    ax.plot(gens, mean_of_best, color="coral", label="Best fitness (avg)")
    ax.fill_between(
        gens,
        mean_of_best - std_of_best,
        mean_of_best + std_of_best,
        color="coral", alpha=0.2,
    )

    # Draw tier boundary
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Phase 1 survival threshold")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Fitness Evolution (n={len(histories)} runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / "fitness_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_survivors(histories, summary, viz_dir):
    """Plot number of phase-1 survivors over generations."""
    pop_size = summary["config_summary"]["population_size"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, h in enumerate(histories):
        gens = np.arange(len(h["num_survivors"]))
        ax.plot(gens, h["num_survivors"], alpha=0.6, label=f"Run {i}")

    ax.axhline(y=pop_size, color="gray", linestyle="--", alpha=0.5, label=f"Pop size ({pop_size})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Survivors (fitness > 0)")
    ax.set_title("Phase 1 Survivors Over Generations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(viz_dir / "survivors.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_fitness_distribution(experiment_dir, viz_dir):
    """Plot final fitness distribution across all runs."""
    data_dir = experiment_dir / "data"
    run_dirs = sorted(data_dir.glob("run_*"))

    fig, ax = plt.subplots(figsize=(10, 5))

    all_fitness = []
    for run_dir in run_dirs:
        fitness_file = run_dir / "final_fitness.npy"
        if fitness_file.exists():
            fitness = np.load(fitness_file)
            all_fitness.append(fitness)

    if not all_fitness:
        plt.close()
        return

    # Combine all runs
    combined = np.concatenate(all_fitness)

    ax.hist(combined, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Phase 1 survival threshold")
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Count")
    ax.set_title(f"Final Fitness Distribution ({len(all_fitness)} runs, {len(combined)} agents)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / "fitness_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_fitness_std(histories, summary, viz_dir):
    """Plot fitness standard deviation over generations."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, h in enumerate(histories):
        gens = np.arange(len(h["std_fitness"]))
        ax.plot(gens, h["std_fitness"], alpha=0.6, label=f"Run {i}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Std Dev")
    ax.set_title("Population Fitness Diversity Over Generations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / "fitness_std.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from ALICE experiment data"
    )
    parser.add_argument(
        "experiment_dir", type=str,
        help="Path to experiment directory (e.g., runs/default_20240101_120000/)"
    )

    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: {experiment_dir} does not exist")
        sys.exit(1)

    summary_file = experiment_dir / "data" / "summary.json"
    if not summary_file.exists():
        print(f"Error: No summary.json found in {experiment_dir / 'data'}")
        sys.exit(1)

    viz_dir = experiment_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    print(f"Loading experiment data from {experiment_dir}")
    summary, histories = load_experiment(experiment_dir)
    print(f"  Found {len(histories)} runs")

    print("Generating plots...")

    plot_fitness_history(histories, summary, viz_dir)
    print("  fitness_history.png")

    plot_fitness_summary(histories, summary, viz_dir)
    print("  fitness_summary.png")

    plot_survivors(histories, summary, viz_dir)
    print("  survivors.png")

    plot_fitness_distribution(experiment_dir, viz_dir)
    print("  fitness_distribution.png")

    plot_fitness_std(histories, summary, viz_dir)
    print("  fitness_std.png")

    print(f"\nAll visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()

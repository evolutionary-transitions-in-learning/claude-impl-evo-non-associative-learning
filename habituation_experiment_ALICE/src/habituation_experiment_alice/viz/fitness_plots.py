"""Fitness-related visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import COLORS, DPI


def plot_fitness_by_phase(histories, summary, viz_dir, **kwargs) -> Path | None:
    """Plot phase 1 survival fraction and phase 2 health separately over generations."""
    if not histories:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phase 1: mean survival fraction
    ax = axes[0]
    for i, h in enumerate(histories):
        if "phase1_mean_survival_frac" not in h:
            return None
        gens = np.arange(len(h["phase1_mean_survival_frac"]))
        ax.plot(gens, h["phase1_mean_survival_frac"], alpha=0.6, label=f"Run {i}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Survival Fraction")
    ax.set_title("Phase 1: Mean Survival Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, label="Full survival")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase 2: mean health (survivors only)
    ax = axes[1]
    for i, h in enumerate(histories):
        gens = np.arange(len(h["phase2_mean_health"]))
        ax.plot(gens, h["phase2_mean_health"], alpha=0.6, label=f"Run {i}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Phase 2 Health (Survivors)")
    ax.set_title("Phase 2: Mean Final Health")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = viz_dir / "fitness_by_phase.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_fitness_summary(histories, summary, viz_dir, **kwargs) -> Path | None:
    """Plot mean fitness with std shading (averaged across runs)."""
    if not histories:
        return None

    min_len = min(len(h["mean_fitness"]) for h in histories)
    gens = np.arange(min_len)

    mean_stack = np.stack([h["mean_fitness"][:min_len] for h in histories])
    best_stack = np.stack([h["best_fitness"][:min_len] for h in histories])

    mean_of_means = np.mean(mean_stack, axis=0)
    std_of_means = np.std(mean_stack, axis=0)
    mean_of_best = np.mean(best_stack, axis=0)
    std_of_best = np.std(best_stack, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gens, mean_of_means, color=COLORS["fitness_mean"], label="Mean fitness (avg)")
    ax.fill_between(
        gens,
        mean_of_means - std_of_means,
        mean_of_means + std_of_means,
        color=COLORS["fitness_mean"], alpha=0.2,
    )

    ax.plot(gens, mean_of_best, color=COLORS["fitness_best"], label="Best fitness (avg)")
    ax.fill_between(
        gens,
        mean_of_best - std_of_best,
        mean_of_best + std_of_best,
        color=COLORS["fitness_best"], alpha=0.2,
    )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Phase 1 survival threshold")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Fitness Evolution (n={len(histories)} runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = viz_dir / "fitness_summary.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_fitness_distribution(histories, summary, viz_dir, experiment_dir=None, **kwargs) -> Path | None:
    """Plot final fitness distribution across all runs."""
    if experiment_dir is None:
        return None

    data_dir = experiment_dir / "data"
    run_dirs = sorted(data_dir.glob("run_*"))

    all_fitness = []
    for run_dir in run_dirs:
        fitness_file = run_dir / "final_fitness.npy"
        if fitness_file.exists():
            all_fitness.append(np.load(fitness_file))

    if not all_fitness:
        return None

    combined = np.concatenate(all_fitness)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(combined, bins=50, edgecolor="black", alpha=0.7, color=COLORS["fitness_mean"])
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Phase 1 survival threshold")
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Count")
    ax.set_title(f"Final Fitness Distribution ({len(all_fitness)} runs, {len(combined)} agents)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = viz_dir / "fitness_distribution.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_fitness_std(histories, summary, viz_dir, **kwargs) -> Path | None:
    """Plot fitness standard deviation over generations."""
    if not histories:
        return None

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
    path = viz_dir / "fitness_std.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_survival_rate(histories, summary, viz_dir, **kwargs) -> Path | None:
    """Plot phase 1 survival rate over generations."""
    if not histories:
        return None

    pop_size = summary["config_summary"]["population_size"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, h in enumerate(histories):
        key = "num_survived_phase1" if "num_survived_phase1" in h else "num_survivors"
        gens = np.arange(len(h[key]))
        rate = h[key] / pop_size * 100
        ax.plot(gens, rate, alpha=0.6, label=f"Run {i}")

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="100%")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Phase 1 Survival Rate (%)")
    ax.set_title("Phase 1 Survival Rate Over Generations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-5, top=105)

    plt.tight_layout()
    path = viz_dir / "survival_rate.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path

"""Genetics and network parameter visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import ALL_PARAM_LABELS, CONNECTION_LABELS, BIAS_LABELS, DPI


def plot_genotype_diversity(histories, summary, viz_dir, **kwargs) -> Path | None:
    """Plot mean pairwise Hamming distance over generations."""
    if not histories or "genotype_diversity" not in histories[0]:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, h in enumerate(histories):
        gens = np.arange(len(h["genotype_diversity"]))
        ax.plot(gens, h["genotype_diversity"], alpha=0.6, label=f"Run {i}")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Pairwise Hamming Distance (normalized)")
    ax.set_title("Genotype Diversity Over Generations")
    ax.set_ylim(-0.05, 0.55)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = viz_dir / "genotype_diversity.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_weight_evolution_heatmap(histories, summary, viz_dir, **kwargs) -> Path | None:
    """Plot heatmap of best agent's weights and biases over generations."""
    if not histories or "best_weights" not in histories[0]:
        return None

    # Use run 0
    h = histories[0]
    weights = h["best_weights"]    # (num_gens, 9)
    biases = h["best_biases"]      # (num_gens, 3)

    # Combine into single matrix
    all_params = np.concatenate([weights, biases], axis=1)  # (num_gens, 12)
    num_gens, num_params = all_params.shape

    labels = ALL_PARAM_LABELS[:num_params]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Transpose so parameters are on y-axis, generations on x-axis
    im = ax.imshow(
        all_params.T,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-np.max(np.abs(all_params)),
        vmax=np.max(np.abs(all_params)),
    )

    ax.set_yticks(range(num_params))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Parameter")
    ax.set_title("Best Agent Weight Evolution (Run 0)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Weight Value")

    # Add horizontal line separating weights from biases
    num_conns = weights.shape[1]
    ax.axhline(y=num_conns - 0.5, color="white", linewidth=2, linestyle="--")

    plt.tight_layout()
    path = viz_dir / "weight_evolution_heatmap.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_learnable_distribution(histories, summary, viz_dir, **kwargs) -> Path | None:
    """Plot fraction of population with each parameter set to learnable over generations."""
    if not histories or "pop_learnable_frac" not in histories[0]:
        return None

    # Use run 0
    h = histories[0]
    pop_learn = h["pop_learnable_frac"]  # (num_gens, 12)
    num_gens, num_params = pop_learn.shape

    labels = ALL_PARAM_LABELS[:num_params]
    gens = np.arange(num_gens)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: stacked area chart
    ax = axes[0]
    # Use distinct colors for each parameter
    cmap = plt.cm.get_cmap("tab20", num_params)
    ax.stackplot(gens, pop_learn.T, labels=labels,
                 colors=[cmap(i) for i in range(num_params)], alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative Learnable Fraction")
    ax.set_title("Learnable Parameter Distribution (Run 0)")
    ax.legend(fontsize=7, loc="upper right", ncol=4)
    ax.grid(True, alpha=0.3)

    # Bottom: individual lines
    ax = axes[1]
    for i in range(num_params):
        ax.plot(gens, pop_learn[:, i], label=labels[i],
                color=cmap(i), alpha=0.8, linewidth=1.0)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="50%")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fraction Learnable")
    ax.set_title("Per-Parameter Learnable Fraction (Run 0)")
    ax.legend(fontsize=7, loc="upper right", ncol=4)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = viz_dir / "learnable_distribution.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path

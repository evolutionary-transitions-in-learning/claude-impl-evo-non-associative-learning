#!/usr/bin/env python3
"""
Analyze and visualize experiment results.

This script generates:
1. Figure 1 reproduction: 3D surface plot of evolution time
2. U-shaped curves for each clump-scale
3. Lifespan hypothesis comparison
4. Network architecture analysis
5. Summary statistics tables
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def load_results(experiment_dir: Path) -> pd.DataFrame:
    """Load all results from an experiment directory into a DataFrame."""
    results = []

    for subdir in ['main', 'lifespan_test']:
        result_dir = experiment_dir / subdir
        if result_dir.exists():
            for filepath in result_dir.glob("result_*.json"):
                with open(filepath) as f:
                    data = json.load(f)
                    data['experiment_type'] = subdir
                    results.append(data)

    if not results:
        raise ValueError(f"No results found in {experiment_dir}")

    return pd.DataFrame(results)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per condition."""
    # Group by condition
    grouped = df.groupby(['clump_scale', 'sensory_accuracy', 'lifespan', 'experiment_type'])

    summary = grouped.agg({
        'generations_to_success': ['mean', 'std', 'min', 'max', 'count'],
        'success': ['mean', 'sum'],  # success rate and count
        'final_mean_fitness': ['mean', 'std'],
        'runtime_seconds': ['mean', 'sum'],
    })

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    return summary


def plot_3d_surface(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Reproduce Figure 1: 3D surface plot of evolution time.

    X-axis: Sensory accuracy (50% to 100%)
    Y-axis: Clump scale (1 to 80)
    Z-axis: Generations to evolve cluster-tracking
    """
    # Filter to main experiment only
    main_df = df[df['experiment_type'] == 'main'].copy()

    # Compute mean generations per condition
    pivot_data = main_df.groupby(['clump_scale', 'sensory_accuracy'])['generations_to_success'].mean()
    pivot_df = pivot_data.reset_index()
    pivot_table = pivot_df.pivot(index='clump_scale', columns='sensory_accuracy', values='generations_to_success')

    # Create meshgrid for plotting
    accuracies = sorted(main_df['sensory_accuracy'].unique())
    clump_scales = sorted(main_df['clump_scale'].unique())

    X, Y = np.meshgrid(accuracies, clump_scales)
    Z = pivot_table.values

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    # Add wireframe for clarity
    ax.plot_wireframe(X, Y, Z, color='black', alpha=0.3, linewidth=0.5)

    # Labels matching paper
    ax.set_xlabel('Sensory Accuracy (%)', fontsize=12)
    ax.set_ylabel('Clump Scale', fontsize=12)
    ax.set_zlabel('Time to Evolve (Generations)', fontsize=12)

    # Format x-axis as percentage
    ax.set_xticks(accuracies)
    ax.set_xticklabels([f'{int(a*100)}%' for a in accuracies])

    ax.set_yticks(clump_scales)
    ax.set_zlim(0, 2000)

    # Set viewing angle similar to paper
    ax.view_init(elev=20, azim=225)

    plt.title('Evolution Time vs Sensory Accuracy and Clump Scale\n(Reproduction of Figure 1)', fontsize=14)

    fig.colorbar(surf, shrink=0.5, aspect=5, label='Generations')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D surface plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_ushape_curves(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Plot U-shaped curves for each clump-scale."""
    main_df = df[df['experiment_type'] == 'main'].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    clump_scales = sorted(main_df['clump_scale'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(clump_scales)))

    for clump_scale, color in zip(clump_scales, colors):
        subset = main_df[main_df['clump_scale'] == clump_scale]
        means = subset.groupby('sensory_accuracy')['generations_to_success'].mean()
        stds = subset.groupby('sensory_accuracy')['generations_to_success'].std()

        ax.errorbar(
            means.index * 100,  # Convert to percentage
            means.values,
            yerr=stds.values,
            label=f'Clump scale {clump_scale}',
            marker='o',
            color=color,
            capsize=3,
        )

    ax.set_xlabel('Sensory Accuracy (%)', fontsize=12)
    ax.set_ylabel('Generations to Evolve Cluster-Tracking', fontsize=12)
    ax.set_title('U-Shaped Curves: Evolution Time vs Accuracy for Each Clump Scale', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 2100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved U-shape curves to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_clumpscale_curves(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Plot evolution time vs clump-scale for each accuracy level."""
    main_df = df[df['experiment_type'] == 'main'].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = sorted(main_df['sensory_accuracy'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(accuracies)))

    for accuracy, color in zip(accuracies, colors):
        subset = main_df[main_df['sensory_accuracy'] == accuracy]
        means = subset.groupby('clump_scale')['generations_to_success'].mean()
        stds = subset.groupby('clump_scale')['generations_to_success'].std()

        ax.errorbar(
            means.index,
            means.values,
            yerr=stds.values,
            label=f'{int(accuracy*100)}% accuracy',
            marker='s',
            color=color,
            capsize=3,
        )

    ax.set_xlabel('Clump Scale', fontsize=12)
    ax.set_ylabel('Generations to Evolve Cluster-Tracking', fontsize=12)
    ax.set_title('Evolution Time vs Clump Scale for Each Accuracy Level', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 2100)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved clump-scale curves to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_lifespan_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Compare standard lifespan (1000) vs extended lifespan (4000)."""
    # Get main experiment data for clump-scale 80, accuracy 70%
    main_df = df[(df['experiment_type'] == 'main') &
                 (df['clump_scale'] == 80) &
                 (df['sensory_accuracy'] == 0.70)]

    # Get lifespan test data
    lifespan_df = df[df['experiment_type'] == 'lifespan_test']

    if len(main_df) == 0 or len(lifespan_df) == 0:
        print("Insufficient data for lifespan comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot comparison
    ax1 = axes[0]
    means = [main_df['generations_to_success'].mean(),
             lifespan_df['generations_to_success'].mean()]
    stds = [main_df['generations_to_success'].std(),
            lifespan_df['generations_to_success'].std()]

    bars = ax1.bar(['Lifespan 1000', 'Lifespan 4000'], means, yerr=stds,
                   capsize=5, color=['steelblue', 'coral'])
    ax1.set_ylabel('Generations to Evolve Cluster-Tracking', fontsize=12)
    ax1.set_title('Lifespan Hypothesis Test\n(Clump-scale 80, 70% accuracy)', fontsize=12)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=11)

    # Distribution comparison
    ax2 = axes[1]
    data = pd.DataFrame({
        'Generations': list(main_df['generations_to_success']) + list(lifespan_df['generations_to_success']),
        'Condition': ['Lifespan 1000'] * len(main_df) + ['Lifespan 4000'] * len(lifespan_df)
    })
    sns.boxplot(data=data, x='Condition', y='Generations', ax=ax2)
    ax2.set_ylabel('Generations to Evolve', fontsize=12)
    ax2.set_title('Distribution of Evolution Times', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved lifespan comparison to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_network_analysis(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Analyze evolved network architectures."""
    main_df = df[df['experiment_type'] == 'main'].copy()

    # Extract weight information
    weight_names = ['Forward (S→M)', 'Feedback (M→S)', 'Self-Sensory', 'Self-Motor']
    bias_names = ['Sensory Bias', 'Motor Bias']

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Weight distributions for successful runs
    ax = axes[0, 0]
    successful = main_df[main_df['success'] == True]
    if len(successful) > 0:
        weights_data = []
        for _, row in successful.iterrows():
            for i, name in enumerate(weight_names):
                weights_data.append({'Weight Type': name, 'Value': row['best_weights'][i]})
        weights_df = pd.DataFrame(weights_data)
        sns.boxplot(data=weights_df, x='Weight Type', y='Value', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Weight Distributions (Successful Runs)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 2. Bias distributions
    ax = axes[0, 1]
    if len(successful) > 0:
        bias_data = []
        for _, row in successful.iterrows():
            for i, name in enumerate(bias_names):
                bias_data.append({'Bias Type': name, 'Value': row['best_biases'][i]})
        bias_df = pd.DataFrame(bias_data)
        sns.boxplot(data=bias_df, x='Bias Type', y='Value', ax=ax)
        ax.set_title('Bias Distributions (Successful Runs)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 3. Forward weight vs accuracy
    ax = axes[0, 2]
    for cs in sorted(main_df['clump_scale'].unique()):
        subset = main_df[(main_df['clump_scale'] == cs) & (main_df['success'] == True)]
        if len(subset) > 0:
            means = subset.groupby('sensory_accuracy').apply(
                lambda x: np.mean([w[0] for w in x['best_weights']])
            )
            ax.plot(means.index * 100, means.values, marker='o', label=f'CS={cs}')
    ax.set_xlabel('Sensory Accuracy (%)')
    ax.set_ylabel('Forward Weight (S→M)')
    ax.set_title('Forward Weight vs Accuracy')
    ax.legend(fontsize=8)

    # 4. Self-recurrent weights heatmap
    ax = axes[1, 0]
    if len(successful) > 0:
        self_sensory = successful.groupby(['clump_scale', 'sensory_accuracy']).apply(
            lambda x: np.mean([w[2] for w in x['best_weights']])
        ).unstack()
        if not self_sensory.empty:
            sns.heatmap(self_sensory, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
            ax.set_title('Self-Sensory Weight by Condition')

    # 5. Learnable connections frequency
    ax = axes[1, 1]
    if len(successful) > 0:
        learnable_counts = np.zeros(6)
        for _, row in successful.iterrows():
            learnable_counts += np.array(row['best_learnable'])
        learnable_freq = learnable_counts / len(successful)
        names = weight_names + bias_names
        ax.bar(names, learnable_freq)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Fraction Learnable')
        ax.set_title('Learnable Connection Frequency')
        ax.set_ylim(0, 1)

    # 6. Success rate heatmap
    ax = axes[1, 2]
    success_rate = main_df.groupby(['clump_scale', 'sensory_accuracy'])['success'].mean().unstack()
    sns.heatmap(success_rate, annot=True, fmt='.0%', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
    ax.set_title('Success Rate by Condition')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved network analysis to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fitness_trajectories(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Plot sample fitness trajectories over generations."""
    main_df = df[df['experiment_type'] == 'main'].copy()

    # Select a few representative conditions
    conditions = [
        (10, 0.75),   # Moderate - expected sweet spot
        (80, 0.75),   # Large clump, moderate accuracy
        (80, 0.60),   # Large clump, low accuracy (anomaly region)
        (1, 0.75),    # No clumping
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (cs, acc) in zip(axes.flat, conditions):
        subset = main_df[(main_df['clump_scale'] == cs) &
                        (main_df['sensory_accuracy'] == acc)]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Clump={cs}, Acc={int(acc*100)}%')
            continue

        for _, row in subset.iterrows():
            gens = row['fitness_history_generations']
            fitness = row['fitness_history_mean']
            ax.plot(gens, fitness, alpha=0.5, linewidth=1)

        # Plot mean trajectory
        all_gens = set()
        for _, row in subset.iterrows():
            all_gens.update(row['fitness_history_generations'])
        all_gens = sorted(all_gens)

        mean_fitness = []
        for g in all_gens:
            values = []
            for _, row in subset.iterrows():
                if g in row['fitness_history_generations']:
                    idx = row['fitness_history_generations'].index(g)
                    values.append(row['fitness_history_mean'][idx])
            mean_fitness.append(np.mean(values) if values else np.nan)

        ax.plot(all_gens, mean_fitness, 'k-', linewidth=2, label='Mean')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Fitness')
        ax.set_title(f'Clump Scale={cs}, Accuracy={int(acc*100)}%')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved fitness trajectories to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_summary_table(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Generate summary statistics table (like paper's data)."""
    summary = compute_summary_statistics(df)

    # Filter to main experiment
    main_summary = summary[summary['experiment_type'] == 'main'].copy()

    # Create pivot table for easy reading
    pivot = main_summary.pivot_table(
        index='clump_scale',
        columns='sensory_accuracy',
        values='generations_to_success_mean',
        aggfunc='first'
    )

    print("\n" + "="*60)
    print("SUMMARY: Mean Generations to Evolve Cluster-Tracking")
    print("="*60)
    print(pivot.round(1).to_string())

    # Success rates
    success_pivot = main_summary.pivot_table(
        index='clump_scale',
        columns='sensory_accuracy',
        values='success_mean',
        aggfunc='first'
    )

    print("\n" + "="*60)
    print("SUMMARY: Success Rate (fraction reaching threshold)")
    print("="*60)
    print((success_pivot * 100).round(0).astype(int).astype(str).add('%').to_string())

    if output_path:
        summary.to_csv(output_path, index=False)
        print(f"\nFull summary saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "experiment_dir", type=str,
        help="Path to experiment directory containing results"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures (default: experiment_dir/figures)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display figures (just save)"
    )
    parser.add_argument(
        "--figure", type=str, default="all",
        choices=["all", "surface", "ushape", "clumpscale", "lifespan", "network", "trajectory", "summary"],
        help="Which figure to generate"
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {experiment_dir}")
    df = load_results(experiment_dir)
    print(f"Loaded {len(df)} results")

    show = not args.no_show

    if args.figure in ["all", "surface"]:
        plot_3d_surface(df, output_dir / "figure1_3d_surface.png", show)

    if args.figure in ["all", "ushape"]:
        plot_ushape_curves(df, output_dir / "ushape_curves.png", show)

    if args.figure in ["all", "clumpscale"]:
        plot_clumpscale_curves(df, output_dir / "clumpscale_curves.png", show)

    if args.figure in ["all", "lifespan"]:
        plot_lifespan_comparison(df, output_dir / "lifespan_comparison.png", show)

    if args.figure in ["all", "network"]:
        plot_network_analysis(df, output_dir / "network_analysis.png", show)

    if args.figure in ["all", "trajectory"]:
        plot_fitness_trajectories(df, output_dir / "fitness_trajectories.png", show)

    if args.figure in ["all", "summary"]:
        generate_summary_table(df, output_dir / "summary_statistics.csv")


if __name__ == "__main__":
    main()

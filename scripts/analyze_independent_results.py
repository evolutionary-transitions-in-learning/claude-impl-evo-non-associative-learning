#!/usr/bin/env python3
"""
Detailed analysis of independent-environment results.

Examines:
1. Raw vs smoothed mean fitness trajectories
2. U-shape detection across accuracy levels
3. Comparison with paper's known data points
4. Effect of smoothing window on convergence detection
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(experiment_dir: Path) -> list[dict]:
    results = []
    main_dir = experiment_dir / "main"
    for f in sorted(main_dir.glob("result_*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def compute_threshold(accuracy, ratio, lifespan=1000):
    optimal = lifespan / 2.0
    sensory_only = lifespan * (accuracy - 0.5)
    return sensory_only + ratio * (optimal - sensory_only)


def smoothed_convergence(mean_history, threshold, window=50):
    """Find first generation where SMOOTHED mean fitness >= threshold.

    Uses a rolling average to filter out noise from independent environments.
    """
    arr = np.array(mean_history)
    if len(arr) < window:
        return None
    # Rolling average
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='valid')
    # Offset: smoothed[i] corresponds to generation i + window//2
    offset = window // 2
    for i, val in enumerate(smoothed):
        if val >= threshold:
            return i + offset
    return None


def main():
    experiment_dir = Path("data/experiments/independent_env_subset")
    results = load_results(experiment_dir)
    print(f"Loaded {len(results)} results")

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[(r['clump_scale'], r['sensory_accuracy'])].append(r)

    output_dir = experiment_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Plot mean fitness trajectories for clump=80
    # ============================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    clump80_accs = [0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
    for i, acc in enumerate(clump80_accs):
        ax = axes[i]
        runs = by_condition.get((80, acc), [])
        sensory_only = 1000 * (acc - 0.5)
        optimal = 500.0

        for r in runs:
            gens = r['fitness_history_generations']
            mean_fit = r['fitness_history_mean']
            ax.plot(gens, mean_fit, alpha=0.3, linewidth=0.5, color='blue')

            # Smoothed
            if len(mean_fit) > 50:
                kernel = np.ones(50) / 50
                smoothed = np.convolve(mean_fit, kernel, mode='valid')
                ax.plot(range(25, 25 + len(smoothed)), smoothed, alpha=0.7, linewidth=1.5, color='red')

        ax.axhline(y=sensory_only, color='green', linestyle='--', label=f'sensory_only={sensory_only:.0f}')
        ax.axhline(y=optimal, color='orange', linestyle='--', label=f'optimal={optimal:.0f}')

        # Show thresholds for various ratios
        for ratio in [0.05, 0.10, 0.20]:
            thresh = compute_threshold(acc, ratio)
            ax.axhline(y=thresh, color='purple', linestyle=':', alpha=0.5)

        ax.set_title(f'Clump=80, Acc={acc:.0%}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean Fitness')
        ax.set_ylim(-200, 600)
        if i == 0:
            ax.legend(fontsize=7)

    axes[-1].set_visible(False)
    plt.suptitle('Mean Fitness Trajectories (clump=80, independent envs)\nBlue=raw, Red=smoothed(50)', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "trajectories_clump80.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved trajectories_clump80.png")

    # ============================================================
    # 2. Smoothed convergence analysis
    # ============================================================
    print("\n" + "="*70)
    print("SMOOTHED CONVERGENCE ANALYSIS")
    print("="*70)

    windows = [20, 50, 100, 200]
    ratios = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    # Paper reference points
    paper_points = {
        (80, 0.70): 187,
        (20, 0.70): 61,
    }

    for window in windows:
        print(f"\n--- Smoothing window = {window} generations ---")
        for ratio in ratios:
            print(f"\n  Ratio = {ratio:.2f}:")
            print(f"  {'Clump':>6}", end="")
            for acc in clump80_accs:
                print(f"  {int(acc*100):>5}%", end="")
            print()

            for cs in [10, 20, 80]:
                print(f"  {cs:>6}", end="")
                for acc in clump80_accs:
                    runs = by_condition.get((cs, acc), [])
                    if not runs:
                        print(f"  {'--':>6}", end="")
                        continue
                    conv_gens = []
                    for r in runs:
                        threshold = compute_threshold(acc, ratio, r['lifespan'])
                        cg = smoothed_convergence(r['fitness_history_mean'], threshold, window)
                        conv_gens.append(cg if cg is not None else 2000)
                    mean_cg = np.mean(conv_gens)
                    paper = paper_points.get((cs, acc))
                    marker = " *" if paper else ""
                    print(f"  {mean_cg:>6.0f}{marker}", end="")
                print()

    # ============================================================
    # 3. Best matching configuration
    # ============================================================
    print("\n" + "="*70)
    print("SEARCHING FOR BEST MATCH TO PAPER DATA")
    print("="*70)
    print("Paper: clump=80/acc=70% -> 187 gens, clump=20/acc=70% -> 61 gens")

    best_score = float('inf')
    best_config = None

    for window in [10, 20, 30, 50, 75, 100, 150, 200]:
        for ratio in np.arange(0.005, 0.30, 0.005):
            # clump=80, acc=70%
            runs_80 = by_condition.get((80, 0.70), [])
            thresh_80 = compute_threshold(0.70, ratio)
            gens_80 = []
            for r in runs_80:
                cg = smoothed_convergence(r['fitness_history_mean'], thresh_80, window)
                gens_80.append(cg if cg is not None else 2000)
            mean_80 = np.mean(gens_80) if gens_80 else 2000

            # clump=20, acc=70%
            runs_20 = by_condition.get((20, 0.70), [])
            thresh_20 = compute_threshold(0.70, ratio)
            gens_20 = []
            for r in runs_20:
                cg = smoothed_convergence(r['fitness_history_mean'], thresh_20, window)
                gens_20.append(cg if cg is not None else 2000)
            mean_20 = np.mean(gens_20) if gens_20 else 2000

            # Score: weighted relative error
            if mean_80 < 2000 and mean_20 < 2000:
                err_80 = abs(mean_80 - 187) / 187
                err_20 = abs(mean_20 - 61) / 61
                score = err_80 + err_20

                if score < best_score:
                    best_score = score
                    best_config = (window, ratio, mean_80, mean_20, score)

    if best_config:
        w, r, m80, m20, sc = best_config
        print(f"\nBest match: window={w}, ratio={r:.3f}")
        print(f"  clump=80/acc=70%: {m80:.0f} gens (paper: 187)")
        print(f"  clump=20/acc=70%: {m20:.0f} gens (paper: 61)")
        print(f"  Score (sum relative errors): {sc:.3f}")
    else:
        print("\nNo configuration found where both conditions converge!")

    # ============================================================
    # 4. U-shape analysis for clump=80
    # ============================================================
    print("\n" + "="*70)
    print("U-SHAPE ANALYSIS FOR CLUMP=80")
    print("="*70)

    # Use the best config if found, otherwise defaults
    if best_config:
        use_window, use_ratio = best_config[0], best_config[1]
    else:
        use_window, use_ratio = 50, 0.05

    print(f"Using window={use_window}, ratio={use_ratio:.3f}")
    print(f"\n{'Acc':>6} {'MeanGens':>10} {'StdGens':>10} {'Pattern':>20}")
    print("-" * 50)

    u_gens = []
    for acc in clump80_accs:
        runs = by_condition.get((80, acc), [])
        if not runs:
            continue
        threshold = compute_threshold(acc, use_ratio)
        conv_gens = []
        for r in runs:
            cg = smoothed_convergence(r['fitness_history_mean'], threshold, use_window)
            conv_gens.append(cg if cg is not None else 2000)
        mean_cg = np.mean(conv_gens)
        std_cg = np.std(conv_gens)
        u_gens.append((acc, mean_cg, std_cg))

        pattern = "SLOW" if mean_cg >= 1500 else "MEDIUM" if mean_cg >= 500 else "FAST"
        print(f"{acc:>6.0%} {mean_cg:>10.0f} {std_cg:>10.0f} {pattern:>20}")

    # Check U-shape: does it decrease then increase?
    if len(u_gens) >= 3:
        gens_only = [g for _, g, _ in u_gens]
        min_idx = np.argmin(gens_only)
        has_u_shape = min_idx > 0 and min_idx < len(gens_only) - 1
        if has_u_shape:
            min_acc = u_gens[min_idx][0]
            print(f"\nU-SHAPE DETECTED! Minimum at accuracy = {min_acc:.0%}")
            print("Left side (slow → fast):", " → ".join(f"{a:.0%}:{g:.0f}" for a, g, _ in u_gens[:min_idx+1]))
            print("Right side (fast → slow):", " → ".join(f"{a:.0%}:{g:.0f}" for a, g, _ in u_gens[min_idx:]))
        else:
            print(f"\nNo clear U-shape. Min at index {min_idx} of {len(gens_only)}")

    # ============================================================
    # 5. Plot U-shape curve
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    for acc, mean_cg, std_cg in u_gens:
        ax.errorbar(acc, mean_cg, yerr=std_cg, fmt='o-', color='blue', markersize=8, capsize=5)

    ax.set_xlabel('Sensory Accuracy', fontsize=12)
    ax.set_ylabel('Mean Generations to Converge', fontsize=12)
    ax.set_title(f'Clump=80: Generations to Converge (Independent Envs)\nSmoothing={use_window}, Ratio={use_ratio:.3f}', fontsize=14)
    ax.set_ylim(0, 2200)
    ax.axhline(y=2000, color='red', linestyle='--', alpha=0.5, label='Max (no convergence)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "u_shape_clump80.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved u_shape_clump80.png")

    # ============================================================
    # 6. Compare clump scales at acc=70%
    # ============================================================
    print("\n" + "="*70)
    print("CLUMP SCALE COMPARISON AT ACC=70%")
    print("="*70)
    for cs in [10, 20, 80]:
        runs = by_condition.get((cs, 0.70), [])
        if not runs:
            continue
        threshold = compute_threshold(0.70, use_ratio)
        conv_gens = []
        for r in runs:
            cg = smoothed_convergence(r['fitness_history_mean'], threshold, use_window)
            conv_gens.append(cg if cg is not None else 2000)
        mean_cg = np.mean(conv_gens)
        print(f"  clump={cs:>3}: {mean_cg:.0f} gens (paper: {paper_points.get((cs, 0.70), '?')})")


if __name__ == "__main__":
    main()

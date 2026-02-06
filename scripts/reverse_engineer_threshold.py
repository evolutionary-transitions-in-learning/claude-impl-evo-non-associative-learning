#!/usr/bin/env python3
"""
Reverse-engineer the paper's success threshold by testing multiple hypotheses
against known data points from the paper.

Known calibration points from Todd & Miller:
  1. clump=80, acc=70%, lifespan=1000 → 187 generations
  2. clump=80, acc=70%, lifespan=4000 → 52 generations
  3. clump=20, acc=70%, lifespan=1000 → 61 generations

Qualitative constraints:
  - U-shaped curves for each clump-scale (slow at low AND high accuracy)
  - Decreasing evolution time as clump-scale increases (for moderate accuracies)
  - Anomaly: high clump + low accuracy = unexpectedly slow
  - Diagonal "ravine" of fastest evolution
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ============================================================================
# Threshold Hypotheses
# ============================================================================

def sensory_only(accuracy, lifespan=1000):
    """Expected fitness with sensory-only strategy."""
    return lifespan * (accuracy - 0.5)

def optimal(lifespan=1000):
    """Theoretical max fitness (eat all food, avoid all poison)."""
    return lifespan / 2.0

def random_baseline():
    """Expected fitness with random eating (50/50)."""
    return 0.0

def worst_possible(lifespan=1000):
    """Worst possible fitness (eat all poison, avoid all food)."""
    return -lifespan / 2.0


HYPOTHESES = {}

# Hypothesis A: Fixed absolute threshold (constant for all conditions)
# threshold = C
def threshold_fixed(accuracy, lifespan, param):
    return param
HYPOTHESES['A_fixed'] = {
    'name': 'Fixed absolute threshold',
    'formula': 'threshold = C',
    'fn': threshold_fixed,
    'param_range': np.arange(50, 451, 10),  # 50 to 450 in steps of 10
    'param_label': 'C',
}

# Hypothesis B: Fraction of optimal fitness
# threshold = ratio * optimal
def threshold_frac_optimal(accuracy, lifespan, param):
    return param * optimal(lifespan)
HYPOTHESES['B_frac_optimal'] = {
    'name': 'Fraction of optimal',
    'formula': 'threshold = ratio × optimal',
    'fn': threshold_frac_optimal,
    'param_range': np.arange(0.1, 0.91, 0.05),
    'param_label': 'ratio',
}

# Hypothesis C: Interpolation between sensory_only and optimal (our current formula)
# threshold = sensory_only + ratio * (optimal - sensory_only)
def threshold_interp(accuracy, lifespan, param):
    so = sensory_only(accuracy, lifespan)
    opt = optimal(lifespan)
    return so + param * (opt - so)
HYPOTHESES['C_interp'] = {
    'name': 'Interpolation (sensory_only → optimal)',
    'formula': 'threshold = sensory_only + ratio × (optimal - sensory_only)',
    'fn': threshold_interp,
    'param_range': np.arange(0.1, 0.91, 0.05),
    'param_label': 'ratio',
}

# Hypothesis D: Fixed margin above sensory_only
# threshold = sensory_only + margin
def threshold_margin(accuracy, lifespan, param):
    return sensory_only(accuracy, lifespan) + param
HYPOTHESES['D_margin'] = {
    'name': 'Fixed margin above sensory-only',
    'formula': 'threshold = sensory_only + margin',
    'fn': threshold_margin,
    'param_range': np.arange(10, 251, 10),  # 10 to 250
    'param_label': 'margin',
}

# Hypothesis E: Multiplicative improvement over sensory_only
# threshold = sensory_only * (1 + ratio)
def threshold_mult(accuracy, lifespan, param):
    so = sensory_only(accuracy, lifespan)
    if so <= 0:
        return so + param * 100  # fallback for near-zero sensory_only
    return so * (1.0 + param)
HYPOTHESES['E_multiplicative'] = {
    'name': 'Multiplicative improvement over sensory-only',
    'formula': 'threshold = sensory_only × (1 + ratio)',
    'fn': threshold_mult,
    'param_range': np.arange(0.1, 2.01, 0.1),
    'param_label': 'ratio',
}

# Hypothesis F: Fraction of full range (worst to optimal)
# threshold = -optimal + ratio * (2 * optimal)
def threshold_full_range(accuracy, lifespan, param):
    opt = optimal(lifespan)
    return -opt + param * 2.0 * opt
HYPOTHESES['F_full_range'] = {
    'name': 'Fraction of full range (-optimal to optimal)',
    'formula': 'threshold = -optimal + ratio × 2 × optimal',
    'fn': threshold_full_range,
    'param_range': np.arange(0.5, 0.96, 0.025),
    'param_label': 'ratio',
}

# Hypothesis G: Fixed threshold per lifespan (proportional to lifespan)
# threshold = ratio * lifespan
def threshold_per_lifespan(accuracy, lifespan, param):
    return param * lifespan
HYPOTHESES['G_per_lifespan'] = {
    'name': 'Proportional to lifespan',
    'formula': 'threshold = ratio × lifespan',
    'fn': threshold_per_lifespan,
    'param_range': np.arange(0.05, 0.46, 0.025),
    'param_label': 'ratio',
}


# ============================================================================
# Data Loading and Analysis
# ============================================================================

def load_results(experiment_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load main and lifespan test results."""
    main_results = []
    main_dir = experiment_dir / "main"
    for f in sorted(main_dir.glob("result_*.json")):
        with open(f) as fh:
            main_results.append(json.load(fh))

    lifespan_results = []
    life_dir = experiment_dir / "lifespan_test"
    if life_dir.exists():
        for f in sorted(life_dir.glob("result_*.json")):
            with open(f) as fh:
                lifespan_results.append(json.load(fh))

    return main_results, lifespan_results


def find_convergence_gen(mean_history: list[float], gen_indices: list[int], threshold: float) -> int:
    """Find first generation where mean fitness >= threshold."""
    for i, gen in enumerate(gen_indices):
        if mean_history[i] >= threshold:
            return gen
    return 2000  # max_generations


def compute_mean_gens_for_condition(
    results: list[dict],
    clump_scale: int,
    accuracy: float,
    lifespan: int,
    threshold_fn,
    param: float,
) -> float:
    """Compute mean generations to convergence for a specific condition."""
    matching = [r for r in results
                if r['clump_scale'] == clump_scale
                and abs(r['sensory_accuracy'] - accuracy) < 0.001
                and r['lifespan'] == lifespan]

    if not matching:
        return float('nan')

    threshold = threshold_fn(accuracy, lifespan, param)
    gens = []
    for r in matching:
        gen = find_convergence_gen(r['fitness_history_mean'], r['fitness_history_generations'], threshold)
        gens.append(gen)

    return np.mean(gens)


# ============================================================================
# Calibration Against Paper's Data Points
# ============================================================================

# Paper's known values
CALIBRATION_POINTS = [
    {'clump_scale': 80, 'accuracy': 0.70, 'lifespan': 1000, 'target': 187, 'source': 'main'},
    {'clump_scale': 20, 'accuracy': 0.70, 'lifespan': 1000, 'target': 61, 'source': 'main'},
    {'clump_scale': 80, 'accuracy': 0.70, 'lifespan': 4000, 'target': 52, 'source': 'lifespan'},
]


def evaluate_hypothesis(
    hypothesis_key: str,
    main_results: list[dict],
    lifespan_results: list[dict],
) -> dict:
    """Evaluate a hypothesis across its parameter range."""
    hyp = HYPOTHESES[hypothesis_key]
    fn = hyp['fn']

    best_error = float('inf')
    best_param = None
    best_predictions = None

    all_scores = []

    for param in hyp['param_range']:
        predictions = []
        for cp in CALIBRATION_POINTS:
            results = main_results if cp['source'] == 'main' else lifespan_results
            pred = compute_mean_gens_for_condition(
                results, cp['clump_scale'], cp['accuracy'], cp['lifespan'], fn, param
            )
            predictions.append(pred)

        targets = [cp['target'] for cp in CALIBRATION_POINTS]

        # Compute error (relative RMSE to handle different scales)
        errors = [(p - t) / max(t, 1) for p, t in zip(predictions, targets)]
        rmse = np.sqrt(np.mean([e**2 for e in errors]))

        all_scores.append((param, rmse, predictions))

        if rmse < best_error:
            best_error = rmse
            best_param = param
            best_predictions = predictions

    return {
        'key': hypothesis_key,
        'name': hyp['name'],
        'formula': hyp['formula'],
        'best_param': best_param,
        'best_error': best_error,
        'best_predictions': best_predictions,
        'all_scores': all_scores,
    }


def compute_ushape_score(
    main_results: list[dict],
    threshold_fn,
    param: float,
    clump_scales: list[int],
    accuracies: list[float],
) -> float:
    """Score how U-shaped the curves are (higher = more U-shaped).

    For each clump_scale that has convergence at multiple accuracies,
    check if there's a minimum at intermediate accuracy with higher values
    at both ends.
    """
    score = 0.0
    count = 0

    for cs in clump_scales:
        gens = []
        for acc in accuracies:
            g = compute_mean_gens_for_condition(main_results, cs, acc, 1000, threshold_fn, param)
            gens.append(g)

        gens = np.array(gens)

        # Skip if all are 2000 (no convergence)
        if np.all(gens >= 2000):
            continue

        # Find the minimum and its position
        min_idx = np.argmin(gens)
        min_val = gens[min_idx]

        # Check if there are higher values on BOTH sides
        if min_idx > 0 and min_idx < len(gens) - 1:
            left_max = np.max(gens[:min_idx])
            right_max = np.max(gens[min_idx+1:])

            # U-shape score: how much higher are the sides vs the minimum?
            if min_val > 0:
                left_ratio = left_max / min_val
                right_ratio = right_max / min_val
                # Both sides should be elevated
                u_score = min(left_ratio, right_ratio)
                if u_score > 1.0:
                    score += u_score - 1.0
                    count += 1

    return score / max(count, 1)


def compute_anomaly_score(
    main_results: list[dict],
    threshold_fn,
    param: float,
) -> float:
    """Score the presence of the high-clump + low-accuracy anomaly.

    The paper shows that clump=80 + low accuracy is SLOWER than clump=80 + moderate accuracy.
    Also slower than clump=40 at same low accuracy.
    """
    # Check: clump=80, acc=55% should be SLOWER than clump=80, acc=70%
    g80_55 = compute_mean_gens_for_condition(main_results, 80, 0.55, 1000, threshold_fn, param)
    g80_70 = compute_mean_gens_for_condition(main_results, 80, 0.70, 1000, threshold_fn, param)
    g80_80 = compute_mean_gens_for_condition(main_results, 80, 0.80, 1000, threshold_fn, param)

    # Check: clump=40 at low acc should be faster than clump=80 at low acc (anomaly)
    g40_55 = compute_mean_gens_for_condition(main_results, 40, 0.55, 1000, threshold_fn, param)

    score = 0.0
    # Anomaly: g80_55 > g80_70 (high clump + low acc is slower than high clump + moderate)
    if g80_55 > g80_70 and g80_70 < 2000:
        score += (g80_55 - g80_70) / max(g80_70, 1)
    # Anomaly: g80_55 > g40_55 (clump=80 slower than clump=40 at low accuracy)
    if g80_55 > g40_55 and g40_55 < 2000:
        score += (g80_55 - g40_55) / max(g40_55, 1)

    return score


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    experiment_dir = Path("data/experiments/paper_reproduction")
    main_results, lifespan_results = load_results(experiment_dir)
    print(f"Loaded {len(main_results)} main + {len(lifespan_results)} lifespan results")

    clump_scales = sorted(set(r['clump_scale'] for r in main_results))
    accuracies = sorted(set(r['sensory_accuracy'] for r in main_results))

    print("\n" + "=" * 80)
    print("CALIBRATION TARGETS FROM PAPER")
    print("=" * 80)
    for cp in CALIBRATION_POINTS:
        print(f"  clump={cp['clump_scale']}, acc={cp['accuracy']:.0%}, "
              f"life={cp['lifespan']} → {cp['target']} generations")

    # Evaluate each hypothesis
    print("\n" + "=" * 80)
    print("HYPOTHESIS EVALUATION")
    print("=" * 80)

    all_results = []
    for key in HYPOTHESES:
        result = evaluate_hypothesis(key, main_results, lifespan_results)
        all_results.append(result)

        print(f"\n{'─'*60}")
        print(f"Hypothesis {result['key']}: {result['name']}")
        print(f"  Formula: {result['formula']}")
        print(f"  Best {HYPOTHESES[key]['param_label']} = {result['best_param']:.4g}")
        print(f"  Relative RMSE = {result['best_error']:.3f}")
        print(f"  Predictions vs targets:")
        for cp, pred in zip(CALIBRATION_POINTS, result['best_predictions']):
            diff = pred - cp['target']
            print(f"    clump={cp['clump_scale']}, acc={cp['accuracy']:.0%}, "
                  f"life={cp['lifespan']}: {pred:.0f} (target={cp['target']}, diff={diff:+.0f})")

    # Rank by calibration error
    all_results.sort(key=lambda r: r['best_error'])

    print("\n" + "=" * 80)
    print("RANKING BY CALIBRATION FIT (relative RMSE)")
    print("=" * 80)
    for i, r in enumerate(all_results):
        print(f"  {i+1}. [{r['key']}] {r['name']} "
              f"({HYPOTHESES[r['key']]['param_label']}={r['best_param']:.4g}) "
              f"→ RMSE={r['best_error']:.3f}")

    # For top 3 hypotheses, also compute U-shape and anomaly scores
    print("\n" + "=" * 80)
    print("QUALITATIVE ANALYSIS (top 3 by calibration fit)")
    print("=" * 80)

    for r in all_results[:3]:
        key = r['key']
        fn = HYPOTHESES[key]['fn']
        param = r['best_param']

        ushape = compute_ushape_score(main_results, fn, param, clump_scales, accuracies)
        anomaly = compute_anomaly_score(main_results, fn, param)

        print(f"\n{'─'*60}")
        print(f"Hypothesis {key}: {r['name']} ({HYPOTHESES[key]['param_label']}={param:.4g})")
        print(f"  U-shape score: {ushape:.3f} (higher = more U-shaped)")
        print(f"  Anomaly score: {anomaly:.3f} (higher = more anomaly)")

        # Print full grid
        print(f"\n  Mean generations to convergence:")
        print(f"  {'Clump':>6}", end="")
        for acc in accuracies:
            print(f"  {int(acc*100):>5}%", end="")
        print()
        for cs in clump_scales:
            print(f"  {cs:>6}", end="")
            for acc in accuracies:
                g = compute_mean_gens_for_condition(main_results, cs, acc, 1000, fn, param)
                print(f"  {g:>6.0f}", end="")
            print()

    # Generate heatmaps for top 3
    output_dir = experiment_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, r in zip(axes, all_results[:3]):
        key = r['key']
        fn = HYPOTHESES[key]['fn']
        param = r['best_param']

        grid = np.zeros((len(clump_scales), len(accuracies)))
        for i, cs in enumerate(clump_scales):
            for j, acc in enumerate(accuracies):
                grid[i, j] = compute_mean_gens_for_condition(main_results, cs, acc, 1000, fn, param)

        sns.heatmap(
            grid, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=[f'{int(a*100)}%' for a in accuracies],
            yticklabels=[str(cs) for cs in clump_scales],
            ax=ax, vmin=0, vmax=2000,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Mean Generations'},
        )
        param_label = HYPOTHESES[key]['param_label']
        ax.set_xlabel('Sensory Accuracy')
        ax.set_ylabel('Clump Scale')
        ax.set_title(f'{r["name"]}\n{param_label}={param:.3g}, RMSE={r["best_error"]:.3f}')

    plt.suptitle('Top 3 Threshold Hypotheses by Calibration Fit', fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "threshold_hypotheses_top3.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved top-3 comparison to {output_dir / 'threshold_hypotheses_top3.png'}")

    # Comprehensive scoring: combine calibration fit, U-shape, and anomaly
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SCORING (all hypotheses)")
    print("=" * 80)
    print(f"{'Hypothesis':<25} {'Param':>8} {'CalibRMSE':>10} {'U-shape':>8} {'Anomaly':>8} {'Combined':>9}")
    print("─" * 80)

    combined_results = []
    for r in all_results:
        key = r['key']
        fn = HYPOTHESES[key]['fn']
        param = r['best_param']
        ushape = compute_ushape_score(main_results, fn, param, clump_scales, accuracies)
        anomaly = compute_anomaly_score(main_results, fn, param)

        # Combined score: low calibration error + high U-shape + high anomaly
        # Normalize: lower RMSE is better, higher U-shape and anomaly are better
        combined = r['best_error'] - 0.5 * ushape - 0.3 * anomaly

        combined_results.append((r, ushape, anomaly, combined))
        param_label = HYPOTHESES[key]['param_label']
        print(f"  {key:<23} {param:>8.3g} {r['best_error']:>10.3f} {ushape:>8.3f} {anomaly:>8.3f} {combined:>9.3f}")

    combined_results.sort(key=lambda x: x[3])
    print(f"\n  Best overall: {combined_results[0][0]['key']} "
          f"({combined_results[0][0]['name']})")


if __name__ == "__main__":
    main()

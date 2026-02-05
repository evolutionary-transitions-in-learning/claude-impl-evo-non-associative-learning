"""Cross-run comparison and sensitivity analysis visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import ALL_PARAM_LABELS, DPI


def _find_best_genotype(experiment_dir, summary):
    """Find the best genotype across all runs in an experiment.

    Uses summary.json to identify the run with the highest final_best_fitness,
    then loads that run's best_genotype.npy.

    Returns:
        (genotype_path, best_run_index) or (None, None) if not found.
    """
    data_dir = experiment_dir / "data"
    runs = summary.get("runs", [])

    if not runs:
        # Fallback: just use run_000
        path = data_dir / "run_000" / "best_genotype.npy"
        return (path, 0) if path.exists() else (None, None)

    best_run_idx = max(range(len(runs)), key=lambda i: runs[i].get("final_best_fitness", 0))
    path = data_dir / f"run_{best_run_idx:03d}" / "best_genotype.npy"
    return (path, best_run_idx) if path.exists() else (None, None)


def plot_run_comparison(summary, viz_dir, **kwargs) -> Path | None:
    """Plot comparison table/heatmap across all runs."""
    runs = summary.get("runs", [])
    if not runs:
        return None

    num_runs = len(runs)
    metrics = ["final_mean_fitness", "final_best_fitness", "final_std_fitness",
               "final_survivors", "generations_run"]
    metric_labels = ["Mean Fitness", "Best Fitness", "Fitness Std", "Survivors", "Generations"]

    # Build data matrix
    data = np.zeros((num_runs, len(metrics)))
    for i, run in enumerate(runs):
        for j, metric in enumerate(metrics):
            data[i, j] = run.get(metric, 0)

    fig, ax = plt.subplots(figsize=(10, max(3, num_runs * 0.6 + 2)))

    # Normalize each column for coloring
    data_norm = np.zeros_like(data)
    for j in range(len(metrics)):
        col = data[:, j]
        col_range = col.max() - col.min()
        if col_range > 0:
            data_norm[:, j] = (col - col.min()) / col_range
        else:
            data_norm[:, j] = 0.5

    im = ax.imshow(data_norm, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    # Add text annotations
    for i in range(num_runs):
        for j in range(len(metrics)):
            val = data[i, j]
            fmt = ".2f" if j < 3 else ".0f"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=9)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=30, ha="right")
    ax.set_yticks(range(num_runs))
    ax.set_yticklabels([f"Run {i}" for i in range(num_runs)])
    ax.set_title("Run Comparison Summary")

    # Add aggregate row
    agg = summary.get("aggregate", {})
    if agg:
        agg_text = (
            f"Aggregate: mean_fitness={agg.get('mean_of_mean_fitness', 0):.2f} "
            f"(std={agg.get('std_of_mean_fitness', 0):.2f}), "
            f"best={agg.get('max_of_best_fitness', 0):.2f}, "
            f"survivors={agg.get('mean_survivors', 0):.1f}"
        )
        fig.text(0.5, 0.01, agg_text, ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    path = viz_dir / "run_comparison.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_fitness_sensitivity(experiment_dir, summary, viz_dir, **kwargs) -> Path | None:
    """Plot fitness sensitivity analysis for the best genotype.

    For binary genotypes: flip each bit and measure fitness change.
    For continuous genotypes: perturb each gene by +/- epsilon and measure fitness change.
    Uses the best agent across all runs. Computed at visualization time (requires JAX).
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return None

    genotype_path, best_run = _find_best_genotype(experiment_dir, summary)
    if genotype_path is None:
        return None

    genotype = jnp.array(np.load(genotype_path))

    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        return None

    from habituation_experiment_alice.config import GenotypeMode, SimulationConfig
    from habituation_experiment_alice.evaluation import evaluate_agent_detailed

    config = SimulationConfig.from_yaml(config_path)
    is_continuous = config.genetic.genotype_mode == GenotypeMode.CONTINUOUS

    # Evaluate original
    key = jax.random.PRNGKey(42)
    original_trace = evaluate_agent_detailed(genotype, key, config)
    original_fitness = float(original_trace.fitness)

    geno_len = len(genotype)
    deltas = np.zeros(geno_len)

    if is_continuous:
        # Continuous: perturb each gene by small epsilon
        epsilon = 0.1
        for idx in range(geno_len):
            mutant_plus = genotype.at[idx].set(genotype[idx] + epsilon)
            mutant_minus = genotype.at[idx].set(genotype[idx] - epsilon)
            fit_plus = float(evaluate_agent_detailed(mutant_plus, key, config).fitness)
            fit_minus = float(evaluate_agent_detailed(mutant_minus, key, config).fitness)
            deltas[idx] = (fit_plus - fit_minus) / (2 * epsilon)
    else:
        # Binary: flip each bit
        for bit_idx in range(geno_len):
            mutant = genotype.at[bit_idx].set(1 - genotype[bit_idx])
            mutant_trace = evaluate_agent_detailed(mutant, key, config)
            deltas[bit_idx] = float(mutant_trace.fitness) - original_fitness

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Top: bar chart of all sensitivities
    ax = axes[0]
    colors = np.where(deltas >= 0, "mediumseagreen", "coral")
    ax.bar(range(geno_len), deltas, color=colors, edgecolor="none", alpha=0.8)
    xlabel = "Gene Index" if is_continuous else "Bit Index"
    ylabel = "dFitness/dGene" if is_continuous else "Fitness Change"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    mode_label = "continuous" if is_continuous else "binary"
    ax.set_title(f"Fitness Sensitivity ({mode_label}, best agent run {best_run}, fitness={original_fitness:.2f})")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    if not is_continuous:
        # Add connection/bias region annotations for binary genotypes
        from habituation_experiment_alice.genetics import BITS_PER_BIAS, BITS_PER_CONNECTION
        num_conns = config.network.num_connections
        num_biases = config.network.num_biases

        offset = 0
        for i in range(num_conns):
            if i % 2 == 0:
                ax.axvspan(offset, offset + BITS_PER_CONNECTION, alpha=0.05, color="blue")
            offset += BITS_PER_CONNECTION
        for i in range(num_biases):
            if i % 2 == 0:
                ax.axvspan(offset, offset + BITS_PER_BIAS, alpha=0.05, color="orange")
            offset += BITS_PER_BIAS
    else:
        # Add region annotations for continuous genotypes
        num_conns = config.network.num_connections
        num_biases = config.network.num_biases
        num_params = config.network.num_params
        ax.axvspan(0, num_conns, alpha=0.05, color="blue", label="weights")
        ax.axvspan(num_conns, num_conns + num_biases, alpha=0.05, color="orange", label="biases")
        ax.axvspan(num_conns + num_biases, num_conns + num_biases + num_params,
                   alpha=0.05, color="green", label="learnable")

    # Bottom: sorted absolute sensitivity
    ax = axes[1]
    sorted_idx = np.argsort(np.abs(deltas))[::-1]
    ax.bar(range(geno_len), np.abs(deltas[sorted_idx]), color="steelblue", alpha=0.7)
    ax.set_xlabel("Rank (most sensitive first)")
    ax.set_ylabel(f"|{ylabel}|")
    ax.set_title("Sorted Absolute Sensitivity")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = viz_dir / "fitness_sensitivity.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def _run_ablation_analysis(experiment_dir, summary):
    """Run ablation analysis: evaluate original + each parameter zeroed out.

    Uses the best agent across all runs. Supports both binary and continuous genotypes.

    Returns None if data is unavailable, otherwise returns:
        (original_trace, ablated_traces, config, best_run) where ablated_traces
        is a list of AgentTrace objects (one per parameter: 9 connections + 3 biases).
    """
    import jax
    import jax.numpy as jnp

    genotype_path, best_run = _find_best_genotype(experiment_dir, summary)
    if genotype_path is None:
        return None

    genotype = jnp.array(np.load(genotype_path))

    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        return None

    from habituation_experiment_alice.config import GenotypeMode, SimulationConfig
    from habituation_experiment_alice.evaluation import evaluate_agent_detailed
    from habituation_experiment_alice.genetics import BITS_PER_BIAS, BITS_PER_CONNECTION

    config = SimulationConfig.from_yaml(config_path)
    num_conns = config.network.num_connections
    num_biases = config.network.num_biases

    key = jax.random.PRNGKey(42)
    original_trace = evaluate_agent_detailed(genotype, key, config)

    ablated_traces = []

    if config.genetic.genotype_mode == GenotypeMode.CONTINUOUS:
        # Continuous mode: zero out each weight directly, then each bias
        for i in range(num_conns):
            mutant = genotype.at[i].set(0.0)
            ablated_traces.append(evaluate_agent_detailed(mutant, key, config))
        for i in range(num_biases):
            mutant = genotype.at[num_conns + i].set(0.0)
            ablated_traces.append(evaluate_agent_detailed(mutant, key, config))
    else:
        # Binary mode: zero presence bit for connections, zero all bits for biases
        for i in range(num_conns):
            mutant = genotype.at[i * BITS_PER_CONNECTION].set(0)
            ablated_traces.append(evaluate_agent_detailed(mutant, key, config))
        for i in range(num_biases):
            mutant = genotype.copy()
            bit_start = num_conns * BITS_PER_CONNECTION + i * BITS_PER_BIAS
            for b in range(BITS_PER_BIAS):
                mutant = mutant.at[bit_start + b].set(0)
            ablated_traces.append(evaluate_agent_detailed(mutant, key, config))

    return original_trace, ablated_traces, config, best_run


def plot_connection_ablation(experiment_dir, summary, viz_dir, **kwargs) -> Path | None:
    """Ablation analysis: zero each connection/bias and measure phase 2 fitness impact.

    For the best evolved agent, systematically removes each of the 12 parameters
    (9 connections + 3 biases) and re-evaluates to identify which connections
    are critical for threat discrimination.
    """
    try:
        import jax  # noqa: F401
    except ImportError:
        return None

    result = _run_ablation_analysis(experiment_dir, summary)
    if result is None:
        return None

    original_trace, ablated_traces, config, best_run = result
    num_conns = config.network.num_connections
    num_biases = config.network.num_biases
    num_params = num_conns + num_biases

    original_fitness = float(original_trace.fitness)
    original_survived = bool(original_trace.survived_phase1)

    ablated_fitness = np.array([float(t.fitness) for t in ablated_traces])
    ablated_survived = np.array([bool(t.survived_phase1) for t in ablated_traces])

    fitness_drop = original_fitness - ablated_fitness
    labels = ALL_PARAM_LABELS[:num_params]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    max_drop = max(np.max(np.abs(fitness_drop)), 1e-6)
    colors = []
    for drop in fitness_drop:
        if drop > 0.1 * max_drop:
            colors.append("coral")
        elif drop < -0.1 * max_drop:
            colors.append("mediumseagreen")
        else:
            colors.append("steelblue")

    x = np.arange(num_params)
    ax.bar(x, ablated_fitness, color=colors, edgecolor="none", alpha=0.85, width=0.7)

    ax.axhline(y=original_fitness, color="black", linestyle="--", linewidth=1.5,
               label=f"Original ({original_fitness:.2f})")

    for i in range(num_params):
        if not ablated_survived[i]:
            ax.text(i, ablated_fitness[i] + 0.02 * max(original_fitness, 0.1), "X",
                    ha="center", va="bottom", fontsize=10, color="red", fontweight="bold")

    ax.axvline(x=num_conns - 0.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(num_conns / 2 - 0.5, ax.get_ylim()[1] * 0.95, "Connections",
            ha="center", fontsize=9, color="gray")
    ax.text(num_conns + num_biases / 2 - 0.5, ax.get_ylim()[1] * 0.95, "Biases",
            ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Phase 2 Fitness")
    ax.set_title(f"Connection Ablation Analysis (Best Agent, Run {best_run})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    survived_label = "survived P1" if original_survived else "died P1"
    fig.text(0.5, 0.01,
             f"Original: {survived_label} | "
             "Red X = died in phase 1 | "
             "Coral = large drop | Green = improved | Blue = minimal change",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    path = viz_dir / "connection_ablation.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def _compute_discrimination(trace):
    """Compute discrimination accuracy metrics from a trace's phase 2 data."""
    outputs = np.asarray(trace.phase2_outputs)
    true_threat = np.asarray(trace.phase2_true_threat)
    threat_present = np.asarray(trace.phase2_threat_present)
    alive = np.asarray(trace.phase2_alive)

    false_threat = threat_present * (1 - true_threat)
    no_threat = 1 - threat_present
    alive_mask = alive > 0.5

    tt_mask = (true_threat > 0.5) & alive_mask
    tt_correct = np.sum((outputs > 0) & tt_mask) / max(np.sum(tt_mask), 1)

    ft_mask = (false_threat > 0.5) & alive_mask
    ft_correct = np.sum((outputs < 0) & ft_mask) / max(np.sum(ft_mask), 1)

    nt_mask = (no_threat > 0.5) & alive_mask
    nt_correct = np.sum((outputs < 0) & nt_mask) / max(np.sum(nt_mask), 1)

    return float(tt_correct), float(ft_correct), float(nt_correct)


def plot_ablation_discrimination(experiment_dir, summary, viz_dir, **kwargs) -> Path | None:
    """Discrimination accuracy for each ablation variant vs the original.

    Shows a grouped bar chart with true-threat withdrawal accuracy,
    false-threat eating accuracy, and no-threat eating accuracy for the
    original agent and each ablated variant.
    """
    try:
        import jax  # noqa: F401
    except ImportError:
        return None

    result = _run_ablation_analysis(experiment_dir, summary)
    if result is None:
        return None

    original_trace, ablated_traces, config, best_run = result
    num_conns = config.network.num_connections
    num_biases = config.network.num_biases
    num_params = num_conns + num_biases

    # Compute discrimination for original + all ablated
    all_labels = ["Original"] + list(ALL_PARAM_LABELS[:num_params])
    all_traces = [original_trace] + ablated_traces
    n = len(all_labels)

    tt_acc = np.zeros(n)
    ft_acc = np.zeros(n)
    nt_acc = np.zeros(n)
    survived = np.zeros(n, dtype=bool)

    for i, trace in enumerate(all_traces):
        tt_acc[i], ft_acc[i], nt_acc[i] = _compute_discrimination(trace)
        survived[i] = bool(trace.survived_phase1)

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n)
    width = 0.25

    bars_tt = ax.bar(x - width, tt_acc, width, label="True Threat (withdraw)",
                     color="coral", alpha=0.85, edgecolor="none")
    bars_ft = ax.bar(x, ft_acc, width, label="False Threat (eat)",
                     color="gold", alpha=0.85, edgecolor="none")
    bars_nt = ax.bar(x + width, nt_acc, width, label="No Threat (eat)",
                     color="mediumseagreen", alpha=0.85, edgecolor="none")

    # Highlight original with a background band
    ax.axvspan(-0.5, 0.5, alpha=0.08, color="blue")

    # Mark variants that died in phase 1
    for i in range(n):
        if not survived[i]:
            ax.text(i, -0.06, "X", ha="center", va="top",
                    fontsize=9, color="red", fontweight="bold")

    # Divider between connections and biases (offset by 1 for "Original")
    ax.axvline(x=num_conns + 0.5, color="gray", linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Correct Response Rate")
    ax.set_ylim(-0.1, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(f"Ablation Discrimination Accuracy (Best Agent, Run {best_run})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.text(0.5, 0.01,
             "Red X = died in phase 1 | Blue band = original agent",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    path = viz_dir / "ablation_discrimination.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path

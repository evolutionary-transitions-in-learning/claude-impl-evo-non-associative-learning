"""Agent behavioral trace visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .common import (
    ALL_PARAM_LABELS,
    COLORS,
    CONNECTION_LABELS,
    DPI,
    get_final_trace,
)


def _plot_single_phase_trace(trace, phase_prefix, ax_env, ax_signals, ax_health, title_suffix):
    """Plot a single phase's trace data on three axes."""
    outputs = trace[f"{phase_prefix}_outputs"]
    health = trace[f"{phase_prefix}_health"]
    alive = trace[f"{phase_prefix}_alive"]
    stimulus = trace[f"{phase_prefix}_stimulus"]
    pain = trace[f"{phase_prefix}_pain"]
    true_threat = trace[f"{phase_prefix}_true_threat"]
    threat_present = trace[f"{phase_prefix}_threat_present"]

    T = len(outputs)
    t = np.arange(T)

    # False threat = threat present but not true threat
    false_threat = threat_present * (1 - true_threat)

    # Top: Environment
    ax = ax_env
    # Shade threat regions
    ax.fill_between(t, 0, 1, where=true_threat > 0.5,
                    alpha=0.3, color=COLORS["true_threat"], label="True threat",
                    step="mid")
    ax.fill_between(t, 0, 1, where=false_threat > 0.5,
                    alpha=0.3, color=COLORS["false_threat"], label="False threat",
                    step="mid")
    ax.set_ylabel("Threat")
    ax.set_title(f"Environment {title_suffix}")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["None", "Present"])
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, T)

    # Middle: Signals
    ax = ax_signals
    ax.plot(t, stimulus, color=COLORS["stimulus"], alpha=0.7, label="Stimulus", linewidth=0.8)
    ax.plot(t, pain, color=COLORS["pain"], alpha=0.7, label="Pain", linewidth=0.8)
    ax.plot(t, outputs, color=COLORS["output"], alpha=0.9, label="Output", linewidth=1.0)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_ylabel("Signal")
    ax.set_title(f"Network Signals {title_suffix}")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, T)

    # Bottom: Health
    ax = ax_health
    ax.plot(t, health, color=COLORS["health"], linewidth=1.0)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Death threshold")
    # Shade dead region
    dead_mask = alive < 0.5
    if np.any(dead_mask):
        ax.fill_between(t, 0, health.max() * 1.1, where=dead_mask,
                        alpha=0.1, color="red", step="mid")
    ax.set_ylabel("Health")
    ax.set_xlabel("Timestep")
    ax.set_title(f"Health Trajectory {title_suffix}")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, T)


def plot_best_agent_trace(all_traces, summary, viz_dir, **kwargs) -> list[Path] | None:
    """Plot best agent behavior traces for both phases.

    Creates two images: best_agent_trace_phase1.png and best_agent_trace_phase2.png
    Uses the final generation's trace from run 0.
    """
    trace = get_final_trace(all_traces)
    if trace is None:
        return None

    paths = []

    for phase_prefix, phase_label in [("phase1", "Phase 1"), ("phase2", "Phase 2")]:
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        fitness = float(trace["fitness"])
        survived = bool(trace["survived_phase1"])
        title_suffix = f"({phase_label}, fitness={fitness:.2f}, survived_p1={survived})"

        _plot_single_phase_trace(trace, phase_prefix, axes[0], axes[1], axes[2], title_suffix)

        plt.tight_layout()
        path = viz_dir / f"best_agent_trace_{phase_prefix}.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close()
        paths.append(path)

    return paths


def plot_discrimination_accuracy(all_traces, summary, viz_dir, **kwargs) -> Path | None:
    """Plot discrimination accuracy: correct responses to true vs false threats.

    For the best agent in phase 2:
    - True threats: fraction where output > 0 (withdrawal)
    - False threats: fraction where output < 0 (eating)
    - No threat: fraction where output < 0 (eating)
    """
    trace = get_final_trace(all_traces)
    if trace is None:
        return None

    outputs = trace["phase2_outputs"]
    true_threat = trace["phase2_true_threat"]
    threat_present = trace["phase2_threat_present"]
    alive = trace["phase2_alive"]

    false_threat = threat_present * (1 - true_threat)
    no_threat = 1 - threat_present

    # Only count alive timesteps
    alive_mask = alive > 0.5

    # True threat: correct = withdrawal (output > 0)
    tt_mask = (true_threat > 0.5) & alive_mask
    tt_correct = np.sum((outputs > 0) & tt_mask) / max(np.sum(tt_mask), 1)

    # False threat: correct = eating (output < 0)
    ft_mask = (false_threat > 0.5) & alive_mask
    ft_correct = np.sum((outputs < 0) & ft_mask) / max(np.sum(ft_mask), 1)

    # No threat: correct = eating (output < 0)
    nt_mask = (no_threat > 0.5) & alive_mask
    nt_correct = np.sum((outputs < 0) & nt_mask) / max(np.sum(nt_mask), 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["True Threat\n(withdraw)", "False Threat\n(eat)", "No Threat\n(eat)"]
    values = [float(tt_correct), float(ft_correct), float(nt_correct)]
    colors = [COLORS["withdrawal"], COLORS["false_threat"], COLORS["eating"]]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Correct Response Rate")
    ax.set_title("Phase 2 Discrimination Accuracy (Best Agent)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = viz_dir / "discrimination_accuracy.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_response_distribution(all_traces, summary, viz_dir, **kwargs) -> Path | None:
    """Plot histogram of output values by threat context in phase 2."""
    trace = get_final_trace(all_traces)
    if trace is None:
        return None

    outputs = trace["phase2_outputs"]
    true_threat = trace["phase2_true_threat"]
    threat_present = trace["phase2_threat_present"]
    alive = trace["phase2_alive"]

    alive_mask = alive > 0.5
    false_threat = threat_present * (1 - true_threat)
    no_threat = 1 - threat_present

    tt_outputs = outputs[(true_threat > 0.5) & alive_mask]
    ft_outputs = outputs[(false_threat > 0.5) & alive_mask]
    nt_outputs = outputs[(no_threat > 0.5) & alive_mask]

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(-1, 1, 41)

    if len(tt_outputs) > 0:
        ax.hist(tt_outputs, bins=bins, alpha=0.5, color=COLORS["true_threat"],
                label=f"True threat (n={len(tt_outputs)})", density=True)
    if len(ft_outputs) > 0:
        ax.hist(ft_outputs, bins=bins, alpha=0.5, color=COLORS["false_threat"],
                label=f"False threat (n={len(ft_outputs)})", density=True)
    if len(nt_outputs) > 0:
        ax.hist(nt_outputs, bins=bins, alpha=0.5, color=COLORS["no_threat"],
                label=f"No threat (n={len(nt_outputs)})", density=True)

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Output Value (-1=eat, +1=withdraw)")
    ax.set_ylabel("Density")
    ax.set_title("Phase 2 Output Distribution by Context (Best Agent)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = viz_dir / "response_distribution.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_health_dynamics_breakdown(all_traces, summary, viz_dir, **kwargs) -> Path | None:
    """Plot decomposition of health changes in phase 2.

    Shows eating gain, passive decay, and threat damage contributions.
    """
    trace = get_final_trace(all_traces)
    if trace is None:
        return None

    outputs = trace["phase2_outputs"]
    true_threat = trace["phase2_true_threat"]
    health = trace["phase2_health"]
    alive = trace["phase2_alive"]

    cfg = summary["config_summary"]
    passive_decay = cfg.get("passive_decay", 0.1)
    eating_gain_rate = cfg.get("eating_gain_rate", 1.0)
    threat_damage = cfg.get("threat_damage", 5.0)

    T = len(outputs)
    t = np.arange(T)

    # Compute per-timestep components
    eating_amount = np.maximum(-outputs, 0.0)
    eating_gain = eating_amount * eating_gain_rate

    protection = np.maximum(outputs, 0.0)
    damage = true_threat * (1.0 - protection) * threat_damage

    decay = np.full(T, passive_decay)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: stacked components
    ax = axes[0]
    ax.fill_between(t, 0, eating_gain, alpha=0.5, color=COLORS["eating"], label="Eating gain")
    ax.fill_between(t, 0, -decay, alpha=0.5, color=COLORS["neutral"], label="Passive decay")
    ax.fill_between(t, -decay, -decay - damage, alpha=0.5, color=COLORS["true_threat"], label="Threat damage")
    ax.plot(t, eating_gain - decay - damage, color="black", linewidth=0.8, alpha=0.7, label="Net delta")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_ylabel("Health Delta / Timestep")
    ax.set_title("Health Dynamics Breakdown (Phase 2, Best Agent)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: cumulative health
    ax = axes[1]
    ax.plot(t, health, color=COLORS["health"], linewidth=1.0)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # Shade threat regions
    false_threat = trace["phase2_threat_present"] * (1 - true_threat)
    ax.fill_between(t, health.min() - 1, health.max() + 1, where=true_threat > 0.5,
                    alpha=0.1, color=COLORS["true_threat"], step="mid")
    ax.fill_between(t, health.min() - 1, health.max() + 1, where=false_threat > 0.5,
                    alpha=0.1, color=COLORS["false_threat"], step="mid")

    ax.set_ylabel("Health")
    ax.set_xlabel("Timestep")
    ax.set_title("Health Trajectory with Threat Regions")
    ax.set_ylim(bottom=min(0, health.min()) - 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = viz_dir / "health_dynamics_breakdown.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_learning_dynamics(all_traces, summary, viz_dir, **kwargs) -> Path | None:
    """Plot weight changes from Hebbian learning (initial vs after each phase)."""
    trace = get_final_trace(all_traces)
    if trace is None:
        return None

    w_init = trace["weights_initial"]
    b_init = trace["biases_initial"]
    w_after_p1 = trace["weights_after_phase1"]
    b_after_p1 = trace["biases_after_phase1"]
    w_after_p2 = trace["weights_after_phase2"]
    b_after_p2 = trace["biases_after_phase2"]
    learnable = trace["learnable_mask"]

    num_conns = len(w_init)
    num_biases = len(b_init)

    # Combine weights and biases
    init_all = np.concatenate([w_init, b_init])
    after_p1_all = np.concatenate([w_after_p1, b_after_p1])
    after_p2_all = np.concatenate([w_after_p2, b_after_p2])

    labels = ALL_PARAM_LABELS[:len(init_all)]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, init_all, width, label="Initial (genetic)",
                   color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, after_p1_all, width, label="After Phase 1",
                   color="coral", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, after_p2_all, width, label="After Phase 2",
                   color="mediumseagreen", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Mark learnable parameters
    for i, is_learnable in enumerate(learnable):
        if is_learnable:
            ax.plot(i, max(abs(init_all[i]), abs(after_p1_all[i]), abs(after_p2_all[i])) + 0.3,
                    "*", color="gold", markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Weight Value")
    ax.set_title("Learning Dynamics: Weight Changes (Best Agent, Final Gen)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    # Add legend note for stars
    gold_star = mpatches.Patch(color="gold", label="Learnable (star)")
    handles, labels_list = ax.get_legend_handles_labels()
    handles.append(gold_star)
    ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    path = viz_dir / "learning_dynamics.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path

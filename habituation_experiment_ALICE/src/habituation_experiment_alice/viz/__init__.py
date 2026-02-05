"""Visualization package for ALICE experiment results.

Provides modular plot functions organized by category:
- fitness_plots: Population fitness metrics over generations
- agent_trace_plots: Individual agent behavior traces
- genetics_plots: Genotype and network parameter evolution
- analysis_plots: Cross-run comparisons and sensitivity analysis

Each plot function has the signature:
    plot_xxx(data, viz_dir, **kwargs) -> Path | None
    Returns the path to the saved image, or None if data was insufficient.
"""

from pathlib import Path

from .fitness_plots import (
    plot_fitness_by_phase,
    plot_fitness_distribution,
    plot_fitness_std,
    plot_fitness_summary,
    plot_survival_rate,
)
from .agent_trace_plots import (
    plot_best_agent_trace,
    plot_discrimination_accuracy,
    plot_health_dynamics_breakdown,
    plot_learning_dynamics,
    plot_response_distribution,
)
from .genetics_plots import (
    plot_genotype_diversity,
    plot_learnable_distribution,
    plot_weight_evolution_heatmap,
)
from .analysis_plots import (
    plot_ablation_discrimination,
    plot_connection_ablation,
    plot_fitness_sensitivity,
    plot_run_comparison,
)


# Registry of all plot functions with their required data sources
PLOT_REGISTRY = {
    # Fitness plots (need histories + summary)
    "fitness_by_phase": plot_fitness_by_phase,
    "fitness_summary": plot_fitness_summary,
    "fitness_distribution": plot_fitness_distribution,
    "fitness_std": plot_fitness_std,
    "survival_rate": plot_survival_rate,
    # Agent trace plots (need traces)
    "best_agent_trace": plot_best_agent_trace,
    "discrimination_accuracy": plot_discrimination_accuracy,
    "response_distribution": plot_response_distribution,
    "health_dynamics_breakdown": plot_health_dynamics_breakdown,
    "learning_dynamics": plot_learning_dynamics,
    # Genetics plots (need histories)
    "genotype_diversity": plot_genotype_diversity,
    "weight_evolution_heatmap": plot_weight_evolution_heatmap,
    "learnable_distribution": plot_learnable_distribution,
    # Analysis plots (need summary + experiment_dir)
    "fitness_sensitivity": plot_fitness_sensitivity,
    "connection_ablation": plot_connection_ablation,
    "ablation_discrimination": plot_ablation_discrimination,
    "run_comparison": plot_run_comparison,
}


def generate_all_visualizations(
    experiment_dir: Path,
    summary: dict,
    histories: list[dict],
    all_traces: list[dict],
    verbose: bool = True,
) -> list[Path]:
    """Generate all visualizations for an experiment.

    Args:
        experiment_dir: Path to experiment directory
        summary: Experiment summary dict
        histories: List of run history dicts
        all_traces: List of run trace dicts
        verbose: Print progress

    Returns:
        List of paths to generated images
    """
    viz_dir = experiment_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    generated = []

    # Fitness plots
    for name, fn in [
        ("fitness_by_phase", plot_fitness_by_phase),
        ("fitness_summary", plot_fitness_summary),
        ("fitness_distribution", plot_fitness_distribution),
        ("fitness_std", plot_fitness_std),
        ("survival_rate", plot_survival_rate),
    ]:
        try:
            result = fn(histories, summary, viz_dir, experiment_dir=experiment_dir)
            if result:
                generated.append(result)
                if verbose:
                    print(f"  {result.name}")
        except Exception as e:
            if verbose:
                print(f"  SKIP {name}: {e}")

    # Agent trace plots
    for name, fn in [
        ("best_agent_trace", plot_best_agent_trace),
        ("discrimination_accuracy", plot_discrimination_accuracy),
        ("response_distribution", plot_response_distribution),
        ("health_dynamics_breakdown", plot_health_dynamics_breakdown),
        ("learning_dynamics", plot_learning_dynamics),
    ]:
        try:
            result = fn(all_traces, summary, viz_dir)
            if result:
                if isinstance(result, list):
                    generated.extend(result)
                    if verbose:
                        for r in result:
                            print(f"  {r.name}")
                else:
                    generated.append(result)
                    if verbose:
                        print(f"  {result.name}")
        except Exception as e:
            if verbose:
                print(f"  SKIP {name}: {e}")

    # Genetics plots
    for name, fn in [
        ("genotype_diversity", plot_genotype_diversity),
        ("weight_evolution_heatmap", plot_weight_evolution_heatmap),
        ("learnable_distribution", plot_learnable_distribution),
    ]:
        try:
            result = fn(histories, summary, viz_dir)
            if result:
                generated.append(result)
                if verbose:
                    print(f"  {result.name}")
        except Exception as e:
            if verbose:
                print(f"  SKIP {name}: {e}")

    # Analysis plots
    try:
        result = plot_run_comparison(summary, viz_dir)
        if result:
            generated.append(result)
            if verbose:
                print(f"  {result.name}")
    except Exception as e:
        if verbose:
            print(f"  SKIP run_comparison: {e}")

    try:
        result = plot_fitness_sensitivity(
            experiment_dir, summary, viz_dir
        )
        if result:
            generated.append(result)
            if verbose:
                print(f"  {result.name}")
    except Exception as e:
        if verbose:
            print(f"  SKIP fitness_sensitivity: {e}")

    for name, fn in [
        ("connection_ablation", plot_connection_ablation),
        ("ablation_discrimination", plot_ablation_discrimination),
    ]:
        try:
            result = fn(experiment_dir, summary, viz_dir)
            if result:
                generated.append(result)
                if verbose:
                    print(f"  {result.name}")
        except Exception as e:
            if verbose:
                print(f"  SKIP {name}: {e}")

    return generated

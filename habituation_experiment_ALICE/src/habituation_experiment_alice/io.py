"""Data I/O for saving and loading experiment results.

Experiment folder structure:
    runs/<experiment_name>/
        config.yaml              # config used for this experiment
        data/
            run_000/
                history.npz      # per-generation stats (core + phase + genetics)
                best_genotype.npy
                final_fitness.npy
                traces/          # checkpoint agent traces
                    gen_0000.npz
                    gen_0100.npz
                    ...
            run_001/
                ...
            summary.json         # aggregate across all runs
        viz/                     # populated by gen_all_viz.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .simulation import SimulationResult


def create_experiment_dir(
    base_dir: Path,
    name: str | None = None,
    config_path: str | None = None,
) -> Path:
    """Create a structured experiment directory under runs/.

    Args:
        base_dir: Project root (where runs/ will be created)
        name: Experiment name. If None, derived from config filename or timestamp.
        config_path: Path to the config file used (for deriving name).

    Returns:
        Path to the experiment directory (runs/<name>/)
    """
    runs_dir = base_dir / "runs"

    if name is None:
        if config_path:
            name = Path(config_path).stem
        else:
            name = "default"

    # Add timestamp to make unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{name}_{timestamp}"

    experiment_dir = runs_dir / experiment_name
    (experiment_dir / "data").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "viz").mkdir(parents=True, exist_ok=True)

    return experiment_dir


def _save_agent_trace(path: Path, trace) -> None:
    """Save an AgentTrace to a .npz file."""
    np.savez(
        path,
        phase1_outputs=np.array(trace.phase1_outputs),
        phase1_health=np.array(trace.phase1_health),
        phase1_alive=np.array(trace.phase1_alive),
        phase1_stimulus=np.array(trace.phase1_stimulus),
        phase1_pain=np.array(trace.phase1_pain),
        phase1_true_threat=np.array(trace.phase1_true_threat),
        phase1_threat_present=np.array(trace.phase1_threat_present),
        phase2_outputs=np.array(trace.phase2_outputs),
        phase2_health=np.array(trace.phase2_health),
        phase2_alive=np.array(trace.phase2_alive),
        phase2_stimulus=np.array(trace.phase2_stimulus),
        phase2_pain=np.array(trace.phase2_pain),
        phase2_true_threat=np.array(trace.phase2_true_threat),
        phase2_threat_present=np.array(trace.phase2_threat_present),
        weights_initial=np.array(trace.weights_initial),
        biases_initial=np.array(trace.biases_initial),
        weights_after_phase1=np.array(trace.weights_after_phase1),
        biases_after_phase1=np.array(trace.biases_after_phase1),
        weights_after_phase2=np.array(trace.weights_after_phase2),
        biases_after_phase2=np.array(trace.biases_after_phase2),
        learnable_mask=np.array(trace.learnable_mask),
        time_constants_initial=np.array(trace.time_constants_initial),
        time_constants_after_phase1=np.array(trace.time_constants_after_phase1),
        time_constants_after_phase2=np.array(trace.time_constants_after_phase2),
        fitness=np.array(float(trace.fitness)),
        survived_phase1=np.array(bool(trace.survived_phase1)),
    )


def save_run_data(
    experiment_dir: Path,
    run_index: int,
    result: SimulationResult,
) -> Path:
    """Save data from a single simulation run.

    Args:
        experiment_dir: Experiment directory (runs/<name>/)
        run_index: Index of this run (0, 1, 2, ...)
        result: SimulationResult from the run

    Returns:
        Path to the run data directory
    """
    run_dir = experiment_dir / "data" / f"run_{run_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    h = result.history

    # Save generation-level history (all metrics)
    np.savez(
        run_dir / "history.npz",
        # Core metrics
        mean_fitness=np.array(h.mean_fitness),
        best_fitness=np.array(h.best_fitness),
        min_fitness=np.array(h.min_fitness),
        std_fitness=np.array(h.std_fitness),
        num_survivors=np.array(h.num_survivors),
        # Phase-specific metrics
        phase1_mean_survival_frac=np.array(h.phase1_mean_survival_frac),
        phase2_mean_health=np.array(h.phase2_mean_health),
        num_survived_phase1=np.array(h.num_survived_phase1),
        # Genetics metrics
        genotype_diversity=np.array(h.genotype_diversity),
        best_weights=np.array(h.best_weights),
        best_biases=np.array(h.best_biases),
        best_learnable_mask=np.array(h.best_learnable_mask),
        pop_learnable_frac=np.array(h.pop_learnable_frac),
    )

    # Save best genotype from final generation
    np.save(run_dir / "best_genotype.npy", np.array(result.final_state.best_genotype))

    # Save final population fitness scores
    np.save(run_dir / "final_fitness.npy", np.array(result.final_state.fitness_scores))

    # Save checkpoint traces
    if result.traces:
        traces_dir = run_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        for gen, trace in result.traces.items():
            _save_agent_trace(traces_dir / f"gen_{gen:04d}.npz", trace)

    return run_dir


def save_experiment_summary(
    experiment_dir: Path,
    results: list[SimulationResult],
    config: SimulationConfig,
) -> None:
    """Save experiment config and aggregate summary.

    Args:
        experiment_dir: Experiment directory
        results: List of SimulationResults from all runs
        config: Configuration used
    """
    # Save config
    config.save_yaml(experiment_dir / "config.yaml")

    # Build summary
    summary = {
        "num_runs": len(results),
        "config_summary": {
            "population_size": config.genetic.population_size,
            "max_generations": config.genetic.max_generations,
            "phase1_lifetime": config.environment.phase1_lifetime,
            "phase2_lifetime": config.environment.phase2_lifetime,
            "clump_scale": config.environment.clump_scale,
            "true_false_ratio": config.environment.true_false_ratio,
            "starting_health": config.health.starting_health,
            "threat_damage": config.health.threat_damage,
            "pain_delay": config.pain.delay,
            "seed": config.seed,
        },
        "runs": [],
    }

    for i, result in enumerate(results):
        run_info = {
            "run_index": i,
            "final_mean_fitness": float(result.final_state.mean_fitness),
            "final_best_fitness": float(result.final_state.best_fitness),
            "final_min_fitness": float(np.min(np.array(result.final_state.fitness_scores))),
            "final_std_fitness": float(np.std(np.array(result.final_state.fitness_scores))),
            "final_survivors": int(result.final_state.num_survivors),
            "generations_run": int(result.final_state.generation),
        }
        summary["runs"].append(run_info)

    # Aggregate stats
    mean_fitnesses = [r["final_mean_fitness"] for r in summary["runs"]]
    best_fitnesses = [r["final_best_fitness"] for r in summary["runs"]]
    survivor_counts = [r["final_survivors"] for r in summary["runs"]]

    summary["aggregate"] = {
        "mean_of_mean_fitness": float(np.mean(mean_fitnesses)),
        "std_of_mean_fitness": float(np.std(mean_fitnesses)),
        "mean_of_best_fitness": float(np.mean(best_fitnesses)),
        "max_of_best_fitness": float(np.max(best_fitnesses)),
        "mean_survivors": float(np.mean(survivor_counts)),
    }

    with open(experiment_dir / "data" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# Loading
# ============================================================================


def load_run_history(run_dir: Path) -> dict:
    """Load history data from a single run directory.

    Args:
        run_dir: Path to run_NNN/ directory

    Returns:
        Dict with arrays for all tracked metrics
    """
    data = np.load(run_dir / "history.npz")
    return dict(data)


def load_agent_trace(path: Path) -> dict:
    """Load an agent trace from a .npz file."""
    return dict(np.load(path, allow_pickle=True))


def load_run_traces(run_dir: Path) -> dict[int, dict]:
    """Load all checkpoint traces from a run directory.

    Returns:
        Dict mapping generation number to trace data dict
    """
    traces_dir = run_dir / "traces"
    if not traces_dir.exists():
        return {}
    traces = {}
    for trace_file in sorted(traces_dir.glob("gen_*.npz")):
        gen = int(trace_file.stem.split("_")[1])
        traces[gen] = load_agent_trace(trace_file)
    return traces


def load_experiment(experiment_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """Load all data from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory (runs/<name>/)

    Returns:
        Tuple of (summary_dict, list_of_run_history_dicts, list_of_run_traces_dicts)
    """
    # Load summary
    with open(experiment_dir / "data" / "summary.json") as f:
        summary = json.load(f)

    # Load all run histories and traces
    data_dir = experiment_dir / "data"
    run_dirs = sorted(data_dir.glob("run_*"))
    histories = [load_run_history(d) for d in run_dirs]
    all_traces = [load_run_traces(d) for d in run_dirs]

    return summary, histories, all_traces

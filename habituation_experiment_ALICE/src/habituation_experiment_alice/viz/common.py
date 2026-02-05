"""Common constants and helpers for visualization."""

import numpy as np

# Connection labels for N=1 (3 neurons: S=Stimulus, P=Pain, O=Output)
CONNECTION_LABELS = [
    "S->O", "P->O",     # forward
    "O->S", "O->P",     # recurrent
    "S->S", "P->P", "O->O",  # self-recurrent
    "S->P", "P->S",     # cross-input
]

BIAS_LABELS = ["B_S", "B_P", "B_O"]

ALL_PARAM_LABELS = CONNECTION_LABELS + BIAS_LABELS

# Color scheme
COLORS = {
    "true_threat": "#d62728",   # red
    "false_threat": "#1f77b4",  # blue
    "no_threat": "#2ca02c",     # green
    "stimulus": "#ff7f0e",      # orange
    "pain": "#9467bd",          # purple
    "output": "#17becf",        # cyan
    "health": "#e377c2",        # pink
    "fitness_mean": "steelblue",
    "fitness_best": "coral",
    "withdrawal": "#d62728",
    "eating": "#2ca02c",
    "neutral": "#7f7f7f",
}

DPI = 150


def get_final_trace(all_traces: list[dict]) -> dict | None:
    """Get the trace from the highest generation of run 0."""
    if not all_traces or not all_traces[0]:
        return None
    run0_traces = all_traces[0]
    if not run0_traces:
        return None
    max_gen = max(run0_traces.keys())
    return run0_traces[max_gen]


def get_trace_at_gen(all_traces: list[dict], gen: int, run: int = 0) -> dict | None:
    """Get trace for a specific generation and run."""
    if run >= len(all_traces) or not all_traces[run]:
        return None
    return all_traces[run].get(gen)


def safe_divide(a, b, default=0.0):
    """Divide with zero-safety."""
    return np.where(b > 0, a / b, default)

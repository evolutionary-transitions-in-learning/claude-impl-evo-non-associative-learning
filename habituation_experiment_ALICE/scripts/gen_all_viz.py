"""Generate all visualizations from experiment data.

Usage:
    python scripts/gen_all_viz.py runs/<experiment_name>/

Reads from data/ subfolder and writes all plots to viz/ subfolder.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from habituation_experiment_alice.io import load_experiment
from habituation_experiment_alice.viz import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description="Generate all visualizations from ALICE experiment data"
    )
    parser.add_argument(
        "experiment_dir", type=str,
        help="Path to experiment directory (e.g., runs/default_20240101_120000/)"
    )

    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: {experiment_dir} does not exist")
        sys.exit(1)

    summary_file = experiment_dir / "data" / "summary.json"
    if not summary_file.exists():
        print(f"Error: No summary.json found in {experiment_dir / 'data'}")
        sys.exit(1)

    print(f"Loading experiment data from {experiment_dir}")
    summary, histories, all_traces = load_experiment(experiment_dir)
    print(f"  Found {len(histories)} runs")

    num_traces = sum(len(t) for t in all_traces)
    print(f"  Found {num_traces} checkpoint traces")

    print("\nGenerating visualizations...")
    generated = generate_all_visualizations(
        experiment_dir, summary, histories, all_traces, verbose=True
    )

    print(f"\n{len(generated)} visualizations saved to {experiment_dir / 'viz'}")


if __name__ == "__main__":
    main()

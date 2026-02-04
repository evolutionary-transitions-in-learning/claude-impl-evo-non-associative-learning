# Reproduction Plan: Todd & Miller Figure 1 and Results

This document outlines the plan to reproduce the results from "Exploring Adaptive Agency III: Simulating the Evolution of Habituation and Sensitization" by Todd & Miller.

## Results to Reproduce

### 1. Main Result: Figure 1 (3D Surface Plot)

**What it shows:**
- 3D surface of evolution time as a function of sensory accuracy and clump-scale
- X-axis: Sensory accuracy (50% → 100%)
- Y-axis: Clump scale (1 → 80)
- Z-axis: Generations to evolve cluster-tracking (0 → 2000)

**Expected patterns:**
1. **U-shaped curves** for each clump-scale as a function of accuracy
2. **Decreasing evolution time** as clump-scale increases (for high accuracies)
3. **Diagonal "ravine"** of fastest evolution (not symmetric)
4. **Anomaly**: High clump-scales + low accuracies = unexpectedly slow evolution

### 2. Lifespan Hypothesis Test

**Hypothesis:** The anomaly (slow evolution with large clumps + low accuracy) is due to high variance in food/poison ratios when clumps are large relative to lifespan.

**Test:** Run with clump-scale 80, 70% accuracy, but 4000 timesteps instead of 1000.

**Expected result:** Evolution time should decrease from ~187 to ~52 generations.

### 3. Network Architecture Analysis

**Questions:**
- What connection weights enable cluster-tracking?
- Do self-recurrent connections evolve to create memory?
- How do evolved architectures differ across conditions?

---

## Experimental Parameters

### Main Experiment Grid

| Parameter | Values |
|-----------|--------|
| Clump scales | 1, 5, 10, 20, 40, 80 |
| Sensory accuracies | 55%, 60%, 70%, 75%, 80%, 90%, 95% |
| Total conditions | 42 (6 × 7) |
| Runs per condition | 5-10 (paper used 5 minimum, 10 for ~75% of cases) |
| Population size | 100 |
| Max generations | 2000 |
| Creature lifespan | 1000 timesteps |

### Lifespan Hypothesis Test

| Parameter | Value |
|-----------|-------|
| Clump scale | 80 |
| Sensory accuracy | 70% |
| Lifespan | 4000 timesteps |
| Runs | 10 |

---

## Data Collection Plan

### Directory Structure

```
data/experiments/
└── experiment_YYYYMMDD_HHMMSS/
    ├── config.json                    # Experiment configuration
    ├── main/                          # Main experiment results
    │   ├── result_clump1_acc55_life1000_run0.json
    │   ├── result_clump1_acc55_life1000_run1.json
    │   └── ...
    ├── lifespan_test/                 # Lifespan hypothesis results
    │   ├── result_clump80_acc70_life4000_run0.json
    │   └── ...
    └── figures/                       # Generated visualizations
        ├── figure1_3d_surface.png
        ├── ushape_curves.png
        ├── clumpscale_curves.png
        ├── lifespan_comparison.png
        ├── network_analysis.png
        ├── fitness_trajectories.png
        └── summary_statistics.csv
```

### Data Stored Per Run

```json
{
  "clump_scale": 10,
  "sensory_accuracy": 0.75,
  "lifespan": 1000,
  "run_id": 0,

  "generations_to_success": 45,
  "success": true,

  "final_mean_fitness": 423.5,
  "final_best_fitness": 456.2,
  "final_std_fitness": 34.1,

  "fitness_history_generations": [0, 10, 20, ...],
  "fitness_history_mean": [12.3, 45.6, 89.2, ...],
  "fitness_history_best": [23.4, 67.8, 112.3, ...],

  "best_genotype": [0, 1, 1, 0, ...],
  "best_weights": [2.1, 0.5, 1.8, 0.0],
  "best_biases": [0.3, -0.2],
  "best_learnable": [true, false, true, false, false, false],

  "seed": 12345,
  "timestamp": "2024-01-15T10:30:45",
  "runtime_seconds": 45.2
}
```

---

## Running the Experiments

### Quick Test (verify everything works)

```bash
cd habituation_evolution
pip install -e ".[dev]"

# Quick test with reduced parameters
python scripts/run_experiment.py --quick --name test_run
```

### Full Reproduction (matches paper)

```bash
# Full experiment (will take many hours)
python scripts/run_experiment.py \
    --runs 5 \
    --name paper_reproduction \
    --output-dir data/experiments
```

### Resumable Execution

The experiment script supports resumption - it checks for existing results and skips completed conditions:

```bash
# If interrupted, just run the same command again
python scripts/run_experiment.py --name paper_reproduction
```

### Run Specific Parts

```bash
# Main experiment only
python scripts/run_experiment.py --main-only --name main_exp

# Lifespan test only
python scripts/run_experiment.py --lifespan-only --name lifespan_exp
```

---

## Visualization

### Generate All Figures

```bash
python scripts/analyze_results.py data/experiments/paper_reproduction
```

### Generate Specific Figures

```bash
# Just the 3D surface (Figure 1)
python scripts/analyze_results.py data/experiments/paper_reproduction --figure surface

# U-shaped curves
python scripts/analyze_results.py data/experiments/paper_reproduction --figure ushape

# Lifespan comparison
python scripts/analyze_results.py data/experiments/paper_reproduction --figure lifespan

# Network analysis
python scripts/analyze_results.py data/experiments/paper_reproduction --figure network
```

### Save Without Displaying

```bash
python scripts/analyze_results.py data/experiments/paper_reproduction --no-show
```

---

## Generated Visualizations

### 1. `figure1_3d_surface.png`
Reproduction of Figure 1 from the paper - 3D surface showing evolution time.

### 2. `ushape_curves.png`
2D view showing U-shaped curves for each clump-scale vs. accuracy.

### 3. `clumpscale_curves.png`
Evolution time vs. clump-scale for each accuracy level.

### 4. `lifespan_comparison.png`
Bar chart comparing lifespan=1000 vs lifespan=4000 conditions.

### 5. `network_analysis.png`
Multi-panel analysis of evolved network architectures:
- Weight distributions for successful runs
- Bias distributions
- Forward weight vs accuracy
- Self-recurrent weight heatmap
- Learnable connection frequency
- Success rate heatmap

### 6. `fitness_trajectories.png`
Sample fitness curves over generations for representative conditions.

### 7. `summary_statistics.csv`
Table of mean generations and success rates per condition.

---

## Expected Results vs. Paper

### Figure 1 Surface Shape

**Paper findings:**
- U-shaped curves for each clump-scale
- Fastest evolution around 70-80% accuracy for large clumps
- Diagonal ravine from (low accuracy, low clump) to (high accuracy, high clump)
- Anomalous slowdown at (low accuracy, high clump)

**Our reproduction should show:**
- Same qualitative patterns
- Quantitative values may differ due to:
  - Implementation details not specified in paper
  - Random variation across runs
  - Potential differences in fitness threshold definition

### Lifespan Test

**Paper findings:**
- Clump-scale 80, 70% accuracy, lifespan 1000: ~187 generations
- Clump-scale 80, 70% accuracy, lifespan 4000: ~52 generations

**Our reproduction should show:**
- Significant decrease in evolution time with longer lifespan
- Exact values may differ but direction should match

---

## Troubleshooting

### If evolution never succeeds (all 2000 generations)

1. Check fitness threshold - may need adjustment
2. Verify GA parameters (mutation rate, crossover rate)
3. Check that sensory noise is being applied correctly

### If evolution is too fast

1. Fitness threshold may be too easy
2. Check that clumping is implemented correctly

### If results don't match paper patterns

1. Compare theoretical fitness values (optimal, sensory-only, etc.)
2. Verify network architecture matches paper (2 units, 4 connections, etc.)
3. Check weight encoding (16 magnitudes, 0.0-4.1 range)

---

## Estimated Compute Time

| Configuration | Runs | Est. Time per Run | Total Est. Time |
|---------------|------|-------------------|-----------------|
| Quick test (3×3×2) | 18 | ~30s | ~10 min |
| Main experiment (6×7×5) | 210 | ~2 min | ~7 hours |
| Lifespan test (1×10) | 10 | ~5 min | ~1 hour |
| Full reproduction | 220 | varies | ~8 hours |

Times are rough estimates and depend on hardware and whether evolution succeeds early.

---

## Next Steps After Reproduction

1. **Compare quantitatively** with paper values (if available)
2. **Analyze evolved networks** - what architectures emerge?
3. **Sensitivity analysis** - how robust are results to parameter changes?
4. **Extensions:**
   - Different noise models (continuous vs discrete)
   - Different learning rules (Oja vs basic Hebbian)
   - Larger networks
   - Different fitness functions

# Project Progress: Paper to Requirements to Implementation

## Overview

This project extracts requirements from a scientific paper and implements a reproducible simulation in Python/JAX.

**Paper:** "Exploring Adaptive Agency III: Simulating the Evolution of Habituation and Sensitization" by Todd & Miller

---

## Completed Steps

### 1. PDF Text Extraction
- **File:** `habituation_paper.md`
- Extracted full text from `habituation_paper.pdf`
- Formatted as clean markdown with proper headings, italics, and references

### 2. Requirements Document
- **File:** `simulation_requirements.md`
- Comprehensive extraction of technical specifications from paper
- Documented 16 ambiguities/design decisions where paper was unclear
- Organized into sections: Environment, Neural Network, Behavior, Learning, Fitness, Genetic Algorithm

### 3. Coding Plan
- **File:** `coding_plan.md`
- Detailed implementation plan for Python/JAX
- 9 implementation phases with milestones
- Module-by-module breakdown with function signatures
- JAX-specific patterns (vmap, scan, JIT)
- Default values for all design decisions

### 4. Implementation
- **Directory:** `habituation_evolution/`
- Complete working implementation

#### Project Structure
```
habituation_evolution/
├── pyproject.toml              # Dependencies and project config
├── README.md                   # Documentation
├── REPRODUCTION_PLAN.md        # Plan to reproduce paper results
├── config/
│   └── default.yaml            # Default parameters
├── src/habituation_evolution/
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration dataclasses
│   ├── environment.py          # Environment generation (both noise modes)
│   ├── network.py              # 2-unit recurrent neural network
│   ├── genetics.py             # Genotype encoding + GA operators
│   ├── fitness.py              # Fitness evaluation
│   └── simulation.py           # Main evolutionary loop
├── tests/
│   ├── test_environment.py
│   ├── test_network.py
│   ├── test_genetics.py
│   ├── test_fitness.py
│   └── test_integration.py
└── scripts/
    ├── run_experiment.py       # Data collection script
    └── analyze_results.py      # Visualization script
```

#### Key Design Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sensory noise | Both modes (discrete + continuous) | User requested both options |
| Motor activation | `tanh(x)` | Simpler, naturally outputs [-1, +1] |
| Weight spacing | Linear (0.0 to 4.1, 16 values) | Simplest interpretation |
| Input encoding | +1 (food), -1 (poison) | Symmetric |
| Hebbian rule | Basic, Oja, and Normalized options | Flexibility |
| Genotype | 40 bits (4×7 for connections + 2×6 for biases) | Matches paper description |

### 5. Reproduction Plan
- **File:** `habituation_evolution/REPRODUCTION_PLAN.md`
- Detailed plan to reproduce Figure 1 from paper
- Data collection strategy with incremental JSON saving
- Visualization scripts for all key figures

### 6. Verification
- Package installs correctly in virtual environment
- All imports work
- Environment generation works
- Genotype encoding/decoding works
- Simulation runs successfully

---

## How to Use

### Setup
```bash
cd habituation_evolution
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Quick Verification
```python
from habituation_evolution import SimulationConfig, run_simulation
import jax.random

config = SimulationConfig()
config.genetic.population_size = 10
config.genetic.max_generations = 5

result = run_simulation(jax.random.PRNGKey(42), config, verbose=True)
```

### Run Experiments
```bash
# Quick test (~10-20 min)
python scripts/run_experiment.py --quick --name test_run

# Full paper reproduction (~8 hours)
python scripts/run_experiment.py --runs 5 --name paper_reproduction
```

### Analyze Results
```bash
python scripts/analyze_results.py data/experiments/paper_reproduction
```

---

## Results to Reproduce from Paper

### 1. Figure 1: 3D Surface Plot
- X: Sensory accuracy (55-95%)
- Y: Clump scale (1, 5, 10, 20, 40, 80)
- Z: Generations to evolve cluster-tracking

**Expected patterns:**
- U-shaped curves for each clump-scale
- Diagonal "ravine" of fastest evolution
- Anomaly: high clump + low accuracy = slow

### 2. Lifespan Hypothesis Test
- Clump-scale 80, 70% accuracy
- Lifespan 1000 vs 4000
- Paper: 187 → 52 generations

### 3. Network Architecture Analysis
- What weights enable cluster-tracking?
- Role of self-recurrent connections

---

## Files Created

| File | Purpose |
|------|---------|
| `habituation_paper.md` | Extracted paper text |
| `simulation_requirements.md` | Technical requirements |
| `coding_plan.md` | Implementation plan |
| `habituation_evolution/` | Complete implementation |
| `REPRODUCTION_PLAN.md` | Experiment reproduction guide |

---

## Known Issues / Notes

1. **JAX Warning:** "CUDA-enabled jaxlib is not installed" - runs on CPU, which is slower but works
2. **JIT Compilation:** First run is slow due to JAX compilation; subsequent runs are faster
3. **Fitness Threshold:** May need tuning to match paper's definition of "cluster-tracking evolved"

---

## Next Steps

1. Run full experimental grid (42 conditions × 5+ runs)
2. Generate Figure 1 reproduction
3. Compare quantitatively with paper
4. Analyze evolved network architectures
5. Sensitivity analysis on GA parameters

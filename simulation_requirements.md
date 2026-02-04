# Requirements: JAX Implementation of Habituation/Sensitization Evolution Simulation

Based on: "Exploring Adaptive Agency III: Simulating the Evolution of Habituation and Sensitization" by Todd & Miller

---

## 1. Environment System

### 1.1 Substance Distribution
- Environment contains two substance types: **food** and **poison**
- Food raises fitness when eaten; poison lowers fitness when eaten
- Substances have perceivable features:
  - Food smells "sweet"
  - Poison smells "sour"

### 1.2 Clumpy Distribution Generation
- Generate initial random sequence of food/poison (presumably 50/50 probability)
- Replace each occurrence with N consecutive instances (clump-scale parameter)
- **Clump-scale values to support:** 1, 5, 10, 20, 40, 80
- Resulting statistics:
  - Minimum clump length: N
  - Modal clump length: N
  - Mean clump length: 2N
  - Maximum clump length: up to creature lifespan

> **AMBIGUITY:** The paper does not specify the exact probability distribution for the initial random sequence. Assumed to be 50% food, 50% poison (Bernoulli with p=0.5).

### 1.3 Sensory Noise Model
- Sensory accuracy parameter controls perception reliability
- **Accuracy levels to support:** 55%, 60%, 70%, 75%, 80%, 90%, 95%
- At X% accuracy, creatures mistake food for poison (and vice-versa) (100-X)% of the time

> **AMBIGUITY:** The paper does not specify *how* the noise is applied. Options:
> 1. Binary flip: with probability (1-accuracy), the sensory signal is inverted
> 2. Gaussian noise added to sensory input
> 3. Some other noise model
>
> **Design Decision Needed:** Choose noise implementation model.

---

## 2. Neural Network Architecture

### 2.1 Network Structure
- **Two units total:**
  1. **Sensory unit** (smell-sensing input unit)
  2. **Motor unit** (output unit determining eating behavior)

### 2.2 Unit Properties
| Unit Type | Activation Function | Activation Range |
|-----------|---------------------|------------------|
| Sensory   | Linear              | Not specified    |
| Motor     | Logistic (sigmoid)  | -1 to +1         |

> **AMBIGUITY:** The paper specifies motor units use "logistic with activations from -1 to +1". Standard logistic outputs [0,1]. This likely means a scaled/shifted sigmoid: `2 * sigmoid(x) - 1` or `tanh(x)`.
>
> **Design Decision Needed:** Confirm activation function form for motor unit.

> **AMBIGUITY:** Linear sensory unit output range not specified. Need to determine input encoding.

### 2.3 Connections
The network has up to **4 connections** and **2 biases**:

| Connection | From | To | Delay |
|------------|------|-----|-------|
| Forward | Sensory | Motor | 0 (immediate) |
| Recurrent feedback | Motor | Sensory | 1 timestep |
| Self-recurrent (sensory) | Sensory | Sensory | 1 timestep |
| Self-recurrent (motor) | Motor | Motor | 1 timestep |

Plus:
- Bias on sensory unit
- Bias on motor unit

### 2.4 Weight Specification
- Each connection/bias can be **positive or negative**
- **16 different magnitudes** available: 0.0 to 4.1

> **AMBIGUITY:** The paper states weights range "from 0.0 to 4.1" with 16 magnitudes. The exact spacing is not specified. Options:
> 1. Linear spacing: 0.0, 0.273, 0.547, ..., 4.1
> 2. Some other distribution (logarithmic, etc.)
>
> **Design Decision Needed:** Determine weight magnitude spacing.

### 2.5 Sensory Input Encoding

> **AMBIGUITY:** The paper does not specify the actual numeric values for "sweet" and "sour" smell inputs.
>
> **Design Decision Needed:** Define input encoding, e.g.:
> - Sweet (food): +1.0
> - Sour (poison): -1.0
> - Or some other scheme

---

## 3. Behavior and Decision Making

### 3.1 Eating Decision
- Motor unit activation determines whether creature eats
- Positive activation → eat
- Negative activation → don't eat

> **AMBIGUITY:** The paper says motor unit "activity determines whether or not the creature eats." The exact threshold is not specified.
>
> **Design Decision Needed:** Define decision threshold (presumably 0.0).

### 3.2 Network Update (per timestep)
1. Receive sensory input based on current substance (with noise applied)
2. Process input through network architecture with current weights
3. Generate behavior based on motor unit activation
4. Apply Hebbian learning to learnable connections
5. Register fitness effects

> **AMBIGUITY:** Order of operations for recurrent connections not fully specified. Presumed standard: use previous timestep activations for delayed connections.

---

## 4. Learning System

### 4.1 Learning Rule
- **Unsupervised Hebbian association**
- Genotype specifies which of the 4 connections and 2 biases are learnable

> **AMBIGUITY:** The paper mentions "Hebbian association" but does not specify:
> 1. The exact Hebbian learning rule variant (basic Hebb, Oja's rule, etc.)
> 2. Learning rate
> 3. Whether learning is bounded/normalized
>
> **Design Decision Needed:** Specify Hebbian learning rule and parameters.

### 4.2 Learnable Specification
- Each of the 6 parameters (4 connections + 2 biases) can be independently marked as learnable or fixed
- This is encoded in the genotype

---

## 5. Fitness System

### 5.1 Fitness Accumulation
- Fitness accumulates over the creature's lifespan
- Eating food → increases fitness
- Eating poison → decreases fitness

> **AMBIGUITY:** The paper does not specify:
> 1. Exact fitness increment for eating food
> 2. Exact fitness decrement for eating poison
> 3. Whether not eating has any fitness effect (e.g., starvation cost)
> 4. Initial fitness value
> 5. Whether fitness can go negative
>
> **Design Decision Needed:** Define fitness reward/penalty values.

### 5.2 Fitness Threshold
- Simulations run until "average population fitness passed a threshold indicating that adaptive cluster-tracking had evolved"

> **AMBIGUITY:** The exact threshold value is not specified.
>
> **Design Decision Needed:** Define fitness threshold for successful cluster-tracking evolution.

---

## 6. Genetic Algorithm

### 6.1 Genotype Encoding
- **Binary genotype**
- Interpreted as a **connectivity constraint matrix**
- Uses "strong genetic specification scheme" (Miller, Todd, & Hegde, 1989)
- Directly specifies:
  - Nature of each unit
  - Nature of each connection
  - Starting weight values (sign and magnitude)
  - Which connections are learnable

> **AMBIGUITY:** The exact genotype structure is not specified in this paper. Reference to Miller, Todd, & Hegde (1989) for details.
>
> **Design Decision Needed:** Define genotype encoding scheme. Suggested structure per connection:
> - 1 bit: connection present/absent
> - 1 bit: sign (+/-)
> - 4 bits: magnitude (16 values)
> - 1 bit: learnable flag
>
> Total per connection: 7 bits × 4 connections = 28 bits
> Total per bias: 6 bits × 2 biases = 12 bits
> Plus unit type specification bits

### 6.2 Population Parameters
- **Population size:** 100 creatures
- **Maximum generations:** 2000 (simulation stops if threshold not reached)

### 6.3 Selection
- **Stochastic fitness-proportionate reproduction**
- **Linear fitness scaling** applied

> **AMBIGUITY:** Linear fitness scaling parameters not specified. Standard linear scaling uses: `f' = a*f + b` where a and b are chosen to achieve target selection pressure.
>
> **Design Decision Needed:** Define linear fitness scaling parameters.

### 6.4 Genetic Operators

#### Crossover
- **Two-point crossover**

> **AMBIGUITY:** Crossover rate/probability not specified.
>
> **Design Decision Needed:** Define crossover probability.

#### Mutation
- **Point mutation**

> **AMBIGUITY:** Mutation rate not specified.
>
> **Design Decision Needed:** Define mutation rate per bit.

### 6.5 Initial Population
- Initial population can contain different unit types:
  - Motor units
  - Processing 'hidden units'
  - Sensory input units
- "In practice the evolving creatures rapidly converge on incorporating a smell-sensing input unit"

> **AMBIGUITY:** Initial population generation method not fully specified. How are initial genotypes generated? Random? Specific distribution?
>
> **Design Decision Needed:** Define initial population generation strategy.

---

## 7. Simulation Parameters

### 7.1 Creature Lifespan
- **Standard:** 1000 timesteps
- **Extended (for hypothesis testing):** 4000 timesteps

### 7.2 Experimental Conditions
- **6 clump-scales:** 1, 5, 10, 20, 40, 80
- **7 accuracy levels:** 55%, 60%, 70%, 75%, 80%, 90%, 95%
- **Total environments:** 42 (6 × 7)

### 7.3 Replication
- Minimum 5 populations per environment
- 10 populations for 75% of cases

> **AMBIGUITY:** Which cases got 10 runs vs 5 runs is not precisely specified.

### 7.4 Metrics
- **Primary metric:** Average number of generations to pass fitness threshold
- For populations that don't reach threshold by generation 2000, use 2000 as the value

---

## 8. JAX Implementation Considerations

### 8.1 Vectorization Requirements
- Vectorize across population (100 creatures simultaneously)
- Vectorize across timesteps where possible
- Vectorize across multiple independent simulation runs

### 8.2 Key Data Structures
```
Genotype: binary array encoding network specification
Phenotype (Network):
  - weights: array of shape (4,) for connections
  - biases: array of shape (2,)
  - learnable_mask: binary array indicating learnable parameters

Environment:
  - substance_sequence: array of shape (lifespan,) with food/poison indicators

Creature State:
  - sensory_activation: scalar
  - motor_activation: scalar
  - current_weights: array
  - fitness: scalar
```

### 8.3 Random Number Generation
- Use JAX's functional random number generation (PRNGKey splitting)
- Ensure reproducibility with seed management

### 8.4 JIT Compilation Targets
- Fitness evaluation function (per creature, per generation)
- Entire generation step (selection, crossover, mutation)
- Environment generation

---

## 9. Summary of Design Decisions Needed

| # | Topic | Options/Notes |
|---|-------|---------------|
| 1 | Initial food/poison probability | Assume 50/50 Bernoulli |
| 2 | Sensory noise model | Binary flip vs Gaussian |
| 3 | Motor unit activation function | `2*sigmoid(x)-1` vs `tanh(x)` |
| 4 | Weight magnitude spacing | Linear vs other |
| 5 | Sensory input encoding | +1/-1 or other values |
| 6 | Eating decision threshold | Presumably 0.0 |
| 7 | Hebbian learning rule variant | Basic Hebb, Oja's, etc. |
| 8 | Hebbian learning rate | Not specified |
| 9 | Fitness values (food/poison) | Not specified |
| 10 | Not-eating fitness effect | Starvation cost? |
| 11 | Fitness threshold for success | Not specified |
| 12 | Genotype encoding details | See Miller et al. 1989 |
| 13 | Linear fitness scaling params | Not specified |
| 14 | Crossover probability | Not specified |
| 15 | Mutation rate | Not specified |
| 16 | Initial population generation | Random? Structured? |

---

## 10. References for Clarification

The following papers may provide additional implementation details:

1. **Miller, G.F., Todd, P.M., and Hegde, S.U. (1989).** "Designing neural networks using genetic algorithms." - For genotype encoding scheme details

2. **Miller, G.F., and Todd, P.M. (1990).** "Exploring adaptive agency I: Theory and methods for simulating the evolution of learning." - For general simulation framework

3. **Todd, P.M., and Miller, G.F. (1990).** "Exploring adaptive agency II: Simulating the evolution of associative learning." - For learning rule details and baseline comparison

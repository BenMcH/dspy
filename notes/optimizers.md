# DSPy Optimizers (Teleprompters)

Optimizers (also called teleprompters) automatically improve DSPy programs by searching over prompt instructions, few-shot demonstrations, and/or model weights.

## How Optimization Works

```python
optimizer = dspy.MIPROv2(metric=my_metric, auto="medium")
optimized_program = optimizer.compile(program, trainset=trainset)
optimized_program.save("optimized.json")
```

All optimizers follow the same pattern:
1. Instantiate with metric and hyperparameters
2. Call `compile(student, trainset=trainset)` to optimize
3. Save and reuse the optimized program

## Metric Function

```python
def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    return 1.0 if pred.answer == example.answer else 0.0
```

- Returns `float`, `int`, or `bool`
- `trace` parameter used during optimization (when not None, stricter checking)
- GEPA metrics additionally return textual feedback

## Data Split Guidance

**Unlike standard ML**: Use **smaller training sets** (20–50 examples) and larger validation sets.
```python
trainset = data[:50]
valset = data[50:300]
```

---

## Few-Shot Learning Optimizers

### dspy.LabeledFewShot
Simplest optimizer: randomly samples k labeled examples as static few-shot demonstrations.

```python
optimizer = dspy.LabeledFewShot(k=16)
optimized = optimizer.compile(student, trainset=trainset)
```
**Use when**: You have quality labeled examples; no metric needed; baseline only.

---

### dspy.BootstrapFewShot
Generates and validates demonstrations by running the program on training examples and keeping those that pass the metric.

```python
optimizer = dspy.BootstrapFewShot(
    metric=metric,
    metric_threshold=None,    # Min score for bootstrapped demos
    max_bootstrapped_demos=4, # Max LM-generated demos per predictor
    max_labeled_demos=16,     # Max labeled demos per predictor
    max_rounds=1,
    teacher_settings={}       # Settings for teacher model
)
optimized = optimizer.compile(student, trainset=trainset)
```
**Use when**: ~10-50 examples; simple tasks; need validated demonstrations.

---

### dspy.BootstrapFewShotWithRandomSearch
Runs BootstrapFewShot N times with random seeds, evaluates all candidates, returns best.

```python
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    num_candidate_programs=16,  # Number of candidates to try
    num_threads=None,
    stop_at_score=None
)
optimized = optimizer.compile(student, trainset=trainset, valset=valset)
```
**Use when**: 50-200 examples; willing to run multiple trials.

---

### dspy.KNNFewShot
At inference time, retrieves the k nearest-neighbor examples (by embedding similarity) as few-shot demos.

```python
optimizer = dspy.KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder("openai/text-embedding-3-small")
)
optimized = optimizer.compile(student)
```
**Use when**: Relevant examples vary by input; have embedding model.

---

## Instruction Optimization

### dspy.COPRO
Hill-climbing coordinate optimization of prompt instructions.

```python
optimizer = dspy.COPRO(
    prompt_model=None,      # LM for generating instructions (defaults to configured LM)
    metric=metric,
    breadth=10,             # Candidates per iteration (must be >1)
    depth=3,                # Improvement rounds
    init_temperature=1.4
)
optimized = optimizer.compile(program, trainset=trainset)
```
**Use when**: Instruction-only optimization; hill-climbing; limited budget.

---

### dspy.MIPROv2 ⭐ (recommended default)
Joint Bayesian optimization of instructions + few-shot demos. Most powerful general-purpose optimizer.

```python
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",           # "light" (6 trials), "medium" (12), "heavy" (18)
    # OR set manually:
    num_candidates=10,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_threads=None,
    prompt_model=None,      # LM for proposing instructions
    task_model=None,        # LM being optimized
    seed=9,
    verbose=False
)
optimized = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,           # Optional; auto-splits from trainset if not provided
    num_trials=None,         # Override auto trial count
    minibatch=True,
    minibatch_size=35,
    program_aware_proposer=True,
    data_aware_proposer=True
)

# Zero-shot (instructions only, no demos)
optimizer = dspy.MIPROv2(metric=metric, auto="light",
                          max_bootstrapped_demos=0, max_labeled_demos=0)
```
**Use when**: 200+ examples; willing to spend $2-20; want best results.

---

### dspy.SIMBA
Stochastic mini-batch ascent with error analysis and rule generation.

```python
optimizer = dspy.SIMBA(
    metric=metric,
    bsize=32,               # Mini-batch size
    num_candidates=6,
    max_steps=8,
    max_demos=4,
    prompt_model=None
)
optimized = optimizer.compile(student, trainset=trainset, seed=0)
```
**Use when**: Want interpretable rules; mixed improvement strategy.

---

### dspy.GEPA
Genetic-Pareto reflective optimizer with LLM feedback on failures.

```python
# GEPA requires metric that returns score + textual feedback
def gepa_metric(example, output, trace) -> dspy.Prediction:
    score = 1.0 if output.answer == example.answer else 0.0
    feedback = f"Expected: {example.answer}, Got: {output.answer}"
    return dspy.Prediction(score=score, feedback=feedback)

optimizer = dspy.GEPA(
    metric=gepa_metric,
    auto="light",           # "light", "medium", "heavy"
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
optimized = optimizer.compile(student, trainset=trainset, valset=valset)
```
**Use when**: Rich textual feedback available; sample-efficient needed.

---

### dspy.InferRules
Discovers natural-language rules from demonstrations, appends to instructions.

```python
optimizer = dspy.InferRules(
    num_candidates=10,
    num_rules=10
)
optimized = optimizer.compile(student, trainset=trainset, valset=valset)
```
**Use when**: Want interpretable/understandable optimization.

---

## Finetuning

### dspy.BootstrapFinetune
Traces teacher execution, creates fine-tuning data, fine-tunes the student model weights.

```python
student.set_lm(dspy.LM("openai/gpt-4o-mini"))  # Must set LM before compile

optimizer = dspy.BootstrapFinetune(
    metric=metric,
    multitask=True,          # Share fine-tuning across predictors using same LM
    train_kwargs={"epochs": 3}
)
optimized = optimizer.compile(student, trainset=trainset, teacher=teacher_program)
```
**Use when**: After prompt optimization; want efficient small model inference.

---

## Meta-Optimizers

### dspy.BetterTogether
Chains multiple optimizers sequentially (prompt optimization + finetuning).

```python
optimizer = dspy.BetterTogether(
    metric=metric,
    p=dspy.MIPROv2(metric=metric),
    w=dspy.BootstrapFinetune(metric=metric)
)
optimized = optimizer.compile(
    student,
    trainset=trainset,
    strategy="p -> w -> p",  # Alternating optimization strategy
    valset_ratio=0.2          # Auto-split 20% of trainset for validation
)
```

---

### dspy.Ensemble
Combines multiple already-optimized programs.

```python
optimizer = dspy.Ensemble(reduce_fn=dspy.majority, size=3)
ensemble = optimizer.compile([prog1, prog2, prog3, prog4, prog5])
```

---

## Optimizer Selection Guide

| Scenario | Recommended Optimizer |
|---|---|
| ~10 labeled examples | `LabeledFewShot` or `BootstrapFewShot` |
| 50-100 examples, moderate budget | `BootstrapFewShotWithRandomSearch` |
| 200+ examples, best results | `MIPROv2(auto="medium")` |
| Instruction-only (no demos) | `MIPROv2(max_*_demos=0)` or `COPRO` |
| Rich textual feedback | `GEPA` |
| Want interpretable rules | `SIMBA` or `InferRules` |
| Combine prompt + weight opt | `BetterTogether` |
| Multiple good programs | `Ensemble` |

## Typical Costs
- Simple: $2 USD, ~10 minutes
- Range: cents to tens of dollars depending on LM size, dataset, and optimizer complexity

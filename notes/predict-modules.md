# DSPy Built-in Modules

All DSPy modules inherit from `dspy.Module`, support state persistence, LM overrides, async execution, and composition.

---

## dspy.Predict

Foundation of all DSPy modules. Maps inputs to outputs via an LM.

```python
predict = dspy.Predict(
    signature,          # "q -> a" or Signature class
    **config            # temperature, n, max_tokens, rollout_id, etc.
)
result = predict(question="What is 2+2?")
print(result.answer)
```

**Key behaviors**: Handles signature string parsing, config merging, few-shot demo injection, state tracking for optimization.

**Multiple completions**:
```python
predict = dspy.Predict("question -> answer", n=5)
result = predict(question="...")
print(result.completions.answer)  # list of 5 answers
```

---

## dspy.ChainOfThought

Injects a `reasoning` field before outputs to elicit step-by-step thinking.

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is 9 * 9?")
print(result.reasoning)  # "Let's think step by step..."
print(result.answer)     # "81"
```

**Custom rationale field**:
```python
cot = dspy.ChainOfThought(
    "question -> answer",
    rationale_field=dspy.OutputField(prefix="Analysis:", desc="${reasoning}")
)
```

---

## dspy.ReAct

Agent that interleaves reasoning with tool execution. Iteratively selects tools, executes them, observes results.

```python
def search(query: str) -> str:
    """Search Wikipedia for information."""
    ...

def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    ...

react = dspy.ReAct("question -> answer", tools=[search, calculate], max_iters=10)
result = react(question="What is the population of France divided by 1000?")
print(result.answer)
```

**Async**: `await react.acall(question="...")`

---

## dspy.ProgramOfThought

Generates Python code to solve the problem, executes it, returns the result.

```python
pot = dspy.ProgramOfThought("question -> answer: float")
result = pot(question="What is the probability of rolling a 6 on a fair die?")
print(result.answer)  # 0.1667
```

**Constraints**: Python stdlib only; no external libraries (numpy, pandas etc.).

---

## dspy.MultiChainComparison

Runs ChainOfThought N independent times, then uses an LM to compare and select the best result.

```python
mcc = dspy.MultiChainComparison("question -> answer", num_chains=5)
result = mcc(question="Explain quantum entanglement")
```

---

## dspy.BestOfN

Samples a module up to N times (using different rollout_ids), returns the first prediction above `threshold` or the highest-scoring.

```python
def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0,
    fail_count=None  # Stop after this many errors (None = no limit)
)
result = best_of_3(question="Capital of France?")
```

---

## dspy.Refine

Extends BestOfN with automatic LM-generated feedback. After each failure, generates hints passed to the next iteration.

```python
refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0
)
result = refine(question="Capital of France?")
```

**BestOfN vs Refine**:
| | BestOfN | Refine |
|---|---|---|
| Strategy | Independent sampling | Sequential with feedback |
| Improvement | Random variation | Guided refinement |
| Replaces | `dspy.Suggest` | `dspy.Assert` |

---

## dspy.Parallel

Runs multiple modules concurrently and collects all outputs.

```python
parallel = dspy.Parallel([
    dspy.ChainOfThought("question -> answer"),
    dspy.ProgramOfThought("question -> answer"),
])
results = parallel(question="...")  # Both run simultaneously
```

---

## dspy.CodeAct

Generates Python code snippets using provided tools and stdlib, executes in a sandbox.

```python
def factorial(n: int) -> int:
    """Compute factorial of n."""
    return 1 if n == 1 else n * factorial(n-1)

act = dspy.CodeAct("n -> result: int", tools=[factorial])
result = act(n=5)
```

**Constraints**:
- Tools must be plain functions (not callable objects/classes)
- No external libraries (numpy, pandas, etc.)
- All dependencies must be in the `tools` list

---

## dspy.RLM (Recursive Language Model)

See `rlms.md` for full coverage.

---

## Shared Module Patterns

### Config at init or call time
```python
module = dspy.Predict("q -> a", temperature=0.7)
result = module(q="...", config={"temperature": 1.0})  # Override at call time
```

### LM Usage Tracking
```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), track_usage=True)
result = module(input="query")
usage = result.get_lm_usage()
# {"openai/gpt-4o-mini": {"completion_tokens": 61, "prompt_tokens": 260, "total_tokens": 321}}
```

### Async Execution
```python
result = await module.acall(question="What is AI?")
```

### Module Selection Guide
- Simple task → `Predict`
- Needs reasoning → `ChainOfThought`
- External tools/info → `ReAct`
- Exact computation → `ProgramOfThought`
- Output quality constraint → `BestOfN` / `Refine`
- Very large contexts → `RLM`
- Multiple independent answers → `Parallel` / `MultiChainComparison`

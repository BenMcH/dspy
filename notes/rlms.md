# DSPy Recursive Language Models (RLM)

> **Status**: Experimental — API may change in future releases.
> 
> **Reference**: ["Recursive Language Models" (Zhang, Kraska, Khattab, 2025)](https://arxiv.org/abs/2512.24601)

## What Are RLMs?

RLMs are an inference strategy that separates **variable space** (data in a REPL) from **token space** (LLM prompts). Instead of feeding huge contexts directly into an LLM prompt (causing "context rot"), the LLM writes Python code to programmatically explore, filter, and analyze large contexts—only loading what it needs.

**Problem solved**: As context size grows, LLM performance degrades ([context rot](https://research.trychroma.com/context-rot)). RLMs let the LLM dynamically decide what to examine rather than consuming the entire context at once.

## When to Use RLM

- Context is **too large** to fit in the LLM's context window effectively
- Task benefits from **programmatic exploration** (searching, filtering, aggregating)
- You want the LLM to decide **how to decompose the problem**

## Basic Usage

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

rlm = dspy.RLM("context, query -> answer")

result = rlm(
    context="...very long document...",
    query="What is the total revenue mentioned?"
)
print(result.answer)
```

## Constructor

```python
rlm = dspy.RLM(
    signature,              # str or Signature class: "context, query -> answer"
    max_iterations=20,      # Max REPL loop iterations before fallback
    max_llm_calls=50,       # Max llm_query/llm_query_batched calls per execution
    max_output_chars=10_000,# Max chars from REPL output per iteration
    verbose=False,          # Log detailed execution info
    tools=None,             # List of additional tool functions
    sub_lm=None,            # Cheaper LM for sub-queries (defaults to configured LM)
    interpreter=None        # Custom CodeInterpreter (defaults to PythonInterpreter)
)
```

## How It Works (Iterative REPL Loop)

1. LLM receives **metadata** about inputs (type, length, preview) — not the full context
2. LLM writes **Python code** to explore data (`print`, `re.findall`, slicing, etc.)
3. Code executes in a **sandboxed interpreter** (Deno/Pyodide WASM)
4. LLM sees output and iterates, optionally calling `llm_query()` for semantic analysis
5. LLM calls `SUBMIT(answer=...)` to return final output

### Example Trace
```python
# Step 1: LLM explores
print(context[:2000])
# → [Preview of document]

# Step 2: LLM searches
import re
matches = re.findall(r'revenue.*?\$[\d,]+', context, re.IGNORECASE)
print(matches)
# → ['Revenue in Q4: $5,000,000', 'Total revenue: $20,000,000']

# Step 3: Semantic analysis via sub-LLM
result = llm_query(f"Extract the total revenue: {matches[1]}")
print(result)
# → $20,000,000

# Step 4: Submit
SUBMIT(answer=result)
```

## Built-in REPL Tools

| Tool | Description |
|------|-------------|
| `llm_query(prompt: str) -> str` | Sub-LLM call for semantic analysis (~500K char capacity) |
| `llm_query_batched(prompts: list[str]) -> list[str]` | Concurrent sub-LLM calls (much faster for batch) |
| `SUBMIT(**kwargs)` | Submit final outputs matching signature fields; ends execution |
| `print()` | Print to see results in REPL |
| Standard library | `re`, `json`, `collections`, `math`, etc. |

## Outputs

```python
result = rlm(context=data, query="Find the magic number")

result.answer                  # Output field from signature
result.trajectory              # List of {reasoning, code, output} per REPL step
result.final_reasoning         # LLM reasoning on final step
```

## Termination Conditions

1. **Success**: LLM calls `SUBMIT()` with all required output fields
2. **Fallback**: Max iterations (`max_iterations`) reached → extract module generates outputs
3. **LLM limit**: Max sub-LLM calls (`max_llm_calls`) exceeded → `RuntimeError` (LLM sees it and can handle it)
4. **Code errors**: Caught and shown to LLM as `[Error]` messages

## Usage Examples

### Long Document Q&A
```python
rlm = dspy.RLM("document, question -> answer", max_iterations=10)

with open("large_report.txt") as f:
    document = f.read()  # 500K+ characters

result = rlm(document=document, question="What were the key Q3 findings?")
```

### Using a Cheaper Sub-LM
```python
main_lm = dspy.LM("openai/gpt-5")
cheap_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=main_lm)

# Root LM (gpt-5) orchestrates; sub-LM (gpt-4o-mini) handles extraction
rlm = dspy.RLM("data, query -> summary", sub_lm=cheap_lm)
```

### Multiple Typed Outputs
```python
rlm = dspy.RLM("logs -> error_count: int, critical_errors: list[str]")

result = rlm(logs=server_logs)
print(f"Found {result.error_count} errors")
print(f"Critical: {result.critical_errors}")
```

### Custom Tools
```python
def fetch_metadata(doc_id: str) -> str:
    """Fetch metadata for a document ID from the database."""
    return database.get_metadata(doc_id)

rlm = dspy.RLM(
    "documents, query -> answer",
    tools=[fetch_metadata]
)
```

### Async Execution
```python
import asyncio

rlm = dspy.RLM("context, query -> answer")

async def process():
    return await rlm.aforward(context=data, query="Summarize this")

result = asyncio.run(process())
```

## Recursive Sub-LLM Mechanism

- Root LLM calls `llm_query(snippet)` for semantic analysis on text excerpts
- `sub_lm` processes the snippet and returns a string result
- Root LLM continues in REPL with the result
- Sub-LM runs simple inference — no nested REPL
- Thread pool executor (max 8 workers) handles `llm_query_batched` concurrency

## Installation Requirements

RLM uses [Deno](https://deno.land/) + [Pyodide](https://pyodide.org/) for the WASM sandbox:

```bash
curl -fsSL https://deno.land/install.sh | sh  # macOS/Linux
# Accept shell profile prompt, then restart shell
```

Alternative: Provide a custom `interpreter` (e.g., using [E2B](https://e2b.dev/) or [Modal](https://modal.com/)).

## Thread Safety

RLM instances are **not thread-safe** when using a custom interpreter. Use the default `PythonInterpreter` (creates a fresh instance per `forward()` call) or create separate RLM instances for concurrent use.

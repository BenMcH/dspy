# DSPy Core Concepts

## What is DSPy?

DSPy (Declarative Self-improving Python) is a framework for **programming—rather than prompting—language models**. It decouples program logic from incidental prompting choices, enabling portable, maintainable, and optimizable AI systems.

### Core Philosophy
- **Modular Design**: Write code, not strings; compose modules like PyTorch layers
- **Portability**: Swap LMs without changing program logic
- **Optimizability**: Compile high-level code into optimized prompts/weights via teleprompters
- **Declarative**: Separate interface (Signatures) from implementation (Modules)

---

## Signatures

A **Signature** declaratively specifies input/output behavior for a DSPy module.

### Inline Syntax
```python
"input -> output"
"question, context: list[str] -> answer: str"
"text -> label: bool, confidence: float"
```

**Supported Types**: `str` (default), `int`, `bool`, `float`, `list[T]`, `dict[K,V]`, `Image`, `Audio`, custom Pydantic models

### Class-Based Syntax
```python
class MyTask(dspy.Signature):
    """Task description passed as instruction to the LM."""
    input_field: str = dspy.InputField(desc="description for LM")
    output_field: str = dspy.OutputField(desc="output constraints")
```

### InputField / OutputField
- `dspy.InputField(desc="...", prefix="...")` — describes input to the LM
- `dspy.OutputField(desc="...", prefix="...")` — constrains LM output

### Signature Methods
```python
sig.with_instructions("Be concise")
sig.with_updated_fields(field_name=type_annotation)
sig.append(name, field)
sig.prepend(name, field)
sig.delete(name)
```

---

## Modules / Programs

A **Module** is a learnable building block with a `forward()` method. Compose modules to build programs.

### Base Module Class
```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.Predict("x -> y")
        self.step2 = dspy.ChainOfThought("y -> z")

    def forward(self, x: str) -> dspy.Prediction:
        y = self.step1(x=x).y
        return self.step2(y=y)
```

**Call via**: `module(param=value)` — not `module.forward(...)`

### Module Methods
```python
module.set_lm(lm)                   # Override LM for this module
module.get_lm()                     # Get current LM
module.named_predictors()           # Dict of name→Predict instances
module.predictors()                 # List of Predict instances
module.save("path.json")            # Persist full state
module.load("path.json")            # Restore state
module.dump_state()                 # Return state dict
module.load_state(state_dict)       # Restore from dict
module.batch(examples, num_threads) # Parallel batch inference
module.inspect_history(n)           # Debug last n LM calls
module.deepcopy()                   # Independent copy
module.reset_copy()                 # Copy with reset parameters
```

### Saving & Loading
```python
module.save("program.json")
loaded = MyModule()
loaded.load("program.json")

# Or using dspy.load
loaded = dspy.load("program.json")
```

---

## Language Models

### LM Class
```python
lm = dspy.LM(
    "provider/model",         # e.g. "openai/gpt-4o-mini"
    model_type="chat",        # "chat", "text", or "responses" (OpenAI Responses API)
    temperature=0.7,
    max_tokens=3000,
    cache=True,               # Default: caching enabled
    num_retries=3
)
```

### Global Configuration
```python
dspy.configure(lm=lm)
dspy.configure(lm=lm, track_usage=True)
```

### Multi-LM Usage
```python
# Temporary context override (thread-safe)
with dspy.context(lm=other_lm):
    result = module(...)

# Module-specific override
module.set_lm(dspy.LM("anthropic/claude-3-5-sonnet"))
```

### Supported Providers (via LiteLLM)
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-5` (responses API)
- **Anthropic**: `anthropic/claude-3-5-sonnet-20241022`
- **Google**: `gemini/gemini-2.5-flash`, `vertex_ai/gemini-2.0-flash`
- **Local/Ollama**: `ollama_chat/llama3.2`
- **SGLang/vLLM**: `openai/local-model` with `api_base="http://localhost:8000/v1"`
- **Databricks**: `databricks/databricks-meta-llama-3-1-70b-instruct`
- **Any OpenAI-compatible**: `openai/your-model` with `api_base=...`

### Calling LMs Directly
```python
lm("Hello, what is 2+2?")          # Returns list of strings
lm(messages=[{"role": "user", "content": "Hi"}])
```

### Caching
- Caching is **enabled by default** — same inputs always return same output
- Bypass: `lm("prompt", rollout_id=1, temperature=1.0)` — different `rollout_id` = cache miss
- Disable globally: `dspy.configure(lm=dspy.LM(..., cache=False))`
- Configure: `dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)`

### LM History & Token Usage
```python
len(lm.history)                     # Number of calls made
lm.history[-1]['cost']              # Estimated cost of last call
lm.history[-1]['usage']             # Token counts of last call

dspy.configure(lm=lm, track_usage=True)
result = module(...)
usage = result.get_lm_usage()       # {"openai/gpt-4o-mini": {"total_tokens": 321, ...}}
```

### Reasoning/Responses API Models
```python
lm = dspy.LM(
    "openai/gpt-5",
    model_type="responses",   # OpenAI Responses API
    temperature=1.0,          # Required for responses API
    max_tokens=16000
)
```

### Embedder
```python
embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=512)
embeddings = embedder(["text1", "text2"])  # Returns numpy array
```

# DSPy Retrievers, Tools, Data Handling & Utilities

---

## Retrievers

### dspy.ColBERTv2

Dense retrieval via remote ColBERT server.

```python
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(rm=colbertv2)

# Use in a module
retrieve = dspy.Retrieve(k=3)
results = retrieve(query="What is DSPy?")
print(results.passages)   # List of retrieved passages
```

```python
# Direct call
results = colbertv2("What is DSPy?", k=5)
# Returns list of dicts with 'text' and metadata
```

### dspy.Embeddings (Embedding-based Retriever)

Local dense retrieval using embedding similarity.

```python
from dspy.retrievers.embeddings import Embeddings

retriever = Embeddings(
    embedder=dspy.Embedder("openai/text-embedding-3-small"),
    corpus=["doc1", "doc2", "doc3"],
    k=3
)
results = retriever("query text")

# Save/Load
retriever.save("corpus.pkl")
retriever = Embeddings.from_saved("corpus.pkl", embedder=embedder)
```

### WeaviateRM
```python
from dspy.retrievers.weaviate_rm import WeaviateRM
retriever = WeaviateRM(
    weaviate_collection_name="MyDocs",
    weaviate_client=client,
    k=5,
    tenant_id=None  # For multi-tenancy
)
dspy.configure(rm=retriever)
```

### DatabricksRM
```python
from dspy.retrievers.databricks_rm import DatabricksRM
retriever = DatabricksRM(
    databricks_index_name="my_index",
    k=5,
    docs_id_column_name="id",
    text_column_name="content"
)
```

### dspy.Retrieve (Module)
Base retrieval module, uses globally configured `rm`.

```python
retrieve = dspy.Retrieve(k=3)
results = retrieve(query="...")
results.passages  # List of retrieved texts
```

---

## Tools

### dspy.PythonInterpreter

Execute Python code in a sandbox.

```python
interpreter = dspy.PythonInterpreter({})
result = interpreter.execute("print(2 + 2)")
# "4"
```

### Function Tools for Agents

```python
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return relevant passages."""
    results = colbertv2(query, k=3)
    return "\n".join(r['text'] for r in results)

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Use with ReAct
react = dspy.ReAct("question -> answer", tools=[search_wikipedia, calculate])
```

---

## Data Handling

### dspy.Example

```python
# Create examples
ex = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
).with_inputs("question")

# Access fields
ex.question       # "What is the capital of France?"
ex.answer         # "Paris"
ex.inputs()       # Example(question=...)
ex.labels()       # Example(answer=...)
ex.without("answer")  # Example without answer
ex.toDict()       # {"question": ..., "answer": ...}

# Build a dataset
dataset = [
    dspy.Example(question=row["q"], answer=row["a"]).with_inputs("question")
    for row in dataframe.to_dict("records")
]
```

### Built-in Datasets

```python
from dspy.datasets import HotPotQA, GSM8k

dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50)
trainset = dataset.train   # List of Example
devset = dataset.dev
testset = dataset.test
```

### Custom Dataset Loading

```python
import pandas as pd
import dspy

df = pd.read_csv("data.csv")
dataset = [
    dspy.Example(input=row["input"], output=row["output"]).with_inputs("input")
    for _, row in df.iterrows()
]
trainset = dataset[:80]
devset = dataset[80:]
```

---

## Utilities

### Caching

```python
# Configure caching
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True
)

# Bypass cache with rollout_id
lm("prompt", rollout_id=1, temperature=1.0)   # Cache miss
lm("prompt", rollout_id=2, temperature=1.0)   # Different cache entry

# Disable cache for LM
lm = dspy.LM("openai/gpt-4o-mini", cache=False)
```

### Logging

```python
dspy.enable_logging()
dspy.disable_logging()
dspy.disable_litellm_logging()
dspy.enable_litellm_logging()

# Configure loggers
from dspy.utils.logging_utils import configure_dspy_loggers
configure_dspy_loggers(level="DEBUG")
```

### History Inspection

```python
dspy.inspect_history(n=3)         # Print last 3 LM calls globally
lm.inspect_history(n=1)           # For specific LM
module.inspect_history(n=1)       # For specific module
```

### Save / Load Programs

```python
# Save
program.save("my_program.json")

# Load (preserves module type)
program = MyProgram()
program.load("my_program.json")

# Or use dspy.load
program = dspy.load("my_program.json")
```

### Async Utilities

```python
# Make a sync module async
async_module = dspy.asyncify(module)
result = await async_module(question="...")

# Make an async module sync
sync_module = dspy.syncify(async_module)
result = sync_module(question="...")
```

### Usage Tracking

```python
dspy.configure(lm=lm, track_usage=True)
result = program(input="...")
usage = result.get_lm_usage()
# {"openai/gpt-4o-mini": {"total_tokens": 321, "prompt_tokens": 260, ...}}
```

---

## MCP Integration

Model Context Protocol support for connecting to MCP servers.

```python
pip install "dspy[mcp]"
```

### HTTP MCP Server
```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def use_mcp_tools():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            dspy_tools = [dspy.Tool.from_mcp_tool(session, tool) for tool in response.tools]

    react_agent = dspy.ReAct(signature=TaskSignature, tools=dspy_tools)
    result = await react_agent.acall(task="...")
```

### Stdio MCP Server
```python
from mcp.client.stdio import stdio_client, StdioServerParameters

async def use_local_mcp():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            dspy_tools = [dspy.Tool.from_mcp_tool(session, t) for t in (await session.list_tools()).tools]
```

---

## Async Support

```python
# Built-in async for all modules
result = await module.acall(question="...")

# Custom async forward
class MyModule(dspy.Module):
    async def aforward(self, question, **kwargs):
        r1 = await self.step1.acall(question=question)
        r2 = await self.step2.acall(answer=r1.answer)
        return r2

# Async-to-sync bridge for sync contexts
with dspy.context(allow_tool_async_sync_conversion=True):
    result = async_tool(param="value")
```

---

## Production Deployment

### Observability (MLflow)
```python
import mlflow
mlflow.set_experiment("dspy-production")

with mlflow.start_run():
    mlflow.dspy.autolog()
    result = program(question="...")
    # Logs: traces, config, token usage, program structure
```

### Thread Safety
- `dspy.context()` uses thread-local storage — safe for concurrent requests
- Each request should use `with dspy.context(lm=lm):` to avoid sharing state

### Module Architecture for Services
```python
# Load once at startup
program = MyProgram()
program.load("optimized.json")

# Per-request inference (thread-safe)
def handle_request(query: str) -> str:
    with dspy.context(lm=production_lm):
        result = program(question=query)
    return result.answer
```

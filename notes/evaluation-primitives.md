# DSPy Evaluation, Primitives, Adapters & Streaming

---

## Evaluation

### dspy.Evaluate

Runs systematic evaluation of a DSPy program against a dataset.

```python
evaluator = dspy.Evaluate(
    devset=devset,
    num_threads=4,
    display_progress=True,
    display_table=5        # Show first 5 rows of results
)
score = evaluator(program, metric=my_metric)
```

### Metric Functions

```python
def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    if trace is not None:
        # Called during optimization — stricter check
        return pred.answer == example.answer
    # Called during evaluation — return float score
    return float(pred.answer.lower() == example.answer.lower())
```

### Built-in Metrics

```python
# Exact match (case-insensitive string comparison)
dspy.evaluate.answer_exact_match(example, pred)  # bool

# Check if answer appears in retrieved passages
dspy.evaluate.answer_passage_match(example, pred)  # bool

# Semantic similarity (uses LM)
semantic_f1 = dspy.evaluate.SemanticF1()
score = semantic_f1(example, pred)  # float

# Complete + grounded in context (uses LM)
cag = dspy.evaluate.CompleteAndGrounded()
score = cag(example, pred)  # float
```

### LM-Based Metrics

```python
class QualityJudge(dspy.Signature):
    """Rate the quality of a response on a scale of 0-1."""
    question: str = dspy.InputField()
    response: str = dspy.InputField()
    score: float = dspy.OutputField()

judge = dspy.Predict(QualityJudge)

def metric(example, pred, trace=None):
    result = judge(question=example.question, response=pred.answer)
    return result.score
```

---

## Assertions (Legacy)

> **Note**: Assertions are deprecated in favor of `dspy.Refine` and `dspy.BestOfN`.

### dspy.Assert (hard constraint)
Fails and retries with feedback when constraint not met. Raises `AssertionError` after `max_backtracks`.

### dspy.Suggest (soft constraint)
Same as Assert but logs failure and continues instead of raising.

```python
class MyModule(dspy.Module):
    def forward(self, text):
        pred = self.predict(text=text)
        dspy.Suggest(len(pred.summary.split()) < 100, "Summary must be under 100 words")
        return pred

# Must activate assertions
module = MyModule().activate_assertions()
```

---

## Primitives

### dspy.Example

Data structure for training/evaluation instances.

```python
ex = dspy.Example(question="What is 2+2?", answer="4").with_inputs("question")

ex.question          # "What is 2+2?"
ex.answer            # "4"
ex.inputs()          # Example with only "question"
ex.labels()          # Example with only "answer"
ex.without("answer") # Copy without answer field
ex.toDict()          # {"question": ..., "answer": ...}
ex.keys()            # dict_keys(["question", "answer"])
```

### dspy.Prediction

Output returned by DSPy modules. Extends Example with LM metadata.

```python
pred = module(question="...")
pred.answer               # Output field
pred.get_lm_usage()       # {"openai/gpt-4o-mini": {"total_tokens": 321}}
pred.completions.answer   # List of answers when n>1
```

### dspy.Image

For multimodal (vision) tasks.

```python
img = dspy.Image.from_file("path/to/image.jpg")
img = dspy.Image.from_url("https://example.com/image.jpg")
img = dspy.Image.from_PIL(pil_image)
```

```python
class DescribeImage(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    description: str = dspy.OutputField()

describe = dspy.Predict(DescribeImage)
result = describe(image=dspy.Image.from_url("..."))
```

### dspy.Audio

For audio/speech tasks.

```python
audio = dspy.Audio.from_file("audio.mp3")
audio = dspy.Audio.from_url("https://example.com/audio.mp3")
audio = dspy.Audio.from_array(array, sample_rate=16000)
```

### dspy.History

Tracks multi-turn conversation history.

```python
history = dspy.History(messages=[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
])

class ChatSignature(dspy.Signature):
    history: dspy.History = dspy.InputField()
    message: str = dspy.InputField()
    response: str = dspy.OutputField()
```

### dspy.Tool

Wraps callables for use with agents.

```python
def search(query: str) -> str:
    """Search Wikipedia for information."""
    ...

tool = dspy.Tool(search)
tool.name   # "search"
tool.desc   # "Search Wikipedia for information."
tool.args   # {"query": {"type": "string"}}

# From LangChain
tool = dspy.Tool.from_langchain(langchain_tool)

# From MCP
tool = dspy.Tool.from_mcp_tool(session, mcp_tool)

# Execution
result = tool(query="Python programming")
result = await tool.acall(query="Python programming")
```

### dspy.ToolCalls

Represents model output with tool call information.

```python
class AgentSignature(dspy.Signature):
    tools: list[dspy.Tool] = dspy.InputField()
    actions: dspy.ToolCalls = dspy.OutputField()

# Execute tool calls
results = tool_calls.execute(functions=[tool1, tool2])
```

---

## Adapters

Adapters translate between DSPy's internal format and LM API formats.

### dspy.ChatAdapter (default)
Formats signatures as chat messages (system + user + assistant).

```python
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
```

### dspy.JSONAdapter
Encodes signatures as JSON schema; expects JSON-formatted responses. Better for structured output tasks.

```python
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
```

### dspy.TwoStepAdapter
Two-phase generation: first reasoning, then final output.

```python
dspy.configure(lm=lm, adapter=dspy.TwoStepAdapter())
```

### Native Function Calling
- `ChatAdapter`: `use_native_function_calling=False` (default)
- `JSONAdapter`: `use_native_function_calling=True` (default)

---

## Streaming

### Token Streaming

```python
stream_predict = dspy.streamify(
    program,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")]
)

async def consume():
    output_stream = stream_predict(input="...")
    async for chunk in output_stream:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)  # Token
        elif isinstance(chunk, dspy.Prediction):
            return chunk  # Final result

result = asyncio.run(consume())
```

### Synchronous Streaming

```python
stream_predict = dspy.streamify(program, stream_listeners=[...], async_streaming=False)
for chunk in stream_predict(input="..."):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk, end="")
```

### Status Message Streaming

```python
class MyStatusProvider(dspy.streaming.StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return "Calling LLM..."
    def tool_start_status_message(self, instance, inputs):
        return f"Calling tool {instance.name}..."

stream_predict = dspy.streamify(
    program,
    stream_listeners=[...],
    status_message_provider=MyStatusProvider()
)
```

### StreamListener Parameters
```python
listener = dspy.streaming.StreamListener(
    signature_field_name="answer",  # Which field to stream
    predict_name="my_predictor",    # Disambiguate when multiple predictors
    allow_reuse=False               # Set True for loops (e.g., ReAct)
)
```

from unittest import mock

import pydantic
import pytest

import dspy_core as dspy


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_passes_structured_output_when_supported_by_model():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_not_using_structured_outputs_when_not_supported_by_model():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_falls_back_when_structured_outputs_fails():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_with_structured_outputs_does_not_mutate_original_signature():
    pass


@pytest.mark.skip(reason="Requires dspy.utils.DummyLM / adapter.__call__ - not available in dspy-core")
def test_json_adapter_sync_call():
    pass


@pytest.mark.skip(reason="Requires dspy.utils.DummyLM / adapter.acall - not available in dspy-core")
@pytest.mark.asyncio
async def test_json_adapter_async_call():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_on_pydantic_model():
    pass


def test_json_adapter_parse_raise_error_on_mismatch_fields():
    signature = dspy.make_signature("question->answer")
    adapter = dspy.JSONAdapter()
    with pytest.raises(dspy.utils.exceptions.AdapterParseError) as e:
        adapter.parse(signature, "{'answer1': 'Paris'}")

    assert e.value.adapter_name == "JSONAdapter"
    assert e.value.signature == signature
    assert e.value.lm_response == "{'answer1': 'Paris'}"
    assert e.value.parsed_result == {}

    assert str(e.value) == (
        "Adapter JSONAdapter failed to parse the LM response. \n\n"
        "LM Response: {'answer1': 'Paris'} \n\n"
        "Expected to find output fields in the LM response: [answer] \n\n"
        "Actual output fields parsed from the LM response: [] \n\n"
    )


def test_json_adapter_formats_image():
    # Test basic image formatting
    image = dspy.Image(url="https://example.com/image.jpg")

    class MySignature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        text: str = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"image": image})

    assert len(messages) == 2
    user_message_content = messages[1]["content"]
    assert user_message_content is not None

    # The message should have 3 chunks of types: text, image_url, text
    assert len(user_message_content) == 3
    assert user_message_content[0]["type"] == "text"
    assert user_message_content[2]["type"] == "text"

    # Assert that the image is formatted correctly
    expected_image_content = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    assert expected_image_content in user_message_content


def test_json_adapter_formats_image_with_few_shot_examples():
    class MySignature(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        text: str = dspy.OutputField()

    adapter = dspy.JSONAdapter()

    demos = [
        dspy.Example(
            image=dspy.Image(url="https://example.com/image1.jpg"),
            text="This is a test image",
        ),
        dspy.Example(
            image=dspy.Image(url="https://example.com/image2.jpg"),
            text="This is another test image",
        ),
    ]
    messages = adapter.format(MySignature, demos, {"image": dspy.Image(url="https://example.com/image3.jpg")})

    # 1 system message, 2 few shot examples (1 user and assistant message for each example), 1 user message
    assert len(messages) == 6

    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in messages[1]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in messages[3]["content"]
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}} in messages[5]["content"]


def test_json_adapter_formats_image_with_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[dspy.Image]
        tag: list[str]

    class MySignature(dspy.Signature):
        image: ImageWrapper = dspy.InputField()
        text: str = dspy.OutputField()

    image1 = dspy.Image(url="https://example.com/image1.jpg")
    image2 = dspy.Image(url="https://example.com/image2.jpg")
    image3 = dspy.Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"image": image_wrapper})

    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}

    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]


def test_json_adapter_formats_with_nested_documents():
    class DocumentWrapper(pydantic.BaseModel):
        documents: list[dspy.experimental.Document]

    class MySignature(dspy.Signature):
        document: DocumentWrapper = dspy.InputField()
        text: str = dspy.OutputField()

    doc1 = dspy.experimental.Document(data="Hello, world!")
    doc2 = dspy.experimental.Document(data="Hello, world 2!")

    document_wrapper = DocumentWrapper(documents=[doc1, doc2])

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"document": document_wrapper})

    expected_doc1_content = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "Hello, world!"},
        "citations": {"enabled": True},
    }
    expected_doc2_content = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "Hello, world 2!"},
        "citations": {"enabled": True},
    }

    assert expected_doc1_content in messages[1]["content"]
    assert expected_doc2_content in messages[1]["content"]


def test_json_adapter_formats_image_with_few_shot_examples_with_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[dspy.Image]
        tag: list[str]

    class MySignature(dspy.Signature):
        image: ImageWrapper = dspy.InputField()
        text: str = dspy.OutputField()

    image1 = dspy.Image(url="https://example.com/image1.jpg")
    image2 = dspy.Image(url="https://example.com/image2.jpg")
    image3 = dspy.Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    demos = [
        dspy.Example(
            image=image_wrapper,
            text="This is a test image",
        ),
    ]

    image_wrapper_2 = ImageWrapper(images=[dspy.Image(url="https://example.com/image4.jpg")], tag=["test", "example"])
    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, demos, {"image": image_wrapper_2})

    assert len(messages) == 4

    # Image information in the few-shot example's user message
    expected_image1_content = {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}
    expected_image2_content = {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    expected_image3_content = {"type": "image_url", "image_url": {"url": "https://example.com/image3.jpg"}}
    assert expected_image1_content in messages[1]["content"]
    assert expected_image2_content in messages[1]["content"]
    assert expected_image3_content in messages[1]["content"]

    # The query image is formatted in the last user message
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image4.jpg"}} in messages[-1]["content"]


def test_json_adapter_with_tool():
    class MySignature(dspy.Signature):
        """Answer question with the help of the tools"""

        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny"

    def get_population(country: str, year: int) -> str:
        """Get the population for a country"""
        return f"The population of {country} in {year} is 1000000"

    tools = [dspy.Tool(get_weather), dspy.Tool(get_population)]

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

    assert len(messages) == 2

    # The output field type description should be included in the system message even if the output field is nested
    assert dspy.ToolCalls.description() in messages[0]["content"]

    # The user message should include the question and the tools
    assert "What is the weather in Tokyo?" in messages[1]["content"]
    assert "get_weather" in messages[1]["content"]
    assert "get_population" in messages[1]["content"]

    # Tool arguments format should be included in the user message
    assert "{'city': {'type': 'string'}}" in messages[1]["content"]
    assert "{'country': {'type': 'string'}, 'year': {'type': 'integer'}}" in messages[1]["content"]


def test_json_adapter_with_code():
    # Test with code as input field
    class CodeAnalysis(dspy.Signature):
        """Analyze the time complexity of the code"""

        code: dspy.Code = dspy.InputField()
        result: str = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    messages = adapter.format(CodeAnalysis, [], {"code": "print('Hello, world!')"})

    assert len(messages) == 2

    # The output field type description should be included in the system message even if the output field is nested
    assert dspy.Code.description() in messages[0]["content"]

    # The user message should include the question and the tools
    assert "print('Hello, world!')" in messages[1]["content"]


def test_json_adapter_formats_conversation_history():
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    history = dspy.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )

    adapter = dspy.JSONAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the capital of France?", "history": history})

    assert len(messages) == 6
    assert messages[1]["content"] == "[[ ## question ## ]]\nWhat is the capital of France?"
    assert messages[2]["content"] == '{\n  "answer": "Paris"\n}'
    assert messages[3]["content"] == "[[ ## question ## ]]\nWhat is the capital of Germany?"
    assert messages[4]["content"] == '{\n  "answer": "Berlin"\n}'


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
@pytest.mark.asyncio
async def test_json_adapter_on_pydantic_model_async():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_fallback_to_json_mode_on_structured_output_failure():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_json_mode_no_structured_outputs():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
@pytest.mark.asyncio
async def test_json_adapter_json_mode_no_structured_outputs_async():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
@pytest.mark.asyncio
async def test_json_adapter_fallback_to_json_mode_on_structured_output_failure_async():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
def test_error_message_on_json_adapter_failure():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm - not available in dspy-core")
@pytest.mark.asyncio
async def test_error_message_on_json_adapter_failure_async():
    pass


@pytest.mark.skip(reason="Requires adapter.__call__ / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_toolcalls_native_function_calling():
    pass


@pytest.mark.skip(reason="Requires adapter.__call__ / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_toolcalls_no_native_function_calling():
    pass


@pytest.mark.skip(reason="Requires adapter.__call__ / dspy.LM / litellm - not available in dspy-core")
def test_json_adapter_native_reasoning():
    pass


@pytest.mark.skip(reason="Requires dspy.Predict / dspy.LM / litellm responses API - not available in dspy-core")
def test_json_adapter_with_responses_api():
    pass


def test_format_system_message():
    class MySignature(dspy.Signature):
        """Answer the question with multiple answers and scores"""

        question: str = dspy.InputField()
        answers: list[str] = dspy.OutputField()
        scores: list[float] = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answers` (list[str]): 
2. `scores` (list[float]):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.

{
  "answers": "{answers}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"string\\"}}",
  "scores": "{scores}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"number\\"}}"
}
In adhering to this structure, your objective is: 
        Answer the question with multiple answers and scores"""
    assert system_message == expected_system_message

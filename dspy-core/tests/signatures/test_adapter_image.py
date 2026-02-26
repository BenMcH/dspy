import os
import tempfile
from io import BytesIO

import pydantic
import pytest
import requests
from PIL import Image as PILImage

import dspy_core as dspy
from dspy_core.adapters.types.image import encode_image


@pytest.fixture
def sample_pil_image():
    """Fixture to provide a sample image for testing"""
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    response = requests.get(url)
    response.raise_for_status()
    return PILImage.open(BytesIO(response.content))


@pytest.fixture
def sample_dspy_image_download():
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    return dspy.Image(url, download=True)


@pytest.fixture
def sample_url():
    return "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"


@pytest.fixture
def sample_dspy_image_no_download():
    return dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")


def count_messages_with_image_url_pattern(messages):
    pattern = {"type": "image_url", "image_url": {"url": lambda x: isinstance(x, str)}}

    try:

        def check_pattern(obj, pattern):
            if isinstance(pattern, dict):
                if not isinstance(obj, dict):
                    return False
                return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
            if callable(pattern):
                return pattern(obj)
            return obj == pattern

        def count_patterns(obj, pattern):
            count = 0
            if check_pattern(obj, pattern):
                count += 1
            if isinstance(obj, dict):
                count += sum(count_patterns(v, pattern) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                count += sum(count_patterns(v, pattern) for v in obj)
            return count

        return count_patterns(messages, pattern)
    except Exception:
        return 0


def setup_predictor(signature, expected_output):
    raise NotImplementedError("setup_predictor requires DummyLM which is not available in dspy-core")


@pytest.mark.skip(reason="Requires DummyLM / dspy.Predict - not available in dspy-core")
@pytest.mark.parametrize(
    "test_case",
    [
        {"name": "probabilistic_classification"},
        {"name": "image_to_code"},
        {"name": "bbox_detection"},
        {"name": "multilingual_caption"},
    ],
)
def test_basic_image_operations(test_case):
    pass


@pytest.mark.skip(reason="Requires DummyLM / dspy.Predict - not available in dspy-core")
@pytest.mark.parametrize(
    "image_input,description",
    [
        ("pil_image", "PIL Image"),
        ("encoded_pil_image", "encoded PIL image string"),
        ("dspy_image_download", "dspy.Image with download=True"),
        ("dspy_image_no_download", "dspy.Image without download"),
    ],
)
def test_image_input_formats(
    request, sample_pil_image, sample_dspy_image_download, sample_dspy_image_no_download, image_input, description
):
    pass


@pytest.mark.skip(reason="Requires DummyLM / dspy.Predict / dspy.teleprompt - not available in dspy-core")
def test_predictor_save_load(sample_url, sample_pil_image):
    pass


@pytest.mark.skip(reason="Requires DummyLM / dspy.Predict / dspy.teleprompt - not available in dspy-core")
def test_save_load_complex_default_types():
    pass


class BasicImageSignature(dspy.Signature):
    """Basic signature with a single image input"""

    image: dspy.Image = dspy.InputField()
    output: str = dspy.OutputField()


class ImageListSignature(dspy.Signature):
    """Signature with a list of images input"""

    image_list: list[dspy.Image] = dspy.InputField()
    output: str = dspy.OutputField()


@pytest.mark.skip(reason="Requires DummyLM / dspy.Predict / dspy.teleprompt - not available in dspy-core")
@pytest.mark.parametrize(
    "test_case",
    [
        {"name": "basic_dspy_signature"},
        {"name": "list_dspy_signature"},
    ],
)
def test_save_load_complex_types(test_case):
    pass


@pytest.mark.skip(reason="Requires DummyLM / dspy.Predict / dspy.teleprompt - not available in dspy-core")
def test_save_load_pydantic_model():
    pass


@pytest.mark.skip(reason="Requires DummyLM / dspy.Predict - not available in dspy-core")
def test_optional_image_field():
    pass


@pytest.mark.skip(reason="Requires network access and DummyLM - not available in dspy-core")
def test_pdf_url_support():
    pass


@pytest.mark.skip(reason="Requires network access - not available in dspy-core tests")
def test_different_mime_types():
    pass


@pytest.mark.skip(reason="Requires network access - not available in dspy-core tests")
def test_mime_type_from_response_headers():
    pass


@pytest.mark.skip(reason="Requires network access and DummyLM - not available in dspy-core")
def test_pdf_from_file():
    pass


def test_image_repr():
    """Test string representation of Image objects"""
    url_image = dspy.Image("https://example.com/dog.jpg")
    assert str(url_image) == (
        "<<CUSTOM-TYPE-START-IDENTIFIER>>"
        '[{"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}}]'
        "<<CUSTOM-TYPE-END-IDENTIFIER>>"
    )
    assert repr(url_image) == "Image(url='https://example.com/dog.jpg')"

    sample_pil = PILImage.new("RGB", (60, 30), color="red")
    pil_image = dspy.Image(sample_pil)
    assert str(pil_image).startswith('<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "image_url",')
    assert str(pil_image).endswith("<<CUSTOM-TYPE-END-IDENTIFIER>>")
    assert "base64" in str(pil_image)


def test_from_methods_warn(tmp_path):
    """Deprecated from_* methods emit warnings"""
    tmp_file = tmp_path / "test.png"
    tmp_file.write_bytes(b"pngdata")

    with pytest.warns(DeprecationWarning):
        dspy.Image.from_url("https://example.com/dog.jpg")
    with pytest.warns(DeprecationWarning):
        dspy.Image.from_file(str(tmp_file))
    sample_pil = PILImage.new("RGB", (10, 10), color="blue")
    with pytest.warns(DeprecationWarning):
        dspy.Image.from_PIL(sample_pil)


def test_invalid_string_format():
    """Test that invalid string formats raise a ValueError"""
    invalid_string = "this_is_not_a_url_or_file"

    # Should raise a ValueError and not pass the string through
    with pytest.raises(ValueError, match="Unrecognized") as warning_info:
        image = dspy.Image(invalid_string)

def test_pil_image_with_download_parameter():
    """Test behavior when PIL image is passed with download=True"""
    sample_pil = PILImage.new("RGB", (60, 30), color="red")

    # PIL image should be encoded regardless of download parameter
    image_no_download = dspy.Image(sample_pil)
    image_with_download = dspy.Image(sample_pil, download=True)

    # Both should result in base64 encoded data URIs
    assert image_no_download.url.startswith("data:")
    assert image_with_download.url.startswith("data:")
    assert "base64," in image_no_download.url
    assert "base64," in image_with_download.url

    # They should be identical since PIL images are always encoded
    assert image_no_download.url == image_with_download.url

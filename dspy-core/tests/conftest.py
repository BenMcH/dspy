import pytest


def pytest_collection_modifyitems(items):
    """Auto-skip tests that require dspy.LM or litellm.completion."""
    skip_mark = pytest.mark.skip(reason="Requires dspy.LM / LM-calling infrastructure, not available in dspy-core")
    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "requires_lm" in markers:
            item.add_marker(skip_mark)

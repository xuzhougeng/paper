"""Tests for LLM JSON parsing."""

import json

import pytest
from pydantic import BaseModel

from papercli.llm import LLMClient


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    value: int


class TestJsonExtraction:
    """Tests for JSON extraction from LLM responses."""

    @pytest.fixture
    def llm_client(self):
        """Create a mock LLM client for testing _extract_json."""
        # We only need the _extract_json method, which doesn't need settings
        class MockSettings:
            def get_llm_api_key(self):
                return "test-key"
            llm = type('obj', (object,), {
                'base_url': 'http://test',
                'timeout': 30,
                'intent_model': 'test',
                'max_retries': 3,
            })()
            def get_intent_model(self):
                return "test"
            def get_eval_model(self):
                return "test"

        return LLMClient(MockSettings())

    def test_direct_json(self, llm_client):
        """Test parsing direct JSON."""
        text = '{"name": "test", "value": 42}'
        result = llm_client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_markdown(self, llm_client):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the result:

```json
{"name": "test", "value": 42}
```

That's the output.'''
        result = llm_client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_plain_code_block(self, llm_client):
        """Test extracting JSON from plain code block."""
        text = '''```
{"name": "test", "value": 42}
```'''
        result = llm_client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_json_with_surrounding_text(self, llm_client):
        """Test extracting JSON embedded in text."""
        text = 'The answer is {"name": "test", "value": 42} as shown.'
        result = llm_client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_json_with_whitespace(self, llm_client):
        """Test parsing JSON with extra whitespace."""
        text = '''
        {
            "name": "test",
            "value": 42
        }
        '''
        result = llm_client._extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_nested_json(self, llm_client):
        """Test parsing nested JSON."""
        text = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = llm_client._extract_json(text)
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_invalid_json_raises(self, llm_client):
        """Test that invalid JSON raises JSONDecodeError."""
        text = "This is not JSON at all"
        with pytest.raises(json.JSONDecodeError):
            llm_client._extract_json(text)

    def test_malformed_json_raises(self, llm_client):
        """Test that malformed JSON raises error."""
        text = '{"name": "test", "value": }'
        with pytest.raises(json.JSONDecodeError):
            llm_client._extract_json(text)


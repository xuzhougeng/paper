"""Tests for LLM JSON parsing."""

import json

import pytest
from pydantic import BaseModel

from papercli.llm import LLMClient, LLMError


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    value: int


class TestLLMError:
    """Tests for LLMError with diagnostic context."""

    def test_llmerror_basic(self):
        """Test basic LLMError creation."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_llmerror_with_model_and_url(self):
        """Test LLMError includes model and base_url in string output."""
        error = LLMError(
            "Parse failed",
            model="gpt-4o-mini",
            base_url="https://api.example.com/v1",
        )
        error_str = str(error)
        assert "Parse failed" in error_str
        assert "gpt-4o-mini" in error_str
        assert "https://api.example.com/v1" in error_str

    def test_llmerror_with_raw_responses(self):
        """Test LLMError includes raw responses in string output."""
        raw_response = "Here is my answer:\n```\n{'invalid': json}\n```"
        error = LLMError(
            "JSON parse failed",
            model="test-model",
            raw_responses=[raw_response],
        )
        error_str = str(error)
        assert "JSON parse failed" in error_str
        assert "Raw LLM response" in error_str
        assert "invalid" in error_str

    def test_llmerror_truncates_long_responses(self):
        """Test that very long raw responses are truncated."""
        long_response = "x" * 2000
        error = LLMError(
            "Parse failed",
            raw_responses=[long_response],
        )
        error_str = str(error)
        # Should truncate to ~800 chars + "..."
        assert "..." in error_str
        assert len(error_str) < 1500  # Should be much shorter than 2000

    def test_llmerror_with_original_exception(self):
        """Test LLMError includes original exception info."""
        original = ValueError("Missing required field")
        error = LLMError(
            "Validation failed",
            original_exception=original,
        )
        error_str = str(error)
        assert "ValueError" in error_str
        assert "Missing required field" in error_str

    def test_llmerror_with_multiple_responses(self):
        """Test LLMError with multiple raw responses (first try + retry)."""
        error = LLMError(
            "Both attempts failed",
            raw_responses=["First response", "Second response"],
        )
        error_str = str(error)
        assert "[1]" in error_str
        assert "[2]" in error_str
        assert "First response" in error_str
        assert "Second response" in error_str

    def test_llmerror_attributes_accessible(self):
        """Test that LLMError attributes are accessible for verbose logging."""
        error = LLMError(
            "Test error",
            model="test-model",
            base_url="http://test.com",
            raw_responses=["response1", "response2"],
            original_exception=ValueError("test"),
        )
        assert error.model == "test-model"
        assert error.base_url == "http://test.com"
        assert error.raw_responses == ["response1", "response2"]
        assert isinstance(error.original_exception, ValueError)


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

    def test_single_quotes_json_fails(self, llm_client):
        """Test that single-quoted JSON (Python dict style) fails parsing."""
        # Common LLM mistake: using Python dict syntax instead of JSON
        text = "{'name': 'test', 'value': 42}"
        with pytest.raises(json.JSONDecodeError):
            llm_client._extract_json(text)

    def test_trailing_comma_fails(self, llm_client):
        """Test that trailing comma (common LLM mistake) fails parsing."""
        text = '{"name": "test", "value": 42,}'
        with pytest.raises(json.JSONDecodeError):
            llm_client._extract_json(text)


class TestCompleteJsonErrorDiagnostics:
    """Tests for complete_json error diagnostics with LLMError."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        class MockSettings:
            def get_llm_api_key(self):
                return "test-key"
            llm = type('obj', (object,), {
                'base_url': 'https://test.example.com/v1',
                'timeout': 30,
                'intent_model': 'test-model',
                'max_retries': 3,
            })()
            def get_intent_model(self):
                return "test-model"
            def get_eval_model(self):
                return "test-model"
        return MockSettings()

    @pytest.mark.asyncio
    async def test_complete_json_raises_llmerror_on_parse_failure(self, mock_settings, mocker):
        """Test that complete_json raises LLMError with context on JSON parse failure."""
        client = LLMClient(mock_settings)

        # Mock the complete method to return invalid JSON
        invalid_response = "Sure! Here's the answer:\n\n{'name': 'test', 'value': 42}"
        mocker.patch.object(client, 'complete', return_value=invalid_response)

        with pytest.raises(LLMError) as exc_info:
            await client.complete_json(
                prompt="test",
                response_model=SimpleModel,
                retry_on_parse_error=False,  # Disable retry for simpler test
            )

        error = exc_info.value
        assert error.model == "test-model"
        assert error.base_url == "https://test.example.com/v1"
        assert invalid_response in error.raw_responses
        assert error.original_exception is not None

    @pytest.mark.asyncio
    async def test_complete_json_includes_both_responses_on_retry_failure(self, mock_settings, mocker):
        """Test that complete_json includes both first and retry responses in LLMError."""
        client = LLMClient(mock_settings)

        # Mock complete to return invalid JSON twice
        call_count = [0]
        def mock_complete(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "First attempt: {'bad': 'json'}"
            return "Retry attempt: still {'bad': 'json'}"

        mocker.patch.object(client, 'complete', side_effect=mock_complete)

        with pytest.raises(LLMError) as exc_info:
            await client.complete_json(
                prompt="test",
                response_model=SimpleModel,
                retry_on_parse_error=True,
            )

        error = exc_info.value
        assert len(error.raw_responses) == 2
        assert "First attempt" in error.raw_responses[0]
        assert "Retry attempt" in error.raw_responses[1]
        assert "retry" in str(error).lower()

    @pytest.mark.asyncio
    async def test_complete_json_error_message_is_actionable(self, mock_settings, mocker):
        """Test that LLMError message helps users diagnose the issue."""
        client = LLMClient(mock_settings)

        mocker.patch.object(client, 'complete', return_value="Not JSON at all")

        with pytest.raises(LLMError) as exc_info:
            await client.complete_json(
                prompt="test",
                response_model=SimpleModel,
                retry_on_parse_error=False,
            )

        error_str = str(exc_info.value)
        # Should include enough context to diagnose
        assert "test-model" in error_str
        assert "https://test.example.com/v1" in error_str
        assert "Not JSON at all" in error_str


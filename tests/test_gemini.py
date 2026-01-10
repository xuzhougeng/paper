"""Tests for Gemini API client."""

import base64
import json

import pytest
import respx
from httpx import Response

from papercli.gemini import GeminiClient, GeminiError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""

    class MockGeminiConfig:
        base_url = "https://api.test.com/v1beta"
        api_key = "test-api-key"
        text_model = "gemini-3-flash-preview"
        image_model = "gemini-3-pro-image-preview"
        timeout = 60.0
        max_retries = 3

    class MockSettings:
        gemini = MockGeminiConfig()

        def get_gemini_api_key(self):
            return "test-api-key"

    return MockSettings()


@pytest.fixture
def sample_text_response():
    """Sample Gemini text generation response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a test response from Gemini."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
    }


@pytest.fixture
def sample_image_response():
    """Sample Gemini image generation response with standard field names."""
    # Create a small test PNG (1x1 transparent pixel)
    png_data = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("ascii")

    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is your slide:"},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": png_data,
                            }
                        },
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
    }


@pytest.fixture
def sample_image_response_snake_case():
    """Sample Gemini image response with snake_case field names."""
    png_data = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("ascii")

    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": png_data,
                            }
                        },
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
    }


# =============================================================================
# Text Generation Tests
# =============================================================================


class TestGeminiTextGeneration:
    """Tests for Gemini text generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_success(self, mock_settings, sample_text_response):
        """Test successful text generation."""
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_text_response))

        client = GeminiClient(mock_settings)
        try:
            result = await client.generate_text("Tell me a joke")
            assert result == "This is a test response from Gemini."
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_with_system_instruction(
        self, mock_settings, sample_text_response
    ):
        """Test text generation with system instruction."""
        route = respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_text_response))

        client = GeminiClient(mock_settings)
        try:
            await client.generate_text(
                "Tell me a joke", system_instruction="Be funny and concise."
            )

            # Verify system instruction was included
            request_body = json.loads(route.calls[0].request.content)
            assert "systemInstruction" in request_body
            assert request_body["systemInstruction"]["parts"][0]["text"] == "Be funny and concise."
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_custom_model(self, mock_settings, sample_text_response):
        """Test text generation with custom model."""
        respx.post(
            "https://api.test.com/v1beta/models/custom-model:generateContent"
        ).mock(return_value=Response(200, json=sample_text_response))

        client = GeminiClient(mock_settings)
        try:
            result = await client.generate_text("Hello", model="custom-model")
            assert result == "This is a test response from Gemini."
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_api_error(self, mock_settings):
        """Test handling of API error."""
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(500, text="Internal Server Error"))

        client = GeminiClient(mock_settings)
        try:
            with pytest.raises(GeminiError) as exc_info:
                await client.generate_text("Hello")

            assert exc_info.value.status_code == 500
            assert exc_info.value.model == "gemini-3-flash-preview"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_empty_response(self, mock_settings):
        """Test handling of empty response."""
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(200, json={"candidates": []}))

        client = GeminiClient(mock_settings)
        try:
            with pytest.raises(GeminiError, match="No candidates"):
                await client.generate_text("Hello")
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_includes_api_key_header(
        self, mock_settings, sample_text_response
    ):
        """Test that API key is included in request header."""
        route = respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_text_response))

        client = GeminiClient(mock_settings)
        try:
            await client.generate_text("Hello")

            request = route.calls[0].request
            assert request.headers.get("x-goog-api-key") == "test-api-key"
        finally:
            await client.close()


# =============================================================================
# JSON Generation Tests
# =============================================================================


class TestGeminiJsonGeneration:
    """Tests for Gemini JSON generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_json_success(self, mock_settings):
        """Test successful JSON generation."""
        json_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"name": "John", "age": 30}'}],
                        "role": "model",
                    }
                }
            ]
        }
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(200, json=json_response))

        client = GeminiClient(mock_settings)
        try:
            result = await client.generate_text_json("Generate a person JSON")
            assert result == {"name": "John", "age": 30}
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_json_with_markdown(self, mock_settings):
        """Test JSON extraction from markdown code block."""
        json_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Here's the JSON:\n```json\n{\"key\": \"value\"}\n```"}
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(200, json=json_response))

        client = GeminiClient(mock_settings)
        try:
            result = await client.generate_text_json("Generate JSON")
            assert result == {"key": "value"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_text_json_invalid(self, mock_settings):
        """Test handling of invalid JSON response."""
        json_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "This is not JSON at all"}],
                        "role": "model",
                    }
                }
            ]
        }
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-flash-preview:generateContent"
        ).mock(return_value=Response(200, json=json_response))

        client = GeminiClient(mock_settings)
        try:
            with pytest.raises(GeminiError, match="Failed to parse"):
                await client.generate_text_json("Generate JSON")
        finally:
            await client.close()


# =============================================================================
# Image Generation Tests
# =============================================================================


class TestGeminiImageGeneration:
    """Tests for Gemini image generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_image_success(self, mock_settings, sample_image_response):
        """Test successful image generation."""
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_image_response))

        client = GeminiClient(mock_settings)
        try:
            image_bytes, mime_type = await client.generate_image(
                "Create a slide about AI"
            )

            assert isinstance(image_bytes, bytes)
            assert len(image_bytes) > 0
            assert image_bytes[:4] == b"\x89PNG"  # PNG magic bytes
            assert mime_type == "image/png"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_image_snake_case_fields(
        self, mock_settings, sample_image_response_snake_case
    ):
        """Test image generation with snake_case field names."""
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_image_response_snake_case))

        client = GeminiClient(mock_settings)
        try:
            image_bytes, mime_type = await client.generate_image(
                "Create a slide about AI"
            )

            assert isinstance(image_bytes, bytes)
            assert image_bytes[:4] == b"\x89PNG"
            assert mime_type == "image/png"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_image_custom_aspect_ratio(
        self, mock_settings, sample_image_response
    ):
        """Test image generation with custom aspect ratio."""
        route = respx.post(
            "https://api.test.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_image_response))

        client = GeminiClient(mock_settings)
        try:
            await client.generate_image("Create a slide", aspect_ratio="4:3")

            request_body = json.loads(route.calls[0].request.content)
            assert request_body["generationConfig"]["imageConfig"]["aspectRatio"] == "4:3"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_image_custom_size(self, mock_settings, sample_image_response):
        """Test image generation with custom size."""
        route = respx.post(
            "https://api.test.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_image_response))

        client = GeminiClient(mock_settings)
        try:
            await client.generate_image("Create a slide", image_size="2K")

            request_body = json.loads(route.calls[0].request.content)
            assert request_body["generationConfig"]["imageConfig"]["imageSize"] == "2K"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_image_fallback_on_400(
        self, mock_settings, sample_image_response
    ):
        """Test fallback to different aspect ratio on 400 error."""
        # First call with 16:9 fails
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        ).mock(
            side_effect=[
                Response(400, text="Invalid aspect ratio"),
                Response(200, json=sample_image_response),  # Second call succeeds
            ]
        )

        client = GeminiClient(mock_settings)
        try:
            image_bytes, _ = await client.generate_image(
                "Create a slide", aspect_ratio="16:9"
            )

            assert isinstance(image_bytes, bytes)
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_image_no_image_in_response(self, mock_settings):
        """Test handling of response without image data."""
        text_only_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Sorry, I cannot generate images."}],
                        "role": "model",
                    }
                }
            ]
        }
        respx.post(
            "https://api.test.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        ).mock(return_value=Response(200, json=text_only_response))

        client = GeminiClient(mock_settings)
        try:
            with pytest.raises(GeminiError, match="No image data"):
                await client.generate_image("Create a slide")
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_image_includes_response_modalities(
        self, mock_settings, sample_image_response
    ):
        """Test that responseModalities is included in request."""
        route = respx.post(
            "https://api.test.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        ).mock(return_value=Response(200, json=sample_image_response))

        client = GeminiClient(mock_settings)
        try:
            await client.generate_image("Create a slide")

            request_body = json.loads(route.calls[0].request.content)
            assert request_body["generationConfig"]["responseModalities"] == [
                "TEXT",
                "IMAGE",
            ]
        finally:
            await client.close()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestGeminiErrorHandling:
    """Tests for GeminiError class and error handling."""

    def test_gemini_error_string_representation(self):
        """Test GeminiError string representation."""
        error = GeminiError(
            "Test error",
            model="gemini-3-flash-preview",
            base_url="https://api.test.com",
            status_code=500,
            raw_response="Server error",
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert "gemini-3-flash-preview" in error_str
        assert "https://api.test.com" in error_str
        assert "500" in error_str
        assert "Server error" in error_str

    def test_gemini_error_truncates_long_response(self):
        """Test that long responses are truncated."""
        long_response = "x" * 1000
        error = GeminiError("Test error", raw_response=long_response)

        error_str = str(error)
        assert "..." in error_str
        assert len(error_str) < len(long_response) + 100

    def test_gemini_error_with_original_exception(self):
        """Test GeminiError with original exception."""
        original = ValueError("Original error")
        error = GeminiError("Wrapped error", original_exception=original)

        error_str = str(error)
        assert "Wrapped error" in error_str
        assert "ValueError" in error_str
        assert "Original error" in error_str


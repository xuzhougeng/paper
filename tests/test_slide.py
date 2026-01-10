"""Tests for slide generation functionality."""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from papercli.models import ArticleHighlights
from papercli.slide import (
    STYLE_PROMPTS,
    VALID_STYLES,
    generate_slide,
    render_slide_image,
    summarize_highlights,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_article_text():
    """Sample article text for testing."""
    return """
    Title: Deep Learning for Drug Discovery

    Abstract: We present a novel deep learning approach for predicting drug-target
    interactions. Our method achieves state-of-the-art performance on multiple
    benchmark datasets, demonstrating significant improvements over existing methods.
    The model uses a graph neural network architecture to capture molecular structures.

    Methods: We developed a graph-based neural network that represents molecules as
    graphs with atoms as nodes and bonds as edges. The model was trained on 100,000
    drug-target pairs from public databases.

    Results: Our approach achieved 95% accuracy on held-out test data, outperforming
    previous methods by 15%. The model successfully identified 3 novel drug candidates
    for cancer treatment.

    Conclusion: Deep learning shows great promise for accelerating drug discovery,
    potentially reducing development time and costs significantly.
    """


@pytest.fixture
def sample_highlights():
    """Sample ArticleHighlights for testing."""
    return ArticleHighlights(
        title="Deep Learning for Drug Discovery",
        subtitle="A Graph Neural Network Approach",
        bullets=[
            "Novel deep learning method for drug-target prediction",
            "Graph neural network captures molecular structures",
            "95% accuracy on test data, 15% improvement over baselines",
            "Identified 3 novel cancer drug candidates",
            "Trained on 100,000 drug-target pairs",
        ],
        takeaway="Deep learning can significantly accelerate drug discovery and reduce development costs.",
        keywords=["deep learning", "drug discovery", "graph neural network", "cancer"],
    )


@pytest.fixture
def mock_gemini_client():
    """Create a mock GeminiClient."""
    client = MagicMock()
    client.generate_text_json = AsyncMock()
    client.generate_image = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_cache():
    """Create a mock Cache."""
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def sample_png_bytes():
    """Sample PNG bytes for testing."""
    # 1x1 transparent PNG
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


# =============================================================================
# Style Preset Tests
# =============================================================================


class TestStylePresets:
    """Tests for style presets."""

    def test_all_valid_styles_have_prompts(self):
        """Test that all valid styles have corresponding prompts."""
        for style in VALID_STYLES:
            assert style in STYLE_PROMPTS
            assert len(STYLE_PROMPTS[style]) > 50  # Should have meaningful description

    def test_valid_styles_includes_handdrawn(self):
        """Test that handdrawn style is supported."""
        assert "handdrawn" in VALID_STYLES

    def test_valid_styles_includes_minimal(self):
        """Test that minimal style is supported."""
        assert "minimal" in VALID_STYLES

    def test_valid_styles_includes_academic(self):
        """Test that academic style is supported."""
        assert "academic" in VALID_STYLES

    def test_valid_styles_includes_dark(self):
        """Test that dark style is supported."""
        assert "dark" in VALID_STYLES

    def test_handdrawn_style_mentions_sketch(self):
        """Test handdrawn style description."""
        assert "sketch" in STYLE_PROMPTS["handdrawn"].lower() or "hand" in STYLE_PROMPTS["handdrawn"].lower()


# =============================================================================
# Highlight Summarization Tests
# =============================================================================


class TestSummarizeHighlights:
    """Tests for highlight summarization."""

    @pytest.mark.asyncio
    async def test_summarize_highlights_success(
        self, sample_article_text, sample_highlights, mock_gemini_client, mock_cache
    ):
        """Test successful highlight extraction."""
        mock_gemini_client.generate_text_json.return_value = sample_highlights.model_dump()

        result = await summarize_highlights(
            text=sample_article_text,
            client=mock_gemini_client,
            cache=mock_cache,
            num_bullets=5,
        )

        assert isinstance(result, ArticleHighlights)
        assert result.title == "Deep Learning for Drug Discovery"
        assert len(result.bullets) == 5
        mock_gemini_client.generate_text_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_highlights_uses_cache(
        self, sample_article_text, sample_highlights, mock_gemini_client, mock_cache
    ):
        """Test that cached highlights are used."""
        mock_cache.get.return_value = sample_highlights.model_dump()

        result = await summarize_highlights(
            text=sample_article_text,
            client=mock_gemini_client,
            cache=mock_cache,
        )

        assert result.title == sample_highlights.title
        mock_gemini_client.generate_text_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_highlights_caches_result(
        self, sample_article_text, sample_highlights, mock_gemini_client, mock_cache
    ):
        """Test that results are cached."""
        mock_gemini_client.generate_text_json.return_value = sample_highlights.model_dump()

        await summarize_highlights(
            text=sample_article_text,
            client=mock_gemini_client,
            cache=mock_cache,
        )

        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_highlights_clamps_bullets(
        self, sample_article_text, sample_highlights, mock_gemini_client
    ):
        """Test that bullet count is clamped to valid range."""
        mock_gemini_client.generate_text_json.return_value = sample_highlights.model_dump()

        # Test with too high value
        await summarize_highlights(
            text=sample_article_text,
            client=mock_gemini_client,
            cache=None,
            num_bullets=20,  # Should be clamped to 8
        )

        # Verify prompt includes reasonable bullet count
        call_args = mock_gemini_client.generate_text_json.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "8" in prompt  # Clamped to max

    @pytest.mark.asyncio
    async def test_summarize_highlights_progress_callback(
        self, sample_article_text, sample_highlights, mock_gemini_client
    ):
        """Test progress callback is called."""
        mock_gemini_client.generate_text_json.return_value = sample_highlights.model_dump()
        progress_messages = []

        await summarize_highlights(
            text=sample_article_text,
            client=mock_gemini_client,
            cache=None,
            progress_callback=lambda msg: progress_messages.append(msg),
        )

        assert len(progress_messages) > 0


# =============================================================================
# Slide Image Rendering Tests
# =============================================================================


class TestRenderSlideImage:
    """Tests for slide image rendering."""

    @pytest.mark.asyncio
    async def test_render_slide_image_success(
        self, sample_highlights, mock_gemini_client, sample_png_bytes
    ):
        """Test successful slide image rendering."""
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        result = await render_slide_image(
            highlights=sample_highlights,
            style="handdrawn",
            client=mock_gemini_client,
        )

        assert isinstance(result, bytes)
        assert result == sample_png_bytes
        mock_gemini_client.generate_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_render_slide_image_with_different_styles(
        self, sample_highlights, mock_gemini_client, sample_png_bytes
    ):
        """Test rendering with different styles."""
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        for style in VALID_STYLES:
            await render_slide_image(
                highlights=sample_highlights,
                style=style,
                client=mock_gemini_client,
            )

        assert mock_gemini_client.generate_image.call_count == len(VALID_STYLES)

    @pytest.mark.asyncio
    async def test_render_slide_image_invalid_style_fallback(
        self, sample_highlights, mock_gemini_client, sample_png_bytes
    ):
        """Test that invalid style falls back to minimal."""
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        result = await render_slide_image(
            highlights=sample_highlights,
            style="invalid_style",
            client=mock_gemini_client,
        )

        assert result == sample_png_bytes

    @pytest.mark.asyncio
    async def test_render_slide_image_uses_cache(
        self, sample_highlights, mock_gemini_client, mock_cache, sample_png_bytes
    ):
        """Test that cached images are used."""
        cached_b64 = base64.b64encode(sample_png_bytes).decode("ascii")
        mock_cache.get.return_value = cached_b64

        result = await render_slide_image(
            highlights=sample_highlights,
            style="handdrawn",
            client=mock_gemini_client,
            cache=mock_cache,
            image_size="1K",  # Only 1K/2K are cached
        )

        assert result == sample_png_bytes
        mock_gemini_client.generate_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_render_slide_image_caches_small_images(
        self, sample_highlights, mock_gemini_client, mock_cache, sample_png_bytes
    ):
        """Test that small images are cached."""
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        await render_slide_image(
            highlights=sample_highlights,
            style="handdrawn",
            client=mock_gemini_client,
            cache=mock_cache,
            image_size="1K",
        )

        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_render_slide_image_passes_aspect_ratio(
        self, sample_highlights, mock_gemini_client, sample_png_bytes
    ):
        """Test that aspect ratio is passed to client."""
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        await render_slide_image(
            highlights=sample_highlights,
            style="handdrawn",
            client=mock_gemini_client,
            aspect_ratio="4:3",
        )

        call_kwargs = mock_gemini_client.generate_image.call_args.kwargs
        assert call_kwargs["aspect_ratio"] == "4:3"

    @pytest.mark.asyncio
    async def test_render_slide_image_passes_image_size(
        self, sample_highlights, mock_gemini_client, sample_png_bytes
    ):
        """Test that image size is passed to client."""
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        await render_slide_image(
            highlights=sample_highlights,
            style="handdrawn",
            client=mock_gemini_client,
            image_size="2K",
        )

        call_kwargs = mock_gemini_client.generate_image.call_args.kwargs
        assert call_kwargs["image_size"] == "2K"


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestGenerateSlide:
    """Tests for the full slide generation pipeline."""

    @pytest.mark.asyncio
    async def test_generate_slide_full_pipeline(
        self, sample_article_text, sample_highlights, mock_gemini_client, sample_png_bytes
    ):
        """Test full slide generation pipeline."""
        mock_gemini_client.generate_text_json.return_value = sample_highlights.model_dump()
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        image_bytes, highlights = await generate_slide(
            text=sample_article_text,
            style="handdrawn",
            client=mock_gemini_client,
        )

        assert isinstance(image_bytes, bytes)
        assert image_bytes == sample_png_bytes
        assert isinstance(highlights, ArticleHighlights)
        assert highlights.title == sample_highlights.title

    @pytest.mark.asyncio
    async def test_generate_slide_returns_both_outputs(
        self, sample_article_text, sample_highlights, mock_gemini_client, sample_png_bytes
    ):
        """Test that both image and highlights are returned."""
        mock_gemini_client.generate_text_json.return_value = sample_highlights.model_dump()
        mock_gemini_client.generate_image.return_value = (sample_png_bytes, "image/png")

        result = await generate_slide(
            text=sample_article_text,
            style="minimal",
            client=mock_gemini_client,
        )

        assert len(result) == 2
        image_bytes, highlights = result
        assert isinstance(image_bytes, bytes)
        assert isinstance(highlights, ArticleHighlights)


# =============================================================================
# ArticleHighlights Model Tests
# =============================================================================


class TestArticleHighlightsModel:
    """Tests for ArticleHighlights Pydantic model."""

    def test_model_validation_success(self):
        """Test successful model validation."""
        data = {
            "title": "Test Title",
            "bullets": ["Point 1", "Point 2"],
            "takeaway": "Main conclusion",
        }
        highlights = ArticleHighlights.model_validate(data)

        assert highlights.title == "Test Title"
        assert len(highlights.bullets) == 2
        assert highlights.takeaway == "Main conclusion"
        assert highlights.subtitle is None
        assert highlights.keywords == []

    def test_model_with_optional_fields(self):
        """Test model with all optional fields."""
        data = {
            "title": "Test Title",
            "subtitle": "Test Subtitle",
            "bullets": ["Point 1"],
            "takeaway": "Conclusion",
            "keywords": ["keyword1", "keyword2"],
        }
        highlights = ArticleHighlights.model_validate(data)

        assert highlights.subtitle == "Test Subtitle"
        assert highlights.keywords == ["keyword1", "keyword2"]

    def test_model_bullet_count_validation(self):
        """Test bullet count validation (1-8 bullets)."""
        # Valid: 1 bullet
        data1 = {"title": "T", "bullets": ["One"], "takeaway": "C"}
        h1 = ArticleHighlights.model_validate(data1)
        assert len(h1.bullets) == 1

        # Valid: 8 bullets
        data8 = {
            "title": "T",
            "bullets": [f"Point {i}" for i in range(8)],
            "takeaway": "C",
        }
        h8 = ArticleHighlights.model_validate(data8)
        assert len(h8.bullets) == 8

    def test_model_serialization(self, sample_highlights):
        """Test model serialization."""
        json_str = sample_highlights.model_dump_json()
        data = json.loads(json_str)

        assert data["title"] == sample_highlights.title
        assert data["bullets"] == sample_highlights.bullets


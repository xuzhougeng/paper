"""Tests for Doc2X PDF extraction and JSONL conversion."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import respx
from httpx import Response

from papercli.doc2x import Doc2XClient, Doc2XError
from papercli.extract import (
    collect_all_image_urls,
    extract_page_text,
    find_image_urls,
    generate_image_filename,
    replace_image_urls,
    replace_urls_in_result,
    result_to_jsonl,
    result_to_page_records,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings for Doc2X client testing."""

    class MockDoc2XConfig:
        base_url = "https://v2.doc2x.noedgeai.com"
        api_key = "sk-test-key"
        timeout = 60.0
        poll_interval = 0.1  # Fast for tests
        max_wait = 5.0

    class MockSettings:
        doc2x = MockDoc2XConfig()

        def get_doc2x_api_key(self):
            return "sk-test-key"

    return MockSettings()


@pytest.fixture
def sample_result_with_pages():
    """Sample Doc2X result with pages array."""
    return {
        "pages": [
            {
                "page_no": 1,
                "md": "# Introduction\n\nThis is page one.",
                "blocks": [
                    {"type": "heading", "text": "Introduction"},
                    {"type": "paragraph", "text": "This is page one."},
                ],
            },
            {
                "page_no": 2,
                "md": "## Methods\n\nPage two content.",
                "blocks": [
                    {"type": "heading", "text": "Methods"},
                    {"type": "paragraph", "text": "Page two content."},
                ],
            },
        ]
    }


@pytest.fixture
def sample_result_list():
    """Sample Doc2X result as direct list of pages."""
    return [
        {"text": "First page text", "page_idx": 0},
        {"text": "Second page text", "page_idx": 1},
    ]


@pytest.fixture
def sample_result_single():
    """Sample Doc2X result as single page dict."""
    return {
        "md": "Single page document content.",
        "blocks": [{"type": "text", "content": "Some content"}],
    }


# =============================================================================
# Text Extraction Tests
# =============================================================================


class TestExtractPageText:
    """Tests for extract_page_text function."""

    def test_extract_md_field(self):
        """Test extraction from md field."""
        page = {"md": "# Hello World"}
        assert extract_page_text(page) == "# Hello World"

    def test_extract_text_field(self):
        """Test extraction from text field."""
        page = {"text": "Plain text content"}
        assert extract_page_text(page) == "Plain text content"

    def test_extract_from_blocks(self):
        """Test extraction from blocks array."""
        page = {
            "blocks": [
                {"text": "Block one"},
                {"text": "Block two"},
            ]
        }
        assert extract_page_text(page) == "Block one\nBlock two"

    def test_extract_from_blocks_content_field(self):
        """Test extraction from blocks with content field."""
        page = {
            "blocks": [
                {"content": "Content one"},
                {"content": "Content two"},
            ]
        }
        assert extract_page_text(page) == "Content one\nContent two"

    def test_extract_md_takes_precedence(self):
        """Test that md field takes precedence over blocks."""
        page = {
            "md": "Markdown content",
            "blocks": [{"text": "Block content"}],
        }
        assert extract_page_text(page) == "Markdown content"

    def test_extract_empty_page(self):
        """Test extraction from empty page."""
        assert extract_page_text({}) == ""

    def test_extract_content_as_string(self):
        """Test extraction when content is a string."""
        page = {"content": "String content"}
        assert extract_page_text(page) == "String content"

    def test_extract_content_as_list(self):
        """Test extraction when content is a list."""
        page = {"content": ["Line 1", "Line 2"]}
        assert extract_page_text(page) == "Line 1\nLine 2"


# =============================================================================
# JSONL Conversion Tests
# =============================================================================


class TestResultToPageRecords:
    """Tests for result_to_page_records function."""

    def test_dict_with_pages(self, sample_result_with_pages):
        """Test conversion of dict with pages array."""
        records = list(
            result_to_page_records(
                sample_result_with_pages,
                uid="test-uid",
                source_path="/path/to/doc.pdf",
            )
        )

        assert len(records) == 2

        # First page
        assert records[0]["doc2x_uid"] == "test-uid"
        assert records[0]["source_path"] == "/path/to/doc.pdf"
        assert records[0]["page_index"] == 0
        assert records[0]["page_no"] == 1
        assert "Introduction" in records[0]["text"]
        assert "raw_page" not in records[0]

        # Second page
        assert records[1]["page_index"] == 1
        assert records[1]["page_no"] == 2

    def test_list_of_pages(self, sample_result_list):
        """Test conversion of direct list of pages."""
        records = list(
            result_to_page_records(
                sample_result_list,
                uid="list-uid",
                source_path="/path/to/list.pdf",
            )
        )

        assert len(records) == 2
        assert records[0]["text"] == "First page text"
        assert records[1]["text"] == "Second page text"

    def test_single_page_dict(self, sample_result_single):
        """Test conversion of single page dict."""
        records = list(
            result_to_page_records(
                sample_result_single,
                uid="single-uid",
                source_path="/path/to/single.pdf",
            )
        )

        assert len(records) == 1
        assert records[0]["page_index"] == 0
        assert records[0]["page_no"] == 1
        assert "Single page document" in records[0]["text"]

    def test_include_raw(self, sample_result_with_pages):
        """Test that include_raw adds raw_page field."""
        records = list(
            result_to_page_records(
                sample_result_with_pages,
                uid="raw-uid",
                source_path="/path/to/raw.pdf",
                include_raw=True,
            )
        )

        assert "raw_page" in records[0]
        assert records[0]["raw_page"]["page_no"] == 1
        assert "blocks" in records[0]["raw_page"]

    def test_empty_result(self):
        """Test conversion of empty result."""
        records = list(
            result_to_page_records(
                None,
                uid="empty-uid",
                source_path="/path/to/empty.pdf",
            )
        )

        assert len(records) == 0


class TestResultToJsonl:
    """Tests for result_to_jsonl function."""

    def test_jsonl_format(self, sample_result_with_pages):
        """Test JSONL output format."""
        jsonl = result_to_jsonl(
            sample_result_with_pages,
            uid="jsonl-uid",
            source_path="/path/to/doc.pdf",
        )

        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            record = json.loads(line)
            assert "doc2x_uid" in record
            assert "page_no" in record
            assert "text" in record

    def test_jsonl_unicode(self):
        """Test JSONL handles unicode correctly."""
        result = {
            "pages": [
                {"md": "中文内容 with émojis 🎉"},
            ]
        }

        jsonl = result_to_jsonl(result, uid="unicode-uid", source_path="/doc.pdf")
        record = json.loads(jsonl)

        assert "中文内容" in record["text"]
        assert "🎉" in record["text"]


# =============================================================================
# Doc2X Client Tests
# =============================================================================


class TestDoc2XClient:
    """Tests for Doc2XClient."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_preupload_success(self, mock_settings):
        """Test successful preupload request."""
        respx.post("https://v2.doc2x.noedgeai.com/api/v2/parse/preupload").mock(
            return_value=Response(
                200,
                json={
                    "code": "success",
                    "data": {
                        "uid": "test-uid-123",
                        "url": "https://oss.example.com/upload?token=abc",
                    },
                },
            )
        )

        client = Doc2XClient(mock_settings)
        uid, upload_url = await client.preupload()

        assert uid == "test-uid-123"
        assert "oss.example.com" in upload_url

        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_preupload_rate_limited(self, mock_settings):
        """Test preupload rate limit handling."""
        respx.post("https://v2.doc2x.noedgeai.com/api/v2/parse/preupload").mock(
            return_value=Response(429, json={"error": "rate limited"})
        )

        client = Doc2XClient(mock_settings)

        with pytest.raises(Doc2XError) as exc_info:
            await client.preupload()

        assert "Rate limited" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_parse_status_success(self, mock_settings):
        """Test successful status check."""
        respx.get("https://v2.doc2x.noedgeai.com/api/v2/parse/status").mock(
            return_value=Response(
                200,
                json={
                    "code": "success",
                    "data": {
                        "status": "success",
                        "result": {"pages": [{"md": "Content"}]},
                    },
                },
            )
        )

        client = Doc2XClient(mock_settings)
        status_data = await client.get_parse_status("test-uid")

        assert status_data["status"] == "success"
        assert "result" in status_data

        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_poll_until_complete_success(self, mock_settings):
        """Test polling until success."""
        call_count = 0

        def mock_status(request):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                return Response(
                    200,
                    json={
                        "code": "success",
                        "data": {"status": "processing", "progress": call_count * 30},
                    },
                )
            return Response(
                200,
                json={
                    "code": "success",
                    "data": {
                        "status": "success",
                        "result": {"pages": [{"md": "Done!"}]},
                    },
                },
            )

        respx.get("https://v2.doc2x.noedgeai.com/api/v2/parse/status").mock(
            side_effect=mock_status
        )

        client = Doc2XClient(mock_settings)
        result = await client.poll_until_complete(
            uid="test-uid",
            poll_interval=0.01,  # Fast for tests
            max_wait=5.0,
        )

        assert result == {"pages": [{"md": "Done!"}]}
        assert call_count == 3

        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_poll_until_complete_failure(self, mock_settings):
        """Test polling handles failure status."""
        respx.get("https://v2.doc2x.noedgeai.com/api/v2/parse/status").mock(
            return_value=Response(
                200,
                json={
                    "code": "success",
                    "data": {
                        "status": "failed",
                        "detail": "parse_file_invalid: PDF is corrupted",
                    },
                },
            )
        )

        client = Doc2XClient(mock_settings)

        with pytest.raises(Doc2XError) as exc_info:
            await client.poll_until_complete(
                uid="test-uid",
                poll_interval=0.01,
                max_wait=1.0,
            )

        error = exc_info.value
        assert error.status == "failed"
        assert "parse_file_invalid" in str(error)

        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_poll_timeout(self, mock_settings):
        """Test polling timeout."""
        respx.get("https://v2.doc2x.noedgeai.com/api/v2/parse/status").mock(
            return_value=Response(
                200,
                json={
                    "code": "success",
                    "data": {"status": "processing", "progress": 50},
                },
            )
        )

        client = Doc2XClient(mock_settings)

        with pytest.raises(Doc2XError) as exc_info:
            await client.poll_until_complete(
                uid="test-uid",
                poll_interval=0.01,
                max_wait=0.05,  # Very short timeout
            )

        assert "timed out" in str(exc_info.value).lower()

        await client.close()


class TestDoc2XError:
    """Tests for Doc2XError class."""

    def test_error_basic(self):
        """Test basic error message."""
        error = Doc2XError("Something went wrong")
        assert "Something went wrong" in str(error)

    def test_error_with_uid(self):
        """Test error includes UID."""
        error = Doc2XError("Failed", uid="test-uid-123")
        error_str = str(error)
        assert "test-uid-123" in error_str

    def test_error_with_code_and_hint(self):
        """Test error includes error code and hint."""
        error = Doc2XError(
            "Parse failed",
            error_code="parse_file_too_large",
        )
        error_str = str(error)
        assert "parse_file_too_large" in error_str
        assert "300MB" in error_str  # Hint from ERROR_CODE_MESSAGES

    def test_error_with_detail(self):
        """Test error includes detail."""
        error = Doc2XError(
            "Failed",
            status="failed",
            detail="The PDF is password protected",
        )
        error_str = str(error)
        assert "password protected" in error_str


# =============================================================================
# Image URL Tests
# =============================================================================


class TestFindImageUrls:
    """Tests for find_image_urls function."""

    def test_find_single_url(self):
        """Test finding a single image URL."""
        text = 'Some text <img src="https://cdn.noedgeai.com/abc-123_0.jpg?x=10&y=20"/> more text'
        urls = find_image_urls(text)
        assert len(urls) == 1
        assert "cdn.noedgeai.com" in urls[0]

    def test_find_multiple_urls(self):
        """Test finding multiple image URLs."""
        text = '''
        <img src="https://cdn.noedgeai.com/abc-123_0.jpg?x=10"/>
        <img src="https://cdn.noedgeai.com/abc-123_1.png?y=20"/>
        '''
        urls = find_image_urls(text)
        assert len(urls) == 2

    def test_find_no_urls(self):
        """Test with no image URLs."""
        text = "Just some plain text without images"
        urls = find_image_urls(text)
        assert urls == []

    def test_deduplicate_urls(self):
        """Test that duplicate URLs are removed."""
        text = '''
        <img src="https://cdn.noedgeai.com/abc-123_0.jpg?x=10"/>
        <img src="https://cdn.noedgeai.com/abc-123_0.jpg?x=10"/>
        '''
        urls = find_image_urls(text)
        assert len(urls) == 1


class TestGenerateImageFilename:
    """Tests for generate_image_filename function."""

    def test_standard_doc2x_url(self):
        """Test filename generation for standard Doc2X URL."""
        url = "https://cdn.noedgeai.com/019b9d84-fe68-773d-9652-7f7dbdb9cc5d_5.jpg?x=10&y=20"
        filename = generate_image_filename(url, "test-uid")
        assert filename == "019b9d84-fe68-773d-9652-7f7dbdb9cc5d_p5.jpg"

    def test_different_extensions(self):
        """Test filename preserves extension."""
        url = "https://cdn.noedgeai.com/abc-123_10.png?x=10"
        filename = generate_image_filename(url, "test-uid")
        assert filename.endswith(".png")
        assert "_p10." in filename


class TestReplaceImageUrls:
    """Tests for replace_image_urls function."""

    def test_replace_single_url(self):
        """Test replacing a single URL."""
        text = '<img src="https://cdn.noedgeai.com/abc_0.jpg"/>'
        mapping = {"https://cdn.noedgeai.com/abc_0.jpg": "/images/abc_p0.jpg"}
        result = replace_image_urls(text, mapping)
        assert "/images/abc_p0.jpg" in result
        assert "cdn.noedgeai.com" not in result

    def test_replace_multiple_urls(self):
        """Test replacing multiple URLs."""
        text = '''
        <img src="https://cdn.noedgeai.com/abc_0.jpg"/>
        <img src="https://cdn.noedgeai.com/abc_1.jpg"/>
        '''
        mapping = {
            "https://cdn.noedgeai.com/abc_0.jpg": "/img/abc_p0.jpg",
            "https://cdn.noedgeai.com/abc_1.jpg": "/img/abc_p1.jpg",
        }
        result = replace_image_urls(text, mapping)
        assert "/img/abc_p0.jpg" in result
        assert "/img/abc_p1.jpg" in result
        assert "cdn.noedgeai.com" not in result


class TestCollectAllImageUrls:
    """Tests for collect_all_image_urls function."""

    def test_collect_from_pages(self):
        """Test collecting URLs from pages array."""
        result = {
            "pages": [
                {"md": '<img src="https://cdn.noedgeai.com/019b9d84-fe68-773d-9652-7f7dbdb9cc5d_0.jpg?x=10"/>'},
                {"md": '<img src="https://cdn.noedgeai.com/019b9d84-fe68-773d-9652-7f7dbdb9cc5d_1.jpg?y=20"/>'},
            ]
        }
        urls = collect_all_image_urls(result)
        assert len(urls) == 2

    def test_collect_from_list(self):
        """Test collecting URLs from list result."""
        result = [
            {"text": '<img src="https://cdn.noedgeai.com/019b9d84-fe68-773d-9652-7f7dbdb9cc5d_0.jpg?x=10"/>'},
            {"text": '<img src="https://cdn.noedgeai.com/019b9d84-fe68-773d-9652-7f7dbdb9cc5d_1.jpg?y=20"/>'},
        ]
        urls = collect_all_image_urls(result)
        assert len(urls) == 2

    def test_deduplicate_across_pages(self):
        """Test deduplication across pages."""
        result = {
            "pages": [
                {"md": '<img src="https://cdn.noedgeai.com/019b9d84-fe68-773d-9652-7f7dbdb9cc5d_0.jpg?x=10"/>'},
                {"md": '<img src="https://cdn.noedgeai.com/019b9d84-fe68-773d-9652-7f7dbdb9cc5d_0.jpg?x=10"/>'},
            ]
        }
        urls = collect_all_image_urls(result)
        assert len(urls) == 1


class TestReplaceUrlsInResult:
    """Tests for replace_urls_in_result function."""

    def test_replace_in_pages(self):
        """Test URL replacement in pages array."""
        result = {
            "pages": [
                {"md": '<img src="https://cdn.noedgeai.com/uid_0.jpg"/>'},
            ]
        }
        mapping = {"https://cdn.noedgeai.com/uid_0.jpg": "/local/uid_p0.jpg"}
        new_result = replace_urls_in_result(result, mapping)
        assert "/local/uid_p0.jpg" in new_result["pages"][0]["md"]

    def test_replace_in_text_field(self):
        """Test URL replacement in text field."""
        result = {"text": '<img src="https://cdn.noedgeai.com/uid_0.jpg"/>'}
        mapping = {"https://cdn.noedgeai.com/uid_0.jpg": "/local/uid_p0.jpg"}
        new_result = replace_urls_in_result(result, mapping)
        assert "/local/uid_p0.jpg" in new_result["text"]

    def test_empty_mapping(self):
        """Test with empty mapping returns original."""
        result = {"md": "some content"}
        new_result = replace_urls_in_result(result, {})
        assert new_result == result


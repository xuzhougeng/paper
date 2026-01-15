"""Tests for citation report formatting and sentence splitting."""

import pytest

from papercli.cli import (
    _split_sentences_simple,
    split_sentences_llm,
    Segment,
    SegmentationResponse,
)
from papercli.models import EvalResult, Paper
from papercli.output import format_citation_report


# ---------------------------------------------------------------------------
# Simple (fallback) splitter tests
# ---------------------------------------------------------------------------


def test_split_sentences_simple_basic():
    """Test basic punctuation-based sentence splitting."""
    text = "第一句。第二句！\n第三句？Fourth sentence! Fifth sentence? Sixth sentence."
    sentences = _split_sentences_simple(text)
    assert sentences == [
        "第一句。",
        "第二句！",
        "第三句？",
        "Fourth sentence!",
        "Fifth sentence?",
        "Sixth sentence.",
    ]


# ---------------------------------------------------------------------------
# Pydantic model validation tests
# ---------------------------------------------------------------------------


def test_segment_model_strips_whitespace():
    """Segment.text should be stripped of leading/trailing whitespace."""
    seg = Segment(text="  Hello world  ")
    assert seg.text == "Hello world"


def test_segment_model_rejects_empty_text():
    """Segment.text must not be empty or whitespace-only."""
    with pytest.raises(ValueError, match="must not be empty"):
        Segment(text="")

    with pytest.raises(ValueError, match="must not be empty"):
        Segment(text="   ")


def test_segmentation_response_rejects_empty_segments():
    """SegmentationResponse must have at least one segment."""
    with pytest.raises(ValueError, match="must not be empty"):
        SegmentationResponse(segments=[])


def test_segmentation_response_valid():
    """Valid SegmentationResponse parses correctly."""
    resp = SegmentationResponse(
        segments=[
            Segment(text="First claim.", relation_to_prev=None),
            Segment(text="Second claim.", relation_to_prev="continuation", reason="same topic"),
        ]
    )
    assert len(resp.segments) == 2
    assert resp.segments[0].text == "First claim."
    assert resp.segments[1].relation_to_prev == "continuation"

def test_segment_model_normalizes_relation_to_prev_unknown_values():
    """Unknown relation labels from the LLM should not break validation."""
    seg = Segment(text="Result statement.", relation_to_prev="result")
    assert seg.relation_to_prev == "other"

    seg2 = Segment(text="Cause statement.", relation_to_prev="cause-effect")
    assert seg2.relation_to_prev == "cause"


# ---------------------------------------------------------------------------
# LLM splitter tests (mocked)
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM client that returns predefined responses."""

    def __init__(self, response: SegmentationResponse | Exception):
        self._response = response

    async def complete_json(self, **kwargs):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.mark.asyncio
async def test_split_sentences_llm_valid_response():
    """LLM splitter returns segment texts from valid JSON response."""
    mock_response = SegmentationResponse(
        segments=[
            Segment(text="CRISPR enables precise gene editing."),
            Segment(text="Recent studies show promise in cancer therapy."),
        ]
    )
    mock_client = MockLLMClient(mock_response)

    result = await split_sentences_llm(
        text="CRISPR enables precise gene editing. Recent studies show promise in cancer therapy.",
        llm_client=mock_client,
    )

    assert result == [
        "CRISPR enables precise gene editing.",
        "Recent studies show promise in cancer therapy.",
    ]


@pytest.mark.asyncio
async def test_split_sentences_llm_propagates_error():
    """LLM splitter propagates errors from the LLM client."""
    from papercli.llm import LLMError

    mock_client = MockLLMClient(LLMError("API call failed", model="test-model"))

    with pytest.raises(LLMError, match="API call failed"):
        await split_sentences_llm(text="Some text.", llm_client=mock_client)


def test_format_citation_report_includes_recommended():
    paper = Paper(
        source="pubmed",
        source_id="pmid:123",
        title="Example Paper Title",
        abstract="Example abstract sentence for evidence.",
        year=2024,
        authors=["A. Author", "B. Author"],
        doi="10.1000/example",
        venue="Journal of Examples",
        url="https://example.org/paper",
    )
    result = EvalResult(
        paper=paper,
        score=9.2,
        meets_need=True,
        evidence_quote="Example abstract sentence for evidence.",
        evidence_field="abstract",
        short_reason="Directly supports the sentence.",
    )

    report = format_citation_report(
        [
            {
                "sentence": "This is a test sentence.",
                "results": [result],
                "recommended": result,
                "error": None,
            }
        ],
        sources=["pubmed"],
        top_k=1,
    )

    assert "# Citation Report" in report
    assert "## Sentence 1" in report
    assert "This is a test sentence." in report
    assert "Example Paper Title" in report
    assert "https://doi.org/10.1000/example" in report
    assert "Example abstract sentence for evidence." in report
    assert "Candidates (top-1)" in report

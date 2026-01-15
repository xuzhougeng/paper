"""Tests for citation report formatting and sentence splitting."""

import pytest

from papercli.cli import (
    _split_sentences_simple,
    split_sentences_llm,
    split_segments_llm,
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


def test_segment_needs_citation_defaults_to_true():
    """Segment.needs_citation defaults to True if not specified."""
    seg = Segment(text="Some claim.")
    assert seg.needs_citation is True


def test_segment_needs_citation_accepts_various_formats():
    """Segment.needs_citation normalizes various LLM output formats."""
    # Boolean values
    assert Segment(text="Claim.", needs_citation=True).needs_citation is True
    assert Segment(text="Claim.", needs_citation=False).needs_citation is False

    # String values
    assert Segment(text="Claim.", needs_citation="true").needs_citation is True
    assert Segment(text="Claim.", needs_citation="false").needs_citation is False
    assert Segment(text="Claim.", needs_citation="yes").needs_citation is True
    assert Segment(text="Claim.", needs_citation="no").needs_citation is False
    assert Segment(text="Claim.", needs_citation="1").needs_citation is True
    assert Segment(text="Claim.", needs_citation="0").needs_citation is False

    # Integer values
    assert Segment(text="Claim.", needs_citation=1).needs_citation is True
    assert Segment(text="Claim.", needs_citation=0).needs_citation is False

    # Unknown/invalid values default to True (fail-safe)
    assert Segment(text="Claim.", needs_citation="maybe").needs_citation is True
    assert Segment(text="Claim.", needs_citation=None).needs_citation is True


def test_segment_citation_reason_field():
    """Segment can store citation_reason explanation."""
    seg = Segment(
        text="We conducted an experiment.",
        needs_citation=False,
        citation_reason="Author's own work",
    )
    assert seg.needs_citation is False
    assert seg.citation_reason == "Author's own work"


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


@pytest.mark.asyncio
async def test_split_segments_llm_returns_full_segments():
    """split_segments_llm returns full Segment objects with needs_citation."""
    mock_response = SegmentationResponse(
        segments=[
            Segment(
                text="CRISPR enables precise gene editing.",
                needs_citation=True,
                citation_reason="Domain fact",
            ),
            Segment(
                text="We propose a novel approach.",
                needs_citation=False,
                citation_reason="Author's own work",
            ),
        ]
    )
    mock_client = MockLLMClient(mock_response)

    result = await split_segments_llm(
        text="CRISPR enables precise gene editing. We propose a novel approach.",
        llm_client=mock_client,
    )

    assert len(result) == 2
    assert result[0].text == "CRISPR enables precise gene editing."
    assert result[0].needs_citation is True
    assert result[0].citation_reason == "Domain fact"
    assert result[1].text == "We propose a novel approach."
    assert result[1].needs_citation is False
    assert result[1].citation_reason == "Author's own work"


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
                "needs_citation": True,
            }
        ],
        sources=["pubmed"],
        top_k=1,
    )

    assert "# Citation Report" in report
    assert "## Segment 1" in report
    assert "This is a test sentence." in report
    assert "Example Paper Title" in report
    assert "https://doi.org/10.1000/example" in report
    assert "Example abstract sentence for evidence." in report
    assert "Candidates (top-1)" in report


def test_format_citation_report_no_citation_needed():
    """Report correctly displays segments that don't need citations."""
    report = format_citation_report(
        [
            {
                "sentence": "We conducted an end-to-end benchmark.",
                "results": [],
                "recommended": None,
                "error": None,
                "needs_citation": False,
                "citation_reason": "Author's own work description",
            }
        ],
        sources=["pubmed"],
        top_k=3,
    )

    assert "# Citation Report" in report
    assert "## Segment 1" in report
    assert "We conducted an end-to-end benchmark." in report
    assert "**No citation needed.**" in report
    assert "Author's own work description" in report
    # Should NOT contain search-related output
    assert "Candidates" not in report
    assert "Recommended" not in report
    assert "No results found" not in report


def test_format_citation_report_mixed_needs_citation():
    """Report handles mix of segments needing/not needing citations."""
    paper = Paper(
        source="pubmed",
        source_id="pmid:456",
        title="Related Work Paper",
        abstract="Evidence text here.",
        year=2023,
        authors=["C. Author"],
        doi="10.1000/related",
        venue="Journal",
        url="https://example.org/related",
    )
    result = EvalResult(
        paper=paper,
        score=8.5,
        meets_need=True,
        evidence_quote="Evidence text here.",
        evidence_field="abstract",
        short_reason="Supports the claim.",
    )

    report = format_citation_report(
        [
            {
                "sentence": "We propose a novel method.",
                "results": [],
                "recommended": None,
                "error": None,
                "needs_citation": False,
                "citation_reason": "Own contribution",
            },
            {
                "sentence": "Previous work has shown X.",
                "results": [result],
                "recommended": result,
                "error": None,
                "needs_citation": True,
                "citation_reason": None,
            },
        ],
        sources=["pubmed"],
        top_k=3,
    )

    # First segment - no citation needed
    assert "We propose a novel method." in report
    assert "**No citation needed.**" in report

    # Second segment - has citation
    assert "Previous work has shown X." in report
    assert "Related Work Paper" in report
    assert "Candidates (top-1)" in report

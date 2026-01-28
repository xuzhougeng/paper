"""Tests for paper review critique functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from papercli.review.models import BioPeerReview
from papercli.review.critique import run_critique, format_review_markdown
from papercli.review.prompts import BIO_REVIEW_SYSTEM_PROMPT, BIO_REVIEW_USER_PROMPT


class TestBioPeerReviewModel:
    """Tests for BioPeerReview Pydantic model."""

    def test_minimal_valid_review(self):
        """Test creating a minimal valid review."""
        review = BioPeerReview(
            summary="This paper presents a novel approach.",
        )
        assert review.summary == "This paper presents a novel approach."
        assert review.strengths == []
        assert review.weaknesses == []
        assert review.confidence is None
        assert review.overall_recommendation is None

    def test_full_review(self):
        """Test creating a full review with all fields."""
        review = BioPeerReview(
            summary="This paper presents a novel CRISPR approach.",
            strengths=["Novel methodology", "Rigorous statistics"],
            weaknesses=["Limited sample size"],
            major_concerns=["Missing controls"],
            minor_comments=["Typo in Figure 2"],
            questions_for_authors=["What is the off-target rate?"],
            suggested_experiments=["Include wildtype control"],
            reproducibility_and_data=["Data available on GEO"],
            ethics_and_compliance=["IRB approved"],
            writing_and_clarity=["Well written"],
            confidence=4,
            overall_recommendation="minor_revision",
        )
        assert len(review.strengths) == 2
        assert review.confidence == 4
        assert review.overall_recommendation == "minor_revision"

    def test_invalid_confidence(self):
        """Test that invalid confidence values are rejected."""
        with pytest.raises(ValueError):
            BioPeerReview(
                summary="Test",
                confidence=6,  # Out of range
            )

    def test_invalid_recommendation(self):
        """Test that invalid recommendations are rejected."""
        with pytest.raises(ValueError):
            BioPeerReview(
                summary="Test",
                overall_recommendation="maybe",  # Invalid
            )

    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        review = BioPeerReview(
            summary="Test summary",
            strengths=["Strength 1"],
            weaknesses=["Weakness 1"],
            confidence=3,
            overall_recommendation="accept",
        )
        json_str = review.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["summary"] == "Test summary"
        assert parsed["confidence"] == 3

        # Round-trip
        review2 = BioPeerReview.model_validate(parsed)
        assert review2.summary == review.summary


class TestFormatReviewMarkdown:
    """Tests for markdown formatting of reviews."""

    def test_format_minimal_review(self):
        """Test formatting a minimal review."""
        review = BioPeerReview(summary="This is a test paper.")
        md = format_review_markdown(review)
        assert "# Peer Review Report" in md
        assert "## Summary" in md
        assert "This is a test paper." in md

    def test_format_full_review(self):
        """Test formatting a full review."""
        review = BioPeerReview(
            summary="This paper presents novel findings.",
            strengths=["Well-designed experiments", "Clear writing"],
            weaknesses=["Limited generalizability"],
            major_concerns=["Statistical analysis needs revision"],
            questions_for_authors=["What about batch effects?"],
            suggested_experiments=["Add independent validation"],
            reproducibility_and_data=["Code available on GitHub"],
            writing_and_clarity=["Minor grammar issues"],
            minor_comments=["Figure 3 needs higher resolution"],
            confidence=4,
            overall_recommendation="minor_revision",
        )
        md = format_review_markdown(review)

        assert "## Strengths" in md
        assert "Well-designed experiments" in md
        assert "## Weaknesses" in md
        assert "## Major Concerns" in md
        assert "## Questions for Authors" in md
        assert "## Suggested Experiments" in md
        assert "## Reproducibility and Data" in md
        assert "## Writing and Clarity" in md
        assert "## Minor Comments" in md
        assert "**Recommendation:** Minor Revision" in md
        assert "**Reviewer Confidence:** 4/5" in md

    def test_empty_sections_not_shown(self):
        """Test that empty sections are not shown."""
        review = BioPeerReview(
            summary="Test paper.",
            strengths=["Good work"],
            # All other lists empty
        )
        md = format_review_markdown(review)
        assert "## Strengths" in md
        assert "## Weaknesses" not in md
        assert "## Ethics" not in md


class TestPrompts:
    """Tests for review prompts."""

    def test_system_prompt_contains_checklist(self):
        """Test that system prompt contains biology checklist."""
        assert "Experimental Design" in BIO_REVIEW_SYSTEM_PROMPT
        assert "Statistical Analysis" in BIO_REVIEW_SYSTEM_PROMPT
        assert "replicates" in BIO_REVIEW_SYSTEM_PROMPT
        assert "antibodies" in BIO_REVIEW_SYSTEM_PROMPT.lower()
        assert "QC" in BIO_REVIEW_SYSTEM_PROMPT

    def test_user_prompt_template(self):
        """Test user prompt can be formatted."""
        prompt = BIO_REVIEW_USER_PROMPT.format(text="Sample manuscript text.")
        assert "Sample manuscript text." in prompt
        assert "summary" in prompt
        assert "strengths" in prompt
        assert "weaknesses" in prompt


class TestRunCritique:
    """Tests for the critique runner with mocked LLM."""

    @pytest.mark.asyncio
    async def test_run_critique_calls_llm(self):
        """Test that run_critique calls the LLM client correctly."""
        # Create mock LLM client
        mock_llm = MagicMock()
        mock_review = BioPeerReview(
            summary="Test summary",
            strengths=["Strength 1"],
            weaknesses=["Weakness 1"],
        )
        mock_llm.complete_json = AsyncMock(return_value=mock_review)

        # Run critique
        result = await run_critique(
            text="Test manuscript text",
            llm_client=mock_llm,
            model="gpt-4o",
            temperature=0.3,
        )

        # Verify LLM was called
        mock_llm.complete_json.assert_called_once()
        call_kwargs = mock_llm.complete_json.call_args.kwargs

        assert "Test manuscript text" in call_kwargs["prompt"]
        assert call_kwargs["response_model"] == BioPeerReview
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.3

        # Verify result
        assert result.summary == "Test summary"
        assert result.strengths == ["Strength 1"]

    @pytest.mark.asyncio
    async def test_run_critique_uses_system_prompt(self):
        """Test that run_critique uses the biology review system prompt."""
        mock_llm = MagicMock()
        mock_review = BioPeerReview(summary="Test")
        mock_llm.complete_json = AsyncMock(return_value=mock_review)

        await run_critique(
            text="Test text",
            llm_client=mock_llm,
        )

        call_kwargs = mock_llm.complete_json.call_args.kwargs
        assert "biology" in call_kwargs["system_prompt"].lower() or "biological" in call_kwargs["system_prompt"].lower()
        assert "peer reviewer" in call_kwargs["system_prompt"].lower()

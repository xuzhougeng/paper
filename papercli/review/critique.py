"""LLM-based paper critique functionality."""

from typing import TYPE_CHECKING

from papercli.review.models import BioPeerReview
from papercli.review.prompts import BIO_REVIEW_SYSTEM_PROMPT, BIO_REVIEW_USER_PROMPT

if TYPE_CHECKING:
    from papercli.llm import LLMClient


async def run_critique(
    text: str,
    llm_client: "LLMClient",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 8000,
) -> BioPeerReview:
    """
    Run LLM-based critique on paper text.

    Args:
        text: The manuscript text to review
        llm_client: Initialized LLMClient instance
        model: Optional model override (defaults to instinct_model for thorough analysis)
        temperature: Sampling temperature (lower for more consistent reviews)
        max_tokens: Maximum tokens in response

    Returns:
        BioPeerReview with structured review output

    Raises:
        LLMError: If LLM call fails or returns invalid JSON
    """
    prompt = BIO_REVIEW_USER_PROMPT.format(text=text)

    response = await llm_client.complete_json(
        prompt=prompt,
        response_model=BioPeerReview,
        model=model,
        system_prompt=BIO_REVIEW_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        retry_on_parse_error=True,
    )

    return response


def format_review_markdown(review: BioPeerReview) -> str:
    """Format BioPeerReview as readable Markdown."""
    lines = ["# Peer Review Report", ""]

    # Summary
    lines.append("## Summary")
    lines.append(review.summary)
    lines.append("")

    # Overall recommendation and confidence
    if review.overall_recommendation or review.confidence:
        lines.append("## Assessment")
        if review.overall_recommendation:
            rec_display = review.overall_recommendation.replace("_", " ").title()
            lines.append(f"**Recommendation:** {rec_display}")
        if review.confidence:
            lines.append(f"**Reviewer Confidence:** {review.confidence}/5")
        lines.append("")

    # Strengths
    if review.strengths:
        lines.append("## Strengths")
        for i, s in enumerate(review.strengths, 1):
            lines.append(f"{i}. {s}")
        lines.append("")

    # Weaknesses
    if review.weaknesses:
        lines.append("## Weaknesses")
        for i, w in enumerate(review.weaknesses, 1):
            lines.append(f"{i}. {w}")
        lines.append("")

    # Major Concerns
    if review.major_concerns:
        lines.append("## Major Concerns")
        for i, c in enumerate(review.major_concerns, 1):
            lines.append(f"{i}. {c}")
        lines.append("")

    # Questions for Authors
    if review.questions_for_authors:
        lines.append("## Questions for Authors")
        for i, q in enumerate(review.questions_for_authors, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    # Suggested Experiments
    if review.suggested_experiments:
        lines.append("## Suggested Experiments")
        for i, e in enumerate(review.suggested_experiments, 1):
            lines.append(f"{i}. {e}")
        lines.append("")

    # Reproducibility and Data
    if review.reproducibility_and_data:
        lines.append("## Reproducibility and Data Availability")
        for i, r in enumerate(review.reproducibility_and_data, 1):
            lines.append(f"{i}. {r}")
        lines.append("")

    # Ethics and Compliance
    if review.ethics_and_compliance:
        lines.append("## Ethics and Compliance")
        for i, e in enumerate(review.ethics_and_compliance, 1):
            lines.append(f"{i}. {e}")
        lines.append("")

    # Writing and Clarity
    if review.writing_and_clarity:
        lines.append("## Writing and Clarity")
        for i, w in enumerate(review.writing_and_clarity, 1):
            lines.append(f"{i}. {w}")
        lines.append("")

    # Minor Comments
    if review.minor_comments:
        lines.append("## Minor Comments")
        for i, m in enumerate(review.minor_comments, 1):
            lines.append(f"{i}. {m}")
        lines.append("")

    return "\n".join(lines)

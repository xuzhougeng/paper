"""Data models for paper review."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BioPeerReview(BaseModel):
    """Structured peer review output for biology papers."""

    summary: str = Field(
        ...,
        description="One-paragraph summary of the paper's main contribution and findings",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="List of paper strengths (novelty, methodology, significance)",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="List of paper weaknesses (limitations, gaps)",
    )
    major_concerns: list[str] = Field(
        default_factory=list,
        description="Major concerns requiring author response (experimental design, statistics, controls, causal inference)",
    )
    minor_comments: list[str] = Field(
        default_factory=list,
        description="Minor comments on presentation, formatting, references",
    )
    questions_for_authors: list[str] = Field(
        default_factory=list,
        description="Specific questions authors should address",
    )
    suggested_experiments: list[str] = Field(
        default_factory=list,
        description="Additional experiments or analyses that would strengthen the work",
    )
    reproducibility_and_data: list[str] = Field(
        default_factory=list,
        description="Comments on data availability, code, materials, reproducibility",
    )
    ethics_and_compliance: list[str] = Field(
        default_factory=list,
        description="Comments on ethics (IRB, animal protocols, consent) if applicable",
    )
    writing_and_clarity: list[str] = Field(
        default_factory=list,
        description="Comments on writing quality, clarity, organization",
    )
    confidence: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Reviewer confidence (1=low, 5=expert)",
    )
    overall_recommendation: Optional[
        Literal["accept", "minor_revision", "major_revision", "reject"]
    ] = Field(
        default=None,
        description="Overall recommendation (optional)",
    )


class LintIssue(BaseModel):
    """A single lint issue found in the text."""

    rule_id: str = Field(..., description="Rule identifier (e.g., 'case-title')")
    severity: Literal["info", "warn", "error"] = Field(
        default="warn", description="Issue severity"
    )
    message: str = Field(..., description="Human-readable description of the issue")
    line: Optional[int] = Field(
        default=None, description="1-based line number (if applicable)"
    )
    column: Optional[int] = Field(
        default=None, description="1-based column number (if applicable)"
    )
    span_start: Optional[int] = Field(
        default=None, description="Start character offset in the text"
    )
    span_end: Optional[int] = Field(
        default=None, description="End character offset in the text"
    )
    context: Optional[str] = Field(
        default=None, description="Snippet of text around the issue"
    )
    suggestion: Optional[str] = Field(
        default=None, description="Suggested fix (if available)"
    )


class LintReport(BaseModel):
    """Report containing all lint issues."""

    issues: list[LintIssue] = Field(default_factory=list)
    rules_checked: list[str] = Field(
        default_factory=list, description="List of rule IDs that were checked"
    )
    total_lines: int = Field(default=0, description="Total lines in the input")
    total_chars: int = Field(default=0, description="Total characters in the input")

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")

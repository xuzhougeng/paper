"""Data models for papercli."""

from typing import Optional

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Unified paper model across all sources."""

    source: str = Field(..., description="Source identifier: pubmed, openalex, scholar, arxiv")
    source_id: str = Field(..., description="ID within the source (PMID, OpenAlex ID, etc.)")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    year: Optional[int] = Field(None, description="Publication year")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    url: Optional[str] = Field(None, description="URL to the paper")
    doi: Optional[str] = Field(None, description="DOI identifier")
    venue: Optional[str] = Field(None, description="Journal or conference name")

    # For deduplication
    @property
    def normalized_title(self) -> str:
        """Normalized title for deduplication."""
        import re
        # Lowercase, remove punctuation, collapse whitespace
        t = self.title.lower()
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t


class EvalResult(BaseModel):
    """LLM evaluation result for a paper."""

    paper: Paper
    score: float = Field(..., ge=0, le=10, description="Relevance score (0-10)")
    meets_need: bool = Field(..., description="Whether the paper meets the user's need")
    evidence_quote: str = Field(..., description="Quote from title/abstract supporting relevance")
    evidence_field: str = Field(..., description="Field the quote came from: title or abstract")
    short_reason: str = Field(..., description="Brief explanation of relevance")


class QueryIntent(BaseModel):
    """Parsed query intent from LLM."""

    reasoning: str = Field(..., description="Step-by-step thinking about what the user wants")
    query_en: str = Field(..., description="Optimized English search query")
    query_zh: Optional[str] = Field(None, description="Chinese search query (if applicable)")
    keywords: list[str] = Field(default_factory=list, description="Key terms to search (English only, 1-3 words each)")
    synonyms: dict[str, list[str]] = Field(
        default_factory=dict, description="Synonyms/abbreviations mapping"
    )
    required_phrases: list[str] = Field(
        default_factory=list, description="Phrases that must appear"
    )
    exclude_terms: list[str] = Field(
        default_factory=list, description="Terms to exclude from results"
    )
    # Filter fields for year and venue
    year: Optional[int] = Field(None, description="Exact publication year filter (e.g., 2025)")
    year_min: Optional[int] = Field(None, description="Minimum publication year (inclusive)")
    year_max: Optional[int] = Field(None, description="Maximum publication year (inclusive)")
    venue: Optional[str] = Field(None, description="Journal or conference name filter (e.g., 'Bioinformatics', 'Nature')")


class PlatformQueryResult(BaseModel):
    """Result of platform-specific query generation."""

    platform: str = Field(..., description="Target platform: pubmed, scholar, wos")
    platform_query: str = Field(..., description="The final search query string for the platform")
    notes: str = Field(
        default="",
        description="Platform-specific tips or notes (e.g., filter suggestions, syntax hints)"
    )
    intent: QueryIntent = Field(..., description="The underlying query intent")


class ArticleHighlights(BaseModel):
    """Extracted highlights from an article for slide generation."""

    title: str = Field(..., description="Main title for the slide")
    subtitle: Optional[str] = Field(None, description="Optional subtitle or context")
    bullets: list[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="Key highlight points (1-8 bullets)",
    )
    takeaway: str = Field(..., description="One-sentence conclusion or main takeaway")
    keywords: list[str] = Field(
        default_factory=list,
        description="Optional keywords for visual element hints",
    )


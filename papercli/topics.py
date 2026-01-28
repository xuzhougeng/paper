"""Topic analysis for paper collections."""

import re
from collections import Counter
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from papercli.models import Paper

if TYPE_CHECKING:
    from papercli.cache import Cache
    from papercli.llm import LLMClient


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class KeywordStats(BaseModel):
    """Statistics for extracted keywords."""

    top_keywords: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top keywords with their frequency counts",
    )
    top_bigrams: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top bigrams (2-word phrases) with their counts",
    )
    total_papers: int = Field(..., description="Total number of papers analyzed")
    papers_with_abstract: int = Field(..., description="Number of papers with abstracts")


class Topic(BaseModel):
    """A single research topic extracted from papers."""

    name: str = Field(..., description="Short descriptive name for the topic (2-5 words)")
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms associated with this topic",
    )
    summary: str = Field(..., description="Brief summary of this topic (1-2 sentences)")
    paper_count: int = Field(..., description="Number of papers in this topic")
    representative_papers: list[str] = Field(
        default_factory=list,
        description="Titles of 2-3 representative papers for this topic",
    )


class TopicAnalysis(BaseModel):
    """Complete topic analysis result."""

    topics: list[Topic] = Field(
        default_factory=list,
        description="List of identified topics",
    )
    overall_summary: str = Field(
        ...,
        description="High-level summary of the research landscape (2-3 sentences)",
    )
    methodology_trends: list[str] = Field(
        default_factory=list,
        description="Common methodological approaches observed",
    )
    emerging_themes: list[str] = Field(
        default_factory=list,
        description="Emerging or novel themes identified",
    )


# ---------------------------------------------------------------------------
# Stopwords and text processing
# ---------------------------------------------------------------------------

# Common academic stopwords to filter out
STOPWORDS = {
    # Common English words
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "shall", "can", "need", "dare", "ought", "used", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "it", "they", "them", "us", "me", "him", "her",
    "my", "our", "your", "his", "its", "their", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then", "once", "if",
    # Academic common words
    "study", "studies", "research", "paper", "papers", "article", "articles",
    "results", "result", "method", "methods", "approach", "approaches",
    "data", "analysis", "using", "used", "based", "show", "shows", "showed",
    "new", "novel", "proposed", "propose", "present", "presented", "presents",
    "however", "therefore", "thus", "moreover", "furthermore", "although",
    "conclusion", "conclusions", "introduction", "background", "objective", "objectives",
    "aim", "aims", "purpose", "goal", "goals", "findings", "finding",
    "significant", "significantly", "important", "importantly",
    "abstract", "keywords", "keyword",
}


def tokenize(text: str) -> list[str]:
    """
    Simple tokenization: lowercase, split on non-alphanumeric, filter stopwords.
    """
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    # Filter stopwords and short tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def extract_bigrams(tokens: list[str]) -> list[str]:
    """Extract bigrams from token list."""
    if len(tokens) < 2:
        return []
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


# ---------------------------------------------------------------------------
# Keyword Statistics
# ---------------------------------------------------------------------------


def extract_keywords_stats(
    papers: list[Paper],
    top_n: int = 30,
) -> KeywordStats:
    """
    Extract keyword statistics from a collection of papers.

    Analyzes titles and abstracts to find frequently occurring terms and phrases.

    Args:
        papers: List of papers to analyze
        top_n: Number of top keywords/bigrams to return

    Returns:
        KeywordStats with frequency information
    """
    all_tokens: list[str] = []
    all_bigrams: list[str] = []
    papers_with_abstract = 0

    for paper in papers:
        # Combine title and abstract
        text_parts = [paper.title]
        if paper.abstract:
            text_parts.append(paper.abstract)
            papers_with_abstract += 1

        text = " ".join(text_parts)
        tokens = tokenize(text)
        all_tokens.extend(tokens)
        all_bigrams.extend(extract_bigrams(tokens))

    # Count frequencies
    token_counts = Counter(all_tokens)
    bigram_counts = Counter(all_bigrams)

    # Get top N
    top_keywords = token_counts.most_common(top_n)
    top_bigrams = bigram_counts.most_common(top_n)

    return KeywordStats(
        top_keywords=top_keywords,
        top_bigrams=top_bigrams,
        total_papers=len(papers),
        papers_with_abstract=papers_with_abstract,
    )


# ---------------------------------------------------------------------------
# LLM-based Topic Analysis
# ---------------------------------------------------------------------------

TOPIC_ANALYSIS_SYSTEM_PROMPT = """You are an expert research analyst specializing in academic literature analysis.

Your task is to analyze a collection of research papers and identify the main topics/themes.

Guidelines:
1. Identify 5-15 distinct topics based on the papers' content
2. Each topic should be meaningful and represent a coherent research area
3. Assign papers to topics based on their titles and abstracts
4. Provide a clear, concise name for each topic (2-5 words)
5. List the most relevant keywords for each topic
6. Select 2-3 representative papers that best exemplify each topic
7. Identify any emerging or novel themes that appear in recent work
8. Note common methodological approaches across the collection

Be specific and scientific in your analysis. Avoid vague or overly broad topic names."""


async def analyze_topics_llm(
    papers: list[Paper],
    llm: "LLMClient",
    cache: Optional["Cache"] = None,
    num_topics: int = 10,
) -> TopicAnalysis:
    """
    Analyze papers using LLM to identify topics and themes.

    Args:
        papers: List of papers to analyze
        llm: LLM client for analysis
        cache: Optional cache for storing results
        num_topics: Target number of topics to identify (5-15)

    Returns:
        TopicAnalysis with identified topics and summaries
    """
    if not papers:
        return TopicAnalysis(
            topics=[],
            overall_summary="No papers provided for analysis.",
            methodology_trends=[],
            emerging_themes=[],
        )

    # Build cache key from paper titles
    import hashlib
    paper_hash = hashlib.md5(
        "|".join(sorted(p.title[:50] for p in papers)).encode()
    ).hexdigest()[:16]
    cache_key = f"topics:{paper_hash}:{num_topics}"

    if cache:
        cached = await cache.get(cache_key)
        if cached:
            return TopicAnalysis.model_validate(cached)

    # Prepare paper summaries for the prompt
    paper_summaries = []
    for i, paper in enumerate(papers[:100], 1):  # Limit to 100 papers for context
        summary = f"{i}. {paper.title}"
        if paper.year:
            summary += f" ({paper.year})"
        if paper.venue:
            summary += f" - {paper.venue}"
        if paper.abstract:
            # Truncate abstract to ~200 chars
            abstract_preview = paper.abstract[:200]
            if len(paper.abstract) > 200:
                abstract_preview += "..."
            summary += f"\n   Abstract: {abstract_preview}"
        paper_summaries.append(summary)

    papers_text = "\n\n".join(paper_summaries)

    prompt = f"""Analyze the following {len(papers)} research papers and identify the main topics/themes.

Target number of topics: {num_topics} (can vary between 5-15 based on the actual content)

Papers to analyze:
{papers_text}

Please identify:
1. **topics**: The main research topics, each with:
   - name: Short descriptive name (2-5 words)
   - keywords: Key terms for this topic (3-8 terms)
   - summary: Brief description (1-2 sentences)
   - paper_count: Approximate number of papers in this topic
   - representative_papers: 2-3 paper titles that best represent this topic

2. **overall_summary**: A high-level summary of the research landscape (2-3 sentences)

3. **methodology_trends**: Common methodological approaches you observe (list 3-6 trends)

4. **emerging_themes**: Any emerging or novel themes, especially from recent papers (list 2-5 themes)

Return your analysis in the specified JSON format."""

    # Use reasoning model for better analysis
    analysis = await llm.reasoning_completion(
        prompt=prompt,
        response_model=TopicAnalysis,
        system_prompt=TOPIC_ANALYSIS_SYSTEM_PROMPT,
    )

    # Cache result
    if cache:
        await cache.set(cache_key, analysis.model_dump())

    return analysis


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------


def format_topics_markdown(
    stats: KeywordStats,
    analysis: TopicAnalysis,
    venue_filter: str | None = None,
    year_filter: int | None = None,
) -> str:
    """
    Format topic analysis results as markdown.

    Args:
        stats: Keyword statistics
        analysis: LLM topic analysis
        venue_filter: Optional venue filter used
        year_filter: Optional year filter used

    Returns:
        Formatted markdown string
    """
    lines = ["# Topic Analysis Report", ""]

    # Metadata
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total papers analyzed**: {stats.total_papers}")
    lines.append(f"- **Papers with abstracts**: {stats.papers_with_abstract}")
    if year_filter:
        lines.append(f"- **Year filter**: {year_filter}")
    if venue_filter:
        lines.append(f"- **Venue filter**: {venue_filter}")
    lines.append("")

    # Overall summary
    lines.append("## Summary")
    lines.append("")
    lines.append(analysis.overall_summary)
    lines.append("")

    # Topics
    lines.append("## Identified Topics")
    lines.append("")
    for i, topic in enumerate(analysis.topics, 1):
        lines.append(f"### {i}. {topic.name}")
        lines.append("")
        lines.append(f"**Papers**: ~{topic.paper_count}")
        lines.append("")
        lines.append(f"**Keywords**: {', '.join(topic.keywords)}")
        lines.append("")
        lines.append(f"**Summary**: {topic.summary}")
        lines.append("")
        if topic.representative_papers:
            lines.append("**Representative papers**:")
            for paper_title in topic.representative_papers:
                lines.append(f"- {paper_title}")
        lines.append("")

    # Methodology trends
    if analysis.methodology_trends:
        lines.append("## Methodology Trends")
        lines.append("")
        for trend in analysis.methodology_trends:
            lines.append(f"- {trend}")
        lines.append("")

    # Emerging themes
    if analysis.emerging_themes:
        lines.append("## Emerging Themes")
        lines.append("")
        for theme in analysis.emerging_themes:
            lines.append(f"- {theme}")
        lines.append("")

    # Keyword statistics
    lines.append("## Keyword Statistics")
    lines.append("")
    lines.append("### Top Keywords")
    lines.append("")
    lines.append("| Keyword | Frequency |")
    lines.append("|---------|-----------|")
    for kw, count in stats.top_keywords[:20]:
        lines.append(f"| {kw} | {count} |")
    lines.append("")

    lines.append("### Top Phrases")
    lines.append("")
    lines.append("| Phrase | Frequency |")
    lines.append("|--------|-----------|")
    for phrase, count in stats.top_bigrams[:20]:
        lines.append(f"| {phrase} | {count} |")
    lines.append("")

    return "\n".join(lines)


def format_topics_json(
    stats: KeywordStats,
    analysis: TopicAnalysis,
    venue_filter: str | None = None,
    year_filter: int | None = None,
) -> dict:
    """
    Format topic analysis results as a JSON-serializable dict.

    Args:
        stats: Keyword statistics
        analysis: LLM topic analysis
        venue_filter: Optional venue filter used
        year_filter: Optional year filter used

    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "metadata": {
            "total_papers": stats.total_papers,
            "papers_with_abstract": stats.papers_with_abstract,
            "year_filter": year_filter,
            "venue_filter": venue_filter,
        },
        "overall_summary": analysis.overall_summary,
        "topics": [topic.model_dump() for topic in analysis.topics],
        "methodology_trends": analysis.methodology_trends,
        "emerging_themes": analysis.emerging_themes,
        "keyword_stats": {
            "top_keywords": [{"keyword": kw, "count": count} for kw, count in stats.top_keywords],
            "top_bigrams": [{"phrase": phrase, "count": count} for phrase, count in stats.top_bigrams],
        },
    }

"""Output formatting for papercli results."""

import json
from typing import Literal

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from papercli.models import EvalResult


def format_output(
    results: list[EvalResult],
    format: Literal["table", "json", "md"],
    top_n: int = 3,
) -> str | Table | Group:
    """
    Format evaluation results for output.

    Args:
        results: List of evaluation results
        format: Output format (table, json, md)
        top_n: Number of top results to show

    Returns:
        Formatted output (string for json/md, Rich object for table)
    """
    # Take top N
    results = results[:top_n]

    if not results:
        return "No papers found matching your query."

    if format == "json":
        return _format_json(results)
    elif format == "md":
        return _format_markdown(results)
    else:  # table
        return _format_table(results)


def _format_json(results: list[EvalResult]) -> str:
    """Format results as JSON."""
    output = []
    for i, result in enumerate(results, 1):
        paper = result.paper
        output.append({
            "rank": i,
            "title": paper.title,
            "score": result.score,
            "meets_need": result.meets_need,
            "evidence": {
                "quote": result.evidence_quote,
                "field": result.evidence_field,
            },
            "reason": result.short_reason,
            "year": paper.year,
            "authors": paper.authors[:5],  # Limit authors
            "venue": paper.venue,
            "doi": paper.doi,
            "url": paper.url,
            "source": paper.source,
        })

    return json.dumps(output, indent=2, ensure_ascii=False)


def _format_markdown(results: list[EvalResult]) -> str:
    """Format results as Markdown."""
    lines = ["# Search Results\n"]

    for i, result in enumerate(results, 1):
        paper = result.paper

        lines.append(f"## {i}. {paper.title}\n")

        # Metadata line
        meta_parts = []
        if paper.year:
            meta_parts.append(f"**Year:** {paper.year}")
        if paper.authors:
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
            meta_parts.append(f"**Authors:** {authors_str}")
        if paper.venue:
            meta_parts.append(f"**Venue:** {paper.venue}")

        if meta_parts:
            lines.append(" | ".join(meta_parts) + "\n")

        # Score
        lines.append(f"**Relevance Score:** {result.score:.1f}/10\n")

        # Evidence
        lines.append(f"> {result.evidence_quote}\n")
        lines.append(f"*— from {result.evidence_field}*\n")

        # Reason
        lines.append(f"**Why relevant:** {result.short_reason}\n")

        # Links
        links = []
        if paper.url:
            links.append(f"[View Paper]({paper.url})")
        if paper.doi:
            links.append(f"[DOI](https://doi.org/{paper.doi})")

        if links:
            lines.append(" | ".join(links) + "\n")

        lines.append("---\n")

    return "\n".join(lines)


def _format_table(results: list[EvalResult]) -> Group:
    """Format results as Rich table with panels."""
    panels = []

    for i, result in enumerate(results, 1):
        paper = result.paper

        # Build content
        content = []

        # Score line
        score_text = Text()
        score_text.append("Score: ", style="bold")
        score_color = "green" if result.score >= 7 else "yellow" if result.score >= 4 else "red"
        score_text.append(f"{result.score:.1f}/10", style=score_color)
        if result.meets_need:
            score_text.append(" ✓ Meets need", style="green")
        content.append(score_text)

        # Metadata
        meta_parts = []
        if paper.year:
            meta_parts.append(f"{paper.year}")
        if paper.authors:
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += " et al."
            meta_parts.append(authors_str)
        if paper.venue:
            venue_short = paper.venue[:40] + "..." if len(paper.venue or "") > 40 else paper.venue
            meta_parts.append(venue_short)

        if meta_parts:
            meta_text = Text(" | ".join(meta_parts), style="dim")
            content.append(meta_text)

        # Evidence quote
        evidence_text = Text()
        evidence_text.append("\n\"", style="dim")
        quote = result.evidence_quote[:150]
        if len(result.evidence_quote) > 150:
            quote += "..."
        evidence_text.append(quote, style="italic cyan")
        evidence_text.append("\"", style="dim")
        evidence_text.append(f" — {result.evidence_field}", style="dim")
        content.append(evidence_text)

        # Reason
        reason_text = Text()
        reason_text.append("\n")
        reason_text.append(result.short_reason, style="dim")
        content.append(reason_text)

        # URL
        if paper.url:
            url_text = Text()
            url_text.append("\n")
            url_text.append(paper.url, style="link dim blue")
            content.append(url_text)

        # Create panel
        title_short = paper.title[:80] + "..." if len(paper.title) > 80 else paper.title
        panel = Panel(
            Group(*content),
            title=f"[bold]{i}. {title_short}[/bold]",
            title_align="left",
            border_style="blue" if result.meets_need else "dim",
            padding=(0, 1),
        )
        panels.append(panel)

    return Group(*panels)


def format_summary(results: list[EvalResult], total_searched: int) -> Text:
    """Format a summary line for the search."""
    text = Text()
    text.append(f"Found ", style="dim")
    text.append(f"{len(results)}", style="bold")
    text.append(f" relevant papers from ", style="dim")
    text.append(f"{total_searched}", style="bold")
    text.append(" candidates", style="dim")
    return text


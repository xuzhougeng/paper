"""Output formatting for papercli results."""

import json
from typing import Literal, TypedDict

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from papercli.models import EvalResult, PlatformQueryResult


class SentenceCiteResult(TypedDict):
    sentence: str
    results: list[EvalResult]
    recommended: EvalResult | None
    error: str | None


def format_output(
    results: list[EvalResult],
    format: Literal["table", "json", "md"],
    top_n: int = 3,
    show_all: bool = False,
) -> str | Table | Group:
    """
    Format evaluation results for output.

    Args:
        results: List of evaluation results
        format: Output format (table, json, md)
        top_n: Number of top results to show
        show_all: If True, display all papers without LLM evaluation details

    Returns:
        Formatted output (string for json/md, Rich object for table)
    """
    # Take top N
    results = results[:top_n]

    if not results:
        return ""  # Empty results handled by pipeline

    if format == "json":
        return _format_json(results, show_all)
    elif format == "md":
        return _format_markdown(results, show_all)
    else:  # table
        return _format_table(results, show_all)


def format_citation_report(
    sentence_results: list[SentenceCiteResult],
    sources: list[str],
    top_k: int,
) -> str:
    """Format sentence-level citation results as Markdown."""
    lines = ["# Citation Report", ""]

    if sources:
        lines.append(f"**Sources:** {', '.join(sources)}")
    lines.append(f"**Top-K per sentence:** {top_k}")
    lines.append("")

    for index, item in enumerate(sentence_results, 1):
        sentence = item["sentence"]
        lines.append(f"## Sentence {index}")
        lines.append(sentence)
        lines.append("")

        error = item.get("error")
        if error:
            lines.append(f"**Error:** {error}")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        results = item.get("results") or []
        if not results:
            lines.append("**No results found.**")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        recommended = item.get("recommended") or results[0]
        rec_paper = recommended.paper
        link = ""
        if rec_paper.doi:
            link = f"https://doi.org/{rec_paper.doi}"
        elif rec_paper.url:
            link = rec_paper.url

        lines.append("**Recommended**")
        if link:
            lines.append(f"- {rec_paper.title} ({link})")
        else:
            lines.append(f"- {rec_paper.title}")
        lines.append("")
        lines.append(f"> {recommended.evidence_quote}")
        lines.append(f"*— from {recommended.evidence_field}*")
        lines.append("")
        lines.append(f"**Why relevant:** {recommended.short_reason}")
        lines.append("")

        lines.append(f"**Candidates (top-{min(top_k, len(results))})**")
        for rank, result in enumerate(results, 1):
            paper = result.paper
            status = "yes" if result.meets_need else "no"
            link = ""
            if paper.doi:
                link = f"https://doi.org/{paper.doi}"
            elif paper.url:
                link = paper.url
            link_part = f" ({link})" if link else ""
            lines.append(
                f"- {rank}. {paper.title} (score: {result.score:.1f}, meets_need: {status}){link_part}"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip()


def _format_json(results: list[EvalResult], show_all: bool = False) -> str:
    """Format results as JSON."""
    output = []
    for i, result in enumerate(results, 1):
        paper = result.paper
        item = {
            "rank": i,
            "title": paper.title,
            "year": paper.year,
            "authors": paper.authors[:5],  # Limit authors
            "venue": paper.venue,
            "doi": paper.doi,
            "url": paper.url,
            "source": paper.source,
            "abstract": paper.abstract[:500] if paper.abstract else None,
        }
        # Include LLM evaluation details only when not show_all
        if not show_all:
            item.update({
                "score": result.score,
                "meets_need": result.meets_need,
                "evidence": {
                    "quote": result.evidence_quote,
                    "field": result.evidence_field,
                },
                "reason": result.short_reason,
            })
        output.append(item)

    return json.dumps(output, indent=2, ensure_ascii=False)


def _format_markdown(results: list[EvalResult], show_all: bool = False) -> str:
    """Format results as Markdown."""
    title = "# All Retrieved Papers\n" if show_all else "# Search Results\n"
    lines = [title]

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
        meta_parts.append(f"**Source:** {paper.source}")

        if meta_parts:
            lines.append(" | ".join(meta_parts) + "\n")

        # LLM evaluation details (only when not show_all)
        if not show_all:
            # Score
            lines.append(f"**Relevance Score:** {result.score:.1f}/10\n")

            # Evidence
            lines.append(f"> {result.evidence_quote}\n")
            lines.append(f"*— from {result.evidence_field}*\n")

            # Reason
            lines.append(f"**Why relevant:** {result.short_reason}\n")
        else:
            # Show abstract for show_all mode
            if paper.abstract:
                abstract_preview = paper.abstract[:300]
                if len(paper.abstract) > 300:
                    abstract_preview += "..."
                lines.append(f"> {abstract_preview}\n")

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


def _format_table(results: list[EvalResult], show_all: bool = False) -> Group:
    """Format results as Rich table with panels."""
    panels = []

    for i, result in enumerate(results, 1):
        paper = result.paper

        # Build content
        content = []

        if not show_all:
            # Score line (only when LLM evaluated)
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
        meta_parts.append(f"[{paper.source}]")

        if meta_parts:
            meta_text = Text(" | ".join(meta_parts), style="dim")
            content.append(meta_text)

        if not show_all:
            # Evidence quote (only when LLM evaluated)
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
        else:
            # Show abstract preview for show_all mode
            if paper.abstract:
                abstract_text = Text()
                abstract_text.append("\n")
                abstract_preview = paper.abstract[:200]
                if len(paper.abstract) > 200:
                    abstract_preview += "..."
                abstract_text.append(abstract_preview, style="dim")
                content.append(abstract_text)

        # URL
        if paper.url:
            url_text = Text()
            url_text.append("\n")
            url_text.append(paper.url, style="link dim blue")
            content.append(url_text)

        # DOI
        if paper.doi:
            doi_text = Text()
            doi_text.append("\n")
            doi_text.append("DOI: ", style="dim")
            doi_text.append(paper.doi, style=f"dim cyan link https://doi.org/{paper.doi}")
            content.append(doi_text)

        # Create panel
        title_short = paper.title[:80] + "..." if len(paper.title) > 80 else paper.title
        panel = Panel(
            Group(*content),
            title=f"[bold]{i}. {title_short}[/bold]",
            title_align="left",
            border_style="blue" if (not show_all and result.meets_need) else "dim",
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


# ============================================================================
# Platform Query Output Formatting
# ============================================================================

def format_query_output(
    result: PlatformQueryResult,
    format: Literal["table", "json", "md"],
) -> str | Panel:
    """
    Format platform query generation result for output.

    Args:
        result: The platform query generation result
        format: Output format (table, json, md)

    Returns:
        Formatted output (string for json/md, Rich Panel for table)
    """
    if format == "json":
        return _format_query_json(result)
    elif format == "md":
        return _format_query_markdown(result)
    else:  # table
        return _format_query_table(result)


def _format_query_json(result: PlatformQueryResult) -> str:
    """Format platform query result as JSON."""
    output = {
        "platform": result.platform,
        "platform_query": result.platform_query,
        "notes": result.notes,
        "intent": {
            "query_en": result.intent.query_en,
            "query_zh": result.intent.query_zh,
            "keywords": result.intent.keywords,
            "synonyms": result.intent.synonyms,
            "required_phrases": result.intent.required_phrases,
            "exclude_terms": result.intent.exclude_terms,
        },
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def _format_query_markdown(result: PlatformQueryResult) -> str:
    """Format platform query result as Markdown."""
    lines = [f"# Search Query for {result.platform.upper()}\n"]

    # The main query - easy to copy
    lines.append("## Search Query\n")
    lines.append("```")
    lines.append(result.platform_query)
    lines.append("```\n")

    # Notes
    if result.notes:
        lines.append("## Notes\n")
        lines.append(f"{result.notes}\n")

    # Intent details
    lines.append("## Query Analysis\n")
    lines.append(f"**English Query:** {result.intent.query_en}\n")
    
    if result.intent.query_zh:
        lines.append(f"**Chinese Query:** {result.intent.query_zh}\n")

    if result.intent.keywords:
        lines.append(f"**Keywords:** {', '.join(result.intent.keywords)}\n")

    if result.intent.synonyms:
        lines.append("**Synonyms:**\n")
        for term, syns in result.intent.synonyms.items():
            lines.append(f"- {term}: {', '.join(syns)}")
        lines.append("")

    if result.intent.required_phrases:
        lines.append(f"**Required Phrases:** {', '.join(f'\"{p}\"' for p in result.intent.required_phrases)}\n")

    if result.intent.exclude_terms:
        lines.append(f"**Exclude Terms:** {', '.join(result.intent.exclude_terms)}\n")

    return "\n".join(lines)


def _format_query_table(result: PlatformQueryResult) -> Panel:
    """Format platform query result as Rich Panel."""
    content = Text()

    # Platform name
    platform_display = {
        "pubmed": "PubMed",
        "scholar": "Google Scholar",
        "wos": "Web of Science",
    }.get(result.platform, result.platform.upper())

    # Main query - prominent display
    content.append("🔍 ", style="bold")
    content.append("Search Query:\n", style="bold green")
    content.append(f"   {result.platform_query}\n", style="bold white")

    # Notes
    if result.notes and "[Fallback" not in result.notes:
        content.append("\n💡 ", style="bold")
        content.append("Tips: ", style="bold yellow")
        content.append(f"{result.notes}\n", style="dim")
    elif result.notes:
        content.append("\n⚠️  ", style="bold")
        content.append(f"{result.notes}\n", style="yellow")

    content.append("\n")

    # Keywords
    if result.intent.keywords:
        content.append("🏷️  ", style="bold")
        content.append("Keywords: ", style="bold cyan")
        content.append(", ".join(result.intent.keywords[:10]), style="cyan")
        if len(result.intent.keywords) > 10:
            content.append(f" (+{len(result.intent.keywords) - 10} more)", style="dim")
        content.append("\n")

    # Synonyms
    if result.intent.synonyms:
        content.append("🔄 ", style="bold")
        content.append("Synonyms:\n", style="bold magenta")
        for term, syns in list(result.intent.synonyms.items())[:4]:
            syn_str = ", ".join(syns[:3])
            if len(syns) > 3:
                syn_str += "..."
            content.append(f"   {term} → {syn_str}\n", style="magenta")
        if len(result.intent.synonyms) > 4:
            content.append(f"   ... and {len(result.intent.synonyms) - 4} more\n", style="dim")

    # Required phrases
    if result.intent.required_phrases:
        content.append("📌 ", style="bold")
        content.append("Required: ", style="bold blue")
        content.append(", ".join(f'"{p}"' for p in result.intent.required_phrases), style="blue")
        content.append("\n")

    # Exclude terms
    if result.intent.exclude_terms:
        content.append("🚫 ", style="bold")
        content.append("Exclude: ", style="bold red")
        content.append(", ".join(result.intent.exclude_terms), style="red")
        content.append("\n")

    # Create panel
    panel = Panel(
        content,
        title=f"[bold]{platform_display} Query[/bold]",
        title_align="left",
        border_style="green",
        padding=(0, 1),
    )

    return panel


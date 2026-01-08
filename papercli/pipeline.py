"""Main pipeline for paper search."""

import asyncio
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

from papercli.models import EvalResult, Paper, QueryIntent

if TYPE_CHECKING:
    from papercli.config import Settings

console = Console()


def run_pipeline(
    query: str,
    sources: list[str],
    top_n: int,
    max_per_source: int,
    prefilter_k: int,
    settings: "Settings",
    verbose: bool = False,
    quiet: bool = False,
) -> list[EvalResult]:
    """
    Run the full paper search pipeline.

    Steps:
    1. LLM intent extraction / query rewriting
    2. Multi-source search
    3. Normalize and deduplicate
    4. Coarse ranking to prefilter_k
    5. LLM reranking with evidence extraction
    6. Return top_n results
    """
    return asyncio.run(
        _run_pipeline_async(
            query=query,
            sources=sources,
            top_n=top_n,
            max_per_source=max_per_source,
            prefilter_k=prefilter_k,
            settings=settings,
            verbose=verbose,
            quiet=quiet,
        )
    )


def _display_intent_analysis(intent: QueryIntent, show_progress: bool) -> None:
    """Display the intent analysis results."""
    if not show_progress:
        return

    # Build content for the panel
    content = Text()

    # Reasoning (thinking process)
    content.append("💭 ", style="bold")
    content.append("Thinking:\n", style="bold cyan")
    # Split reasoning into lines and indent
    reasoning_lines = intent.reasoning.strip().split("\n")
    for line in reasoning_lines:
        content.append(f"   {line.strip()}\n", style="dim")

    content.append("\n")

    # Rewritten query
    content.append("🔍 ", style="bold")
    content.append("Search Query:\n", style="bold green")
    content.append(f"   {intent.query_en}\n", style="bold white")

    # Chinese query if available
    if intent.query_zh:
        content.append(f"   ({intent.query_zh})\n", style="dim")

    content.append("\n")

    # Keywords
    if intent.keywords:
        content.append("🏷️  ", style="bold")
        content.append("Keywords: ", style="bold yellow")
        content.append(", ".join(intent.keywords[:8]), style="yellow")
        if len(intent.keywords) > 8:
            content.append(f" (+{len(intent.keywords) - 8} more)", style="dim")
        content.append("\n")

    # Synonyms
    if intent.synonyms:
        content.append("🔄 ", style="bold")
        content.append("Synonyms:\n", style="bold magenta")
        for term, syns in list(intent.synonyms.items())[:4]:
            syn_str = ", ".join(syns[:3])
            if len(syns) > 3:
                syn_str += "..."
            content.append(f"   {term} → {syn_str}\n", style="magenta")
        if len(intent.synonyms) > 4:
            content.append(f"   ... and {len(intent.synonyms) - 4} more\n", style="dim")

    # Required phrases
    if intent.required_phrases:
        content.append("📌 ", style="bold")
        content.append("Required: ", style="bold blue")
        content.append(", ".join(f'"{p}"' for p in intent.required_phrases), style="blue")
        content.append("\n")

    # Exclude terms
    if intent.exclude_terms:
        content.append("🚫 ", style="bold")
        content.append("Exclude: ", style="bold red")
        content.append(", ".join(intent.exclude_terms), style="red")
        content.append("\n")

    # Create panel
    panel = Panel(
        content,
        title="[bold]Query Analysis[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


async def _run_pipeline_async(
    query: str,
    sources: list[str],
    top_n: int,
    max_per_source: int,
    prefilter_k: int,
    settings: "Settings",
    verbose: bool = False,
    quiet: bool = False,
) -> list[EvalResult]:
    """Async implementation of the pipeline."""
    from papercli.cache import Cache
    from papercli.eval import rerank_with_llm
    from papercli.llm import LLMClient, LLMError
    from papercli.query import extract_intent
    from papercli.rank import coarse_rank, deduplicate

    # Initialize components
    cache = Cache(settings.cache_path) if settings.cache_enabled else None
    llm = LLMClient(settings)

    show_progress = not quiet

    # Step 1: Extract query intent (show spinner while processing)
    intent = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not show_progress,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Step 1/5: Analyzing query intent...", total=None)
            intent = await extract_intent(query, llm, cache)
            progress.update(task, description="[green]✓ Query analyzed")
    except LLMError as e:
        # Display actionable error message
        if show_progress:
            console.print(
                "[bold red]❌ Query analysis failed[/bold red]\n\n"
                f"[yellow]{e.args[0]}[/yellow]\n"
            )
            if e.model or e.base_url:
                console.print("[dim]Configuration:[/dim]")
                if e.model:
                    console.print(f"  • Model: [cyan]{e.model}[/cyan]")
                if e.base_url:
                    console.print(f"  • Base URL: [cyan]{e.base_url}[/cyan]")
                console.print()
            if e.original_exception:
                console.print(
                    f"[dim]Original error:[/dim] {type(e.original_exception).__name__}: {e.original_exception}\n"
                )
            if verbose and e.raw_responses:
                console.print("[dim]Raw LLM response(s):[/dim]")
                for i, resp in enumerate(e.raw_responses, 1):
                    truncated = resp[:1000] + "..." if len(resp) > 1000 else resp
                    console.print(f"  [dim][{i}][/dim] {truncated}\n")
            if not verbose:
                console.print("[dim]Run with --verbose for more details (including raw LLM responses).[/dim]")
        raise  # Re-raise for non-zero exit

    # Display intent analysis results
    _display_intent_analysis(intent, show_progress)

    # Continue with the rest of the pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        TaskProgressColumn(),
        console=console,
        disable=not show_progress,
        transient=True,
    ) as progress:
        # Main task for remaining progress (steps 2-5)
        main_task = progress.add_task("[cyan]Searching...", total=100)

        # Step 2: Multi-source search (0% -> 50%)
        progress.update(main_task, description=f"[cyan]Step 2/5: Searching {len(sources)} sources...")
        papers = await _search_all_sources(sources, intent, max_per_source, cache, verbose, progress)
        progress.update(main_task, completed=50)

        if not papers:
            progress.update(main_task, completed=100, description="[yellow]No papers found")
            if show_progress:
                console.print("[yellow]No papers found matching your query.[/yellow]")
            return []

        if verbose:
            console.print(f"  [dim]→ Found {len(papers)} papers from all sources[/dim]")

        # Step 3: Deduplicate (50% -> 60%)
        progress.update(main_task, description="[cyan]Step 3/5: Removing duplicates...")
        original_count = len(papers)
        papers = deduplicate(papers)
        progress.update(main_task, completed=60)

        if verbose:
            removed = original_count - len(papers)
            console.print(f"  [dim]→ Removed {removed} duplicates, {len(papers)} unique papers[/dim]")

        # Step 4: Coarse ranking (60% -> 70%)
        progress.update(main_task, description="[cyan]Step 4/5: Ranking candidates...")
        candidates = coarse_rank(papers, intent, k=prefilter_k)
        progress.update(main_task, completed=70)

        if verbose:
            console.print(f"  [dim]→ Selected top {len(candidates)} candidates for evaluation[/dim]")

        # Step 5: LLM reranking (70% -> 95%)
        progress.update(main_task, description=f"[cyan]Step 5/5: Evaluating {len(candidates)} papers with LLM...")
        results = await rerank_with_llm(
            query, candidates, llm, cache,
            progress_callback=lambda i, total: progress.update(
                main_task,
                completed=70 + int(25 * i / total),
                description=f"[cyan]Step 5/5: Evaluating paper {i}/{total}..."
            )
        )
        progress.update(main_task, completed=95)

        # Step 6: Return top_n (100%)
        progress.update(main_task, description="[cyan]Finalizing results...")
        results = sorted(results, key=lambda r: r.score, reverse=True)
        final_results = results[:top_n]
        progress.update(main_task, completed=100, description="[green]✓ Search complete!")

    # Print summary
    if show_progress:
        meets_need_count = sum(1 for r in final_results if r.meets_need)
        console.print(
            f"\n[dim]Found [bold]{len(final_results)}[/bold] relevant papers "
            f"([green]{meets_need_count}[/green] highly relevant) "
            f"from {len(papers)} candidates[/dim]\n"
        )

    return final_results


async def _search_all_sources(
    sources: list[str],
    intent: QueryIntent,
    max_per_source: int,
    cache: "Cache | None",
    verbose: bool,
    progress: Progress | None = None,
) -> list[Paper]:
    """Search all sources concurrently."""
    from papercli.sources.arxiv import ArxivSource
    from papercli.sources.openalex import OpenAlexSource
    from papercli.sources.pubmed import PubMedSource
    from papercli.sources.scholar import ScholarSource

    source_map = {
        "pubmed": PubMedSource,
        "openalex": OpenAlexSource,
        "scholar": ScholarSource,
        "arxiv": ArxivSource,
    }

    source_names = {
        "pubmed": "PubMed",
        "openalex": "OpenAlex",
        "scholar": "Google Scholar",
        "arxiv": "arXiv",
    }

    tasks = []
    active_sources = []
    for src_name in sources:
        if src_name in source_map:
            source = source_map[src_name](cache=cache)
            tasks.append(source.search(intent, max_results=max_per_source))
            active_sources.append(src_name)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    papers = []
    for i, result in enumerate(results):
        src_name = active_sources[i]
        display_name = source_names.get(src_name, src_name)

        if isinstance(result, Exception):
            if verbose:
                console.print(f"  [yellow]⚠ {display_name}: {result}[/yellow]")
        else:
            count = len(result)
            papers.extend(result)
            if verbose:
                console.print(f"  [dim]→ {display_name}: {count} papers[/dim]")

    return papers

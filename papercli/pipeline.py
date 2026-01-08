"""Main pipeline for paper search."""

import asyncio
from typing import TYPE_CHECKING

from rich.console import Console

from papercli.models import EvalResult, Paper

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
        )
    )


async def _run_pipeline_async(
    query: str,
    sources: list[str],
    top_n: int,
    max_per_source: int,
    prefilter_k: int,
    settings: "Settings",
    verbose: bool = False,
) -> list[EvalResult]:
    """Async implementation of the pipeline."""
    from papercli.cache import Cache
    from papercli.eval import rerank_with_llm
    from papercli.llm import LLMClient
    from papercli.query import extract_intent
    from papercli.rank import coarse_rank, deduplicate

    # Initialize components
    cache = Cache(settings.cache_path) if settings.cache_enabled else None
    llm = LLMClient(settings)

    # Step 1: Extract query intent
    if verbose:
        console.print("[dim]Extracting query intent...[/dim]")
    intent = await extract_intent(query, llm, cache)
    if verbose:
        console.print(f"[dim]Query intent: {intent.query_en}[/dim]")

    # Step 2: Multi-source search
    if verbose:
        console.print(f"[dim]Searching sources: {', '.join(sources)}...[/dim]")
    papers = await _search_all_sources(sources, intent, max_per_source, cache, verbose)
    if verbose:
        console.print(f"[dim]Found {len(papers)} papers from all sources[/dim]")

    if not papers:
        return []

    # Step 3: Deduplicate
    papers = deduplicate(papers)
    if verbose:
        console.print(f"[dim]After deduplication: {len(papers)} papers[/dim]")

    # Step 4: Coarse ranking
    candidates = coarse_rank(papers, intent, k=prefilter_k)
    if verbose:
        console.print(f"[dim]After coarse ranking: {len(candidates)} candidates[/dim]")

    # Step 5: LLM reranking
    if verbose:
        console.print("[dim]Reranking with LLM...[/dim]")
    results = await rerank_with_llm(query, candidates, llm, cache)

    # Step 6: Return top_n
    results = sorted(results, key=lambda r: r.score, reverse=True)
    return results[:top_n]


async def _search_all_sources(
    sources: list[str],
    intent: "QueryIntent",
    max_per_source: int,
    cache: "Cache | None",
    verbose: bool,
) -> list[Paper]:
    """Search all sources concurrently."""
    from papercli.models import QueryIntent
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

    tasks = []
    for src_name in sources:
        if src_name in source_map:
            source = source_map[src_name](cache=cache)
            tasks.append(source.search(intent, max_results=max_per_source))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    papers = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            if verbose:
                console.print(f"[yellow]Warning:[/yellow] {sources[i]} search failed: {result}")
        else:
            papers.extend(result)

    return papers


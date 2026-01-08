"""Command-line interface for papercli."""

from typing import Annotated, Optional

import typer
from rich.console import Console

from papercli import __version__

app = typer.Typer(
    name="paper",
    help="CLI tool for searching academic papers using LLM-powered query understanding and ranking.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"papercli version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """PaperCLI - Find relevant academic papers with a single sentence."""
    pass


@app.command()
def find(
    query: Annotated[str, typer.Argument(help="Your search query (a sentence describing what you're looking for)")],
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of top results to return")] = 3,
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Comma-separated list of sources: pubmed,openalex,scholar,arxiv"),
    ] = "pubmed,openalex,scholar",
    max_per_source: Annotated[
        int,
        typer.Option("--max-per-source", help="Maximum papers to fetch per source"),
    ] = 20,
    prefilter_k: Annotated[
        int,
        typer.Option("--prefilter-k", help="Number of candidates to send to LLM for reranking"),
    ] = 20,
    intent_model: Annotated[
        Optional[str],
        typer.Option("--intent-model", help="Model for query rewriting/intent extraction"),
    ] = None,
    eval_model: Annotated[
        Optional[str],
        typer.Option("--eval-model", help="Model for evaluation/reranking"),
    ] = None,
    llm_base_url: Annotated[
        Optional[str],
        typer.Option("--llm-base-url", help="Base URL for OpenAI-compatible API"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: table, json, or md"),
    ] = "table",
    cache_path: Annotated[
        Optional[str],
        typer.Option("--cache-path", help="Path to SQLite cache file"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable caching"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose output"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output"),
    ] = False,
) -> None:
    """
    Find academic papers relevant to your query.

    Example:
        paper find "CRISPR gene editing for cancer therapy"
    """
    from papercli.config import Settings
    from papercli.pipeline import run_pipeline

    # Parse sources
    source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]
    valid_sources = {"pubmed", "openalex", "scholar", "arxiv"}
    for src in source_list:
        if src not in valid_sources:
            console.print(f"[red]Error:[/red] Unknown source '{src}'. Valid: {', '.join(valid_sources)}")
            raise typer.Exit(1)

    # Validate output format
    if output_format not in ("table", "json", "md"):
        console.print("[red]Error:[/red] --format must be one of: table, json, md")
        raise typer.Exit(1)

    # Build settings
    settings = Settings(
        intent_model=intent_model,
        eval_model=eval_model,
        llm_base_url=llm_base_url,
        cache_path=cache_path,
        cache_enabled=not no_cache,
    )

    # Run the pipeline
    try:
        results = run_pipeline(
            query=query,
            sources=source_list,
            top_n=top_n,
            max_per_source=max_per_source,
            prefilter_k=prefilter_k,
            settings=settings,
            verbose=verbose,
            quiet=quiet,
        )

        # Output results
        from papercli.output import format_output
        output = format_output(results, output_format, top_n)
        console.print(output)

    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


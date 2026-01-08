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
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of top results to return")] = 5,
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Comma-separated list of sources: pubmed,openalex,scholar,arxiv"),
    ] = "pubmed,openalex,scholar",
    max_per_source: Annotated[
        int,
        typer.Option("--max-per-source", help="Maximum papers to fetch per source"),
    ] = 50,
    prefilter_k: Annotated[
        int,
        typer.Option("--prefilter-k", help="Number of candidates to send to LLM for reranking"),
    ] = 30,
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
    show_all: Annotated[
        bool,
        typer.Option("--show-all", "-a", help="Show all retrieved papers (before LLM reranking)"),
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
            show_all=show_all,
        )

        # Output results
        from papercli.output import format_output
        # When show_all, display all results; otherwise use top_n
        display_n = len(results) if show_all else top_n
        output = format_output(results, output_format, display_n, show_all=show_all)
        console.print(output)

    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def check(
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Comma-separated list of sources to check: pubmed,openalex,scholar,arxiv"),
    ] = "pubmed,openalex,arxiv",
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
) -> None:
    """
    Check if LLM and search APIs are working properly.

    Example:
        paper check
        paper check --sources pubmed,openalex
    """
    import asyncio
    from papercli.config import Settings

    settings = Settings(
        intent_model=intent_model,
        eval_model=eval_model,
        llm_base_url=llm_base_url,
    )

    # Parse sources
    source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]

    asyncio.run(_run_checks(settings, source_list))


async def _run_checks(settings: "Settings", sources: list[str]) -> None:
    """Run all API checks."""
    import time
    import httpx
    from rich.table import Table

    console.print("\n[bold cyan]🔍 PaperCLI API Health Check[/bold cyan]\n")

    # Display configuration
    console.print("[dim]Configuration:[/dim]")
    console.print(f"  Intent Model: [cyan]{settings.get_intent_model()}[/cyan]")
    console.print(f"  Eval Model:   [cyan]{settings.get_eval_model()}[/cyan]")
    console.print(f"  LLM Base URL: [cyan]{settings.llm.base_url}[/cyan]")
    console.print()

    results = []

    # Check LLM API
    console.print("[bold]Checking LLM API...[/bold]")
    llm_ok, llm_msg, llm_time = await _check_llm(settings)
    results.append(("LLM (Intent)", settings.get_intent_model(), llm_ok, llm_msg, llm_time))

    # Check search sources
    console.print("[bold]Checking Search APIs...[/bold]")

    if "pubmed" in sources:
        ok, msg, t = await _check_pubmed()
        results.append(("PubMed", "NCBI E-utilities", ok, msg, t))

    if "openalex" in sources:
        ok, msg, t = await _check_openalex()
        results.append(("OpenAlex", "api.openalex.org", ok, msg, t))

    if "arxiv" in sources:
        ok, msg, t = await _check_arxiv()
        results.append(("arXiv", "export.arxiv.org", ok, msg, t))

    if "scholar" in sources:
        ok, msg, t = await _check_scholar(settings)
        results.append(("Google Scholar", "SerpAPI", ok, msg, t))

    # Display results table
    console.print()
    table = Table(title="API Status", show_header=True, header_style="bold")
    table.add_column("Service", style="cyan")
    table.add_column("Endpoint", style="dim")
    table.add_column("Status")
    table.add_column("Message")
    table.add_column("Time", justify="right")

    all_ok = True
    for service, endpoint, ok, msg, t in results:
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAIL[/red]"
        if not ok:
            all_ok = False
        time_str = f"{t:.0f}ms" if t else "-"
        table.add_row(service, endpoint, status, msg, time_str)

    console.print(table)
    console.print()

    if all_ok:
        console.print("[bold green]✓ All checks passed![/bold green]")
    else:
        console.print("[bold yellow]⚠ Some checks failed. Please review the configuration.[/bold yellow]")


async def _check_llm(settings: "Settings") -> tuple[bool, str, float | None]:
    """Check LLM API connectivity."""
    import time
    try:
        from papercli.llm import LLMClient
        llm = LLMClient(settings)
        start = time.time()
        response = await llm.complete(
            prompt="Say 'OK' if you can read this.",
            model=settings.get_intent_model(),
            max_tokens=10,
        )
        elapsed = (time.time() - start) * 1000
        if response:
            return True, "Connected", elapsed
        return False, "Empty response", elapsed
    except Exception as e:
        error_msg = str(e)[:50]
        return False, error_msg, None


async def _check_pubmed() -> tuple[bool, str, float | None]:
    """Check PubMed API connectivity."""
    import time
    import httpx
    import socket
    try:
        # Resolve once so we can provide actionable diagnostics on connect failures.
        resolved_ip: str | None = None
        try:
            resolved_ip = socket.gethostbyname("eutils.ncbi.nlm.nih.gov")
        except Exception:
            resolved_ip = None

        start = time.time()
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed&retmode=json"
            response = await client.get(url)
            response.raise_for_status()
        elapsed = (time.time() - start) * 1000
        return True, "Connected", elapsed
    except httpx.TimeoutException:
        return False, "Connection timeout", None
    except httpx.ConnectError as e:
        detail = str(e) if str(e) else "Cannot connect to server"
        if resolved_ip:
            # 198.18.0.0/15 is reserved for benchmarking; seeing it here usually means DNS hijack/block.
            if resolved_ip.startswith("198.18."):
                detail = f"{detail} (DNS -> {resolved_ip}; possible DNS hijack/block)"
            else:
                detail = f"{detail} (DNS -> {resolved_ip})"
        return False, detail[:80], None
    except Exception as e:
        error_msg = str(e)[:50] if str(e) else "Connection failed"
        return False, error_msg, None


async def _check_openalex() -> tuple[bool, str, float | None]:
    """Check OpenAlex API connectivity."""
    import time
    import httpx
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            url = "https://api.openalex.org/works?per_page=1"
            response = await client.get(url)
            response.raise_for_status()
        elapsed = (time.time() - start) * 1000
        return True, "Connected", elapsed
    except Exception as e:
        error_msg = str(e)[:50] if str(e) else "Connection failed"
        return False, error_msg, None


async def _check_arxiv() -> tuple[bool, str, float | None]:
    """Check arXiv API connectivity."""
    import time
    import httpx
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            # Use HTTPS directly to avoid redirect
            url = "https://export.arxiv.org/api/query?search_query=all:test&max_results=1"
            response = await client.get(url)
            response.raise_for_status()
        elapsed = (time.time() - start) * 1000
        return True, "Connected", elapsed
    except Exception as e:
        error_msg = str(e)[:50] if str(e) else "Connection failed"
        return False, error_msg, None


async def _check_scholar(settings: "Settings") -> tuple[bool, str, float | None]:
    """Check Google Scholar (SerpAPI) connectivity."""
    import os
    api_key = settings.get_serpapi_key() or os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return False, "SERPAPI_API_KEY not configured", None

    import time
    import httpx
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            url = f"https://serpapi.com/account?api_key={api_key}"
            response = await client.get(url)
            response.raise_for_status()
        elapsed = (time.time() - start) * 1000
        return True, "Connected", elapsed
    except Exception as e:
        error_msg = str(e)[:50] if str(e) else "Connection failed"
        return False, error_msg, None


@app.command("gen-query")
def gen_query(
    query: Annotated[str, typer.Argument(help="Your search query (natural language)")],
    platform: Annotated[
        str,
        typer.Option("--platform", "-p", help="Target platform: pubmed, scholar, wos (or aliases)"),
    ] = "pubmed",
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: table, json, or md"),
    ] = "table",
    intent_model: Annotated[
        Optional[str],
        typer.Option("--intent-model", help="Model for query rewriting/intent extraction"),
    ] = None,
    llm_base_url: Annotated[
        Optional[str],
        typer.Option("--llm-base-url", help="Base URL for OpenAI-compatible API"),
    ] = None,
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
    Generate a platform-specific search query from natural language.

    Supported platforms:
    - pubmed: PubMed/MEDLINE (default)
    - scholar: Google Scholar
    - wos: Web of Science (aliases: web_of_science, world_of_knowledge)

    Examples:
        paper gen-query "CRISPR gene editing for cancer therapy"
        paper gen-query "single cell RNA velocity" --platform scholar
        paper gen-query "machine learning drug discovery" --platform wos --format md
    """
    import asyncio
    from papercli.config import Settings
    from papercli.query import PLATFORM_ALIASES, VALID_PLATFORMS

    # Validate platform
    normalized_platform = PLATFORM_ALIASES.get(platform.lower(), platform.lower())
    if normalized_platform not in VALID_PLATFORMS:
        console.print(
            f"[red]Error:[/red] Unknown platform '{platform}'. "
            f"Valid platforms: {', '.join(sorted(VALID_PLATFORMS))} "
            f"(aliases: {', '.join(sorted(PLATFORM_ALIASES.keys()))})"
        )
        raise typer.Exit(1)

    # Validate output format
    if output_format not in ("table", "json", "md"):
        console.print("[red]Error:[/red] --format must be one of: table, json, md")
        raise typer.Exit(1)

    # Build settings
    settings = Settings(
        intent_model=intent_model,
        llm_base_url=llm_base_url,
        cache_path=cache_path,
        cache_enabled=not no_cache,
    )

    # Run the generation
    try:
        result = asyncio.run(
            _run_gen_query(
                query=query,
                platform=normalized_platform,
                settings=settings,
                verbose=verbose,
                quiet=quiet,
            )
        )

        # Output result
        from papercli.output import format_query_output
        output = format_query_output(result, output_format)
        console.print(output)

    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


async def _run_gen_query(
    query: str,
    platform: str,
    settings: "Settings",
    verbose: bool = False,
    quiet: bool = False,
) -> "PlatformQueryResult":
    """Run the platform query generation."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from papercli.cache import Cache
    from papercli.llm import LLMClient
    from papercli.models import PlatformQueryResult
    from papercli.query import generate_platform_query

    # Initialize components
    cache = Cache(settings.cache_path) if settings.cache_enabled else None
    llm = LLMClient(settings)

    show_progress = not quiet

    platform_display = {
        "pubmed": "PubMed",
        "scholar": "Google Scholar",
        "wos": "Web of Science",
    }.get(platform, platform.upper())

    # Generate with progress spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not show_progress,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]Generating {platform_display} query...", total=None)
        result = await generate_platform_query(query, platform, llm, cache)
        progress.update(task, description=f"[green]✓ Query generated for {platform_display}")

    if verbose and show_progress:
        console.print(f"[dim]Intent model: {settings.get_intent_model()}[/dim]")
        console.print(f"[dim]LLM base URL: {settings.llm.base_url}[/dim]")
        console.print()

    return result


if __name__ == "__main__":
    app()


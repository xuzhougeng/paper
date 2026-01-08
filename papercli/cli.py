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
    import re
    try:
        from papercli.llm import LLMClient
        llm = LLMClient(settings)
        start = time.time()
        response = await llm.complete(
            prompt="Say 'OK' if you can read this.",
            model=settings.get_intent_model(),
            max_tokens=16,
        )
        elapsed = (time.time() - start) * 1000
        if response:
            return True, "Connected", elapsed
        return False, "Empty response", elapsed
    except Exception as e:
        # Keep this single-line so the Rich table stays readable.
        raw = str(e) if str(e) else repr(e)
        one_line = re.sub(r"\s+", " ", raw).strip()

        model = settings.get_intent_model()
        base_url = settings.llm.base_url
        hint = ""
        # Common failure mode: OpenAI-compatible proxies that don't support newer models/endpoints.
        if ("openai-proxy" in base_url.lower() or "proxy" in base_url.lower()) and model.startswith("gpt-5"):
            hint = "Hint: this proxy may not support gpt-5.* models; try LLM_BASE_URL=https://api.openai.com/v1 or use gpt-4o."

        msg = one_line
        if hint and hint not in msg:
            msg = f"{msg} | {hint}"

        # Avoid truncating too aggressively; Rich will wrap long cells anyway.
        return False, msg[:220], None


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


@app.command()
def extract(
    pdf_path: Annotated[str, typer.Argument(help="Path to the PDF file to extract")],
    out: Annotated[
        Optional[str],
        typer.Option("--out", "-o", help="Output file path (default: stdout)"),
    ] = None,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", help="Seconds between status polls"),
    ] = 2.0,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Maximum wait time in seconds"),
    ] = 900.0,
    include_raw: Annotated[
        bool,
        typer.Option("--include-raw/--no-include-raw", help="Include raw page data in output"),
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
    Extract text from a PDF using Doc2X and output as JSONL.

    Each line in the output is a JSON object representing one page with fields:
    - doc2x_uid: The Doc2X task ID
    - source_path: Original PDF file path
    - page_index: 0-based page index
    - page_no: 1-based page number
    - text: Extracted text content
    - raw_page: (optional) Raw page data if --include-raw is set

    Requires DOC2X_API_KEY environment variable or [doc2x] section in config.

    Examples:
        paper extract paper.pdf
        paper extract paper.pdf --out result.jsonl
        paper extract paper.pdf --include-raw --verbose
    """
    import asyncio
    from pathlib import Path

    from papercli.config import Settings

    pdf = Path(pdf_path)
    if not pdf.exists():
        console.print(f"[red]Error:[/red] PDF file not found: {pdf_path}")
        raise typer.Exit(1)

    if not pdf.suffix.lower() == ".pdf":
        console.print(f"[red]Error:[/red] File does not have .pdf extension: {pdf_path}")
        raise typer.Exit(1)

    # Build settings
    settings = Settings()

    # Validate Doc2X API key early
    try:
        settings.get_doc2x_api_key()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Run the extraction
    try:
        asyncio.run(
            _run_extract(
                pdf_path=pdf,
                out=out,
                poll_interval=poll_interval,
                timeout=timeout,
                include_raw=include_raw,
                settings=settings,
                verbose=verbose,
                quiet=quiet,
            )
        )
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def structure(
    jsonl_path: Annotated[
        str,
        typer.Argument(
            help="Path to page-level JSONL produced by `paper extract` (e.g., result.jsonl)"
        ),
    ],
    out: Annotated[
        Optional[str],
        typer.Option("--out", "-o", help="Output file path (default: stdout)"),
    ] = None,
    output_format: Annotated[
        Optional[str],
        typer.Option("--format", "-f", help="Output format: json or md (default: infer from --out, else json)"),
    ] = None,
    pretty: Annotated[
        bool,
        typer.Option("--pretty/--no-pretty", help="Pretty-print JSON output"),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Second-pass parsing: turn `result.jsonl` into database-friendly structured fields.

    Outputs a single JSON object with fields like:
    - title, abstract, methods, results, references, appendix
    - main_figures/main_tables and supp_figures/supp_tables

    Example:
        paper structure result.jsonl --out structured.json
    """
    import json
    from pathlib import Path
    import sys

    from papercli.structure import structure_from_jsonl_path

    try:
        structured = structure_from_jsonl_path(jsonl_path)

        fmt = (output_format or "").strip().lower() or None
        if fmt is None:
            if out:
                suffix = Path(out).suffix.lower()
                if suffix in {".md", ".markdown"}:
                    fmt = "md"
                else:
                    fmt = "json"
            else:
                fmt = "json"

        if fmt not in ("json", "md"):
            console.print("[red]Error:[/red] --format must be one of: json, md")
            raise typer.Exit(1)

        if fmt == "json":
            payload = structured.model_dump()
            text = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
        else:
            from papercli.structure import structured_paper_to_markdown

            text = structured_paper_to_markdown(structured)

        if out:
            out_path = Path(out)
            out_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            console.print(f"[green]✓ Wrote structured output to {out_path}[/green]")
        else:
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

        if verbose and structured.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for w in structured.warnings:
                console.print(f"- {w}")

    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


async def _run_extract(
    pdf_path: "Path",
    out: str | None,
    poll_interval: float,
    timeout: float,
    include_raw: bool,
    settings: "Settings",
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Run the PDF extraction workflow."""
    from pathlib import Path  # noqa: F811
    import sys  # noqa: F401

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    from papercli.doc2x import Doc2XClient, Doc2XError
    from papercli.extract import result_to_jsonl, write_jsonl

    show_progress = not quiet
    client = Doc2XClient(settings)

    try:
        if verbose and show_progress:
            console.print(f"[dim]Doc2X Base URL: {settings.doc2x.base_url}[/dim]")
            console.print(f"[dim]PDF: {pdf_path}[/dim]")
            console.print()

        # Progress tracking
        current_progress = 0

        def on_progress(progress: int) -> None:
            nonlocal current_progress
            current_progress = progress

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            disable=not show_progress,
            transient=True,
        ) as progress:
            # Step 1: Upload
            upload_task = progress.add_task("[cyan]Uploading PDF to Doc2X...", total=None)

            uid, upload_url = await client.preupload()
            if verbose and show_progress:
                console.print(f"[dim]Task UID: {uid}[/dim]")

            await client.upload_pdf(upload_url, pdf_path)
            progress.update(upload_task, description="[green]✓ PDF uploaded")

            # Step 2: Parse (with progress bar)
            parse_task = progress.add_task("[cyan]Parsing PDF...", total=100)

            def update_parse_progress(p: int) -> None:
                progress.update(parse_task, completed=p)

            result = await client.poll_until_complete(
                uid=uid,
                poll_interval=poll_interval,
                max_wait=timeout,
                on_progress=update_parse_progress,
            )
            progress.update(parse_task, completed=100, description="[green]✓ PDF parsed")

        # Step 3: Convert to JSONL
        source_path_str = str(pdf_path.resolve())

        if out:
            # Write to file
            out_path = Path(out)
            count = write_jsonl(result, uid, source_path_str, str(out_path), include_raw)
            if show_progress:
                console.print(f"[green]✓ Wrote {count} page(s) to {out_path}[/green]")
        else:
            # Write to stdout
            jsonl_output = result_to_jsonl(result, uid, source_path_str, include_raw)
            # Use sys.stdout directly to avoid Rich formatting
            import sys
            sys.stdout.write(jsonl_output)
            if jsonl_output and not jsonl_output.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

    except Doc2XError as e:
        raise RuntimeError(str(e)) from e
    finally:
        await client.close()


if __name__ == "__main__":
    app()


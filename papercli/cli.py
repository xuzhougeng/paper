"""Command-line interface for papercli."""

from typing import TYPE_CHECKING, Annotated, Literal, Optional

import typer
from pydantic import BaseModel, field_validator
from rich.console import Console

from papercli import __version__

if TYPE_CHECKING:
    from papercli.llm import LLMClient


# ---------------------------------------------------------------------------
# LLM-based sentence segmentation schema
# ---------------------------------------------------------------------------


class Segment(BaseModel):
    """A single discourse segment from LLM segmentation."""

    text: str
    relation_to_prev: Literal[
        "continuation", "contrast", "cause", "elaboration", "shift", "other"
    ] | None = None
    reason: str | None = None
    needs_citation: bool = True
    citation_reason: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Segment text must not be empty")
        return v.strip()

    @field_validator("needs_citation", mode="before")
    @classmethod
    def normalize_needs_citation(cls, v: object) -> bool:
        """
        Normalize needs_citation from various LLM outputs to bool.

        Accepts: true/false, yes/no, 1/0, "true"/"false", etc.
        Defaults to True if value is invalid (fail-safe: don't skip citations).
        """
        if v is None:
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("false", "no", "0", "n"):
                return False
            if s in ("true", "yes", "1", "y"):
                return True
        # Default to True (fail-safe: don't skip citations on unknown values)
        return True

    @field_validator("relation_to_prev", mode="before")
    @classmethod
    def normalize_relation_to_prev(cls, v: object) -> object:
        """
        Be tolerant to slightly-off labels from the LLM.

        The CLI currently only uses Segment.text, but strict validation here can
        fail the whole cite run if the model returns a near-miss like "result".
        """
        if v is None:
            return None
        if not isinstance(v, str):
            return "other"

        s = v.strip().lower()
        if not s:
            return None

        # Normalize common variants/synonyms into our supported enum.
        synonyms = {
            # continuation
            "continue": "continuation",
            "continued": "continuation",
            "cont": "continuation",
            "same": "continuation",
            # contrast
            "contradiction": "contrast",
            "opposition": "contrast",
            "however": "contrast",
            # cause
            "cause-effect": "cause",
            "cause_effect": "cause",
            "cause and effect": "cause",
            "effect": "cause",
            "because": "cause",
            "reason": "cause",
            # elaboration
            "detail": "elaboration",
            "details": "elaboration",
            "explanation": "elaboration",
            "example": "elaboration",
            # shift
            "transition": "shift",
            "topic shift": "shift",
            # other / frequent "extra" labels we see from LLMs
            "result": "other",
            "results": "other",
            "finding": "other",
            "findings": "other",
            "conclusion": "other",
            "summary": "other",
        }
        s = synonyms.get(s, s)

        allowed = {"continuation", "contrast", "cause", "elaboration", "shift", "other"}
        return s if s in allowed else "other"


class SegmentationResponse(BaseModel):
    """LLM response containing segmented text."""

    segments: list[Segment]

    @field_validator("segments")
    @classmethod
    def segments_must_not_be_empty(cls, v: list[Segment]) -> list[Segment]:
        if not v:
            raise ValueError("segments list must not be empty")
        return v

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


def _split_sentences_simple(text: str) -> list[str]:
    """Split text into sentences using Chinese/English punctuation and newlines.

    This is a simple fallback implementation. Prefer split_sentences_llm for
    discourse-aware segmentation.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        sentence = "".join(buffer).strip()
        if sentence:
            sentences.append(sentence)
        buffer.clear()

    for ch in normalized:
        if ch == "\n":
            flush()
            continue
        buffer.append(ch)
        if ch in "。！？!?.":
            flush()

    flush()
    return sentences


# ---------------------------------------------------------------------------
# LLM-based sentence segmentation prompt
# ---------------------------------------------------------------------------

_SEGMENTATION_SYSTEM_PROMPT = """\
You are an expert discourse analyst. Your task is to segment the given text into
meaningful discourse units and determine whether each segment needs a citation.

Guidelines for segmentation:
1. Segment by semantic/discourse boundaries, NOT just punctuation.
2. Group closely related statements that share a single claim or idea.
3. Separate statements that make distinct claims requiring different citations.
4. Consider discourse relations: continuation, contrast, cause-effect, elaboration.
5. Each segment should be a complete, citable unit of meaning.

Guidelines for citation judgment (needs_citation):
1. Set needs_citation=false for:
   - Author's own work/contributions ("We propose...", "We conducted...", "Our approach...")
   - Paper structure descriptions ("This paper is organized as follows...")
   - Transitional sentences that don't make factual claims
   - Common knowledge that requires no citation
2. Set needs_citation=true for:
   - Claims about others' research findings or conclusions
   - Specific methods/techniques from prior work
   - Domain facts, statistics, or consensus statements
   - Comparisons referencing other work

Return ONLY valid JSON matching the required schema."""

_SEGMENTATION_USER_PROMPT = """\
Segment the following text into discourse units and determine if each needs citation.

Text to segment:
---
{text}
---

Return JSON with a "segments" array. Each segment must have:
- "text": the exact text of this segment (required, non-empty)
- "needs_citation": whether this segment needs a literature citation (required, boolean)
- "citation_reason": brief explanation for the citation decision (optional, 1 sentence)
- "relation_to_prev": how this relates to the previous segment (optional; MUST be one of: continuation, contrast, cause, elaboration, shift, other; if unsure use "other")
- "reason": brief explanation for this segmentation choice (optional)"""


async def split_sentences_llm(
    text: str,
    llm_client: "LLMClient",
    model: str | None = None,
) -> list[str]:
    """
    Split text into discourse-aware segments using LLM.

    Args:
        text: The text to segment
        llm_client: Initialized LLMClient instance
        model: Optional model override (defaults to reasoning_model)

    Returns:
        List of segment texts

    Raises:
        LLMError: If LLM call fails or returns invalid JSON
        ValueError: If the response is missing segments or has empty text
    """
    from papercli.llm import LLMClient, LLMError

    prompt = _SEGMENTATION_USER_PROMPT.format(text=text)

    response = await llm_client.complete_json(
        prompt=prompt,
        response_model=SegmentationResponse,
        model=model,
        system_prompt=_SEGMENTATION_SYSTEM_PROMPT,
        temperature=0.3,  # Lower temperature for more consistent segmentation
        max_tokens=4000,
        retry_on_parse_error=True,
    )

    return [seg.text for seg in response.segments]


async def split_segments_llm(
    text: str,
    llm_client: "LLMClient",
    model: str | None = None,
) -> list[Segment]:
    """
    Split text into discourse-aware segments using LLM, with citation intent.

    This function returns full Segment objects including needs_citation
    and citation_reason fields, suitable for the cite command.

    Args:
        text: The text to segment
        llm_client: Initialized LLMClient instance
        model: Optional model override (defaults to reasoning_model)

    Returns:
        List of Segment objects with text, needs_citation, citation_reason, etc.

    Raises:
        LLMError: If LLM call fails or returns invalid JSON
        ValueError: If the response is missing segments or has empty text
    """
    from papercli.llm import LLMClient, LLMError

    prompt = _SEGMENTATION_USER_PROMPT.format(text=text)

    response = await llm_client.complete_json(
        prompt=prompt,
        response_model=SegmentationResponse,
        model=model,
        system_prompt=_SEGMENTATION_SYSTEM_PROMPT,
        temperature=0.3,  # Lower temperature for more consistent segmentation
        max_tokens=4000,
        retry_on_parse_error=True,
    )

    return response.segments


def _read_cite_input(input_file: Optional[str], input_text: Optional[str]) -> str:
    """Read cite input text from --in, --text, or stdin."""
    from pathlib import Path
    import sys

    if input_file and input_text:
        raise ValueError("Use only one of --in or --text, not both.")

    if input_text is not None:
        return input_text

    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return path.read_text(encoding="utf-8")

    if sys.stdin.isatty():
        raise ValueError("No input provided. Use --in, --text, or pipe text to stdin.")

    return sys.stdin.read()


@app.command()
def find(
    query: Annotated[str, typer.Argument(help="Your search query (a sentence describing what you're looking for)")],
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of top results to return")] = 5,
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Comma-separated list of sources: pubmed,openalex,scholar,arxiv,zotero"),
    ] = "pubmed,openalex,scholar",
    max_per_source: Annotated[
        int,
        typer.Option("--max-per-source", help="Maximum papers to fetch per source"),
    ] = 50,
    prefilter_k: Annotated[
        int,
        typer.Option("--prefilter-k", help="Number of candidates to send to LLM for reranking"),
    ] = 30,
    # Filter options
    year: Annotated[
        Optional[int],
        typer.Option("--year", "-y", help="Filter by exact publication year (e.g., 2025)"),
    ] = None,
    year_min: Annotated[
        Optional[int],
        typer.Option("--year-min", help="Filter by minimum publication year (inclusive)"),
    ] = None,
    year_max: Annotated[
        Optional[int],
        typer.Option("--year-max", help="Filter by maximum publication year (inclusive)"),
    ] = None,
    venue: Annotated[
        Optional[str],
        typer.Option("--venue", "-j", help="Filter by journal/venue name (e.g., 'Bioinformatics', 'Nature')"),
    ] = None,
    # Model options
    reasoning_model: Annotated[
        Optional[str],
        typer.Option("--reasoning-model", help="Model for reasoning tasks (query rewriting, intent extraction)"),
    ] = None,
    instinct_model: Annotated[
        Optional[str],
        typer.Option("--instinct-model", help="Model for instinct tasks (evaluation, reranking)"),
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

    Examples:
        paper find "CRISPR gene editing for cancer therapy"
        paper find "single cell RNA-seq" --year 2025 --venue Bioinformatics
        paper find "machine learning" --year-min 2020 --year-max 2025
    """
    from papercli.config import Settings
    from papercli.pipeline import run_pipeline

    # Parse sources
    source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]
    valid_sources = {"pubmed", "openalex", "scholar", "arxiv", "zotero"}
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
        reasoning_model=reasoning_model,
        instinct_model=instinct_model,
        llm_base_url=llm_base_url,
        cache_path=cache_path,
        cache_enabled=not no_cache,
    )

    # Validate year filter combinations
    if year is not None and (year_min is not None or year_max is not None):
        console.print("[red]Error:[/red] Cannot use --year with --year-min/--year-max. Use one or the other.")
        raise typer.Exit(1)

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
            # Filter parameters
            year=year,
            year_min=year_min,
            year_max=year_max,
            venue=venue,
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
def pubmed_dump(
    venue: Annotated[
        str,
        typer.Option("--venue", "-j", help="Journal name to filter (e.g., 'Bioinformatics')"),
    ],
    year: Annotated[
        Optional[int],
        typer.Option("--year", "-y", help="Filter by exact publication year (e.g., 2025)"),
    ] = None,
    year_min: Annotated[
        Optional[int],
        typer.Option("--year-min", help="Filter by minimum publication year (inclusive)"),
    ] = None,
    year_max: Annotated[
        Optional[int],
        typer.Option("--year-max", help="Filter by maximum publication year (inclusive)"),
    ] = None,
    query: Annotated[
        Optional[str],
        typer.Option("--query", help="Optional keyword query to narrow results"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="PubMed esearch batch size"),
    ] = 500,
    max_results: Annotated[
        Optional[int],
        typer.Option("--max-results", help="Maximum number of papers to return"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or jsonl"),
    ] = "jsonl",
    out: Annotated[
        Optional[str],
        typer.Option("--out", help="Write output to a file path"),
    ] = None,
    cache_path: Annotated[
        Optional[str],
        typer.Option("--cache-path", help="Path to SQLite cache file"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable caching"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output"),
    ] = False,
) -> None:
    """
    Dump PubMed results for a venue and year range without LLM reranking.

    Examples:
        paper pubmed-dump --venue Bioinformatics --year-min 2020 --year-max 2025
        paper pubmed-dump --venue Bioinformatics --year 2024 --format json --out bioinfo-2024.json
    """
    import asyncio
    import json
    from pathlib import Path

    from papercli.cache import Cache
    from papercli.config import Settings
    from papercli.models import QueryIntent
    from papercli.sources.pubmed import PubMedSource

    if output_format not in ("json", "jsonl"):
        console.print("[red]Error:[/red] --format must be one of: json, jsonl")
        raise typer.Exit(1)

    if year is not None and (year_min is not None or year_max is not None):
        console.print("[red]Error:[/red] Cannot use --year with --year-min/--year-max. Use one or the other.")
        raise typer.Exit(1)

    if batch_size <= 0:
        console.print("[red]Error:[/red] --batch-size must be a positive integer.")
        raise typer.Exit(1)

    settings = Settings(
        cache_path=cache_path,
        cache_enabled=not no_cache,
    )

    cache = None
    if settings.cache_enabled:
        cache = Cache(path=settings.cache.path, ttl_hours=settings.cache.ttl_hours)

    intent = QueryIntent(
        reasoning="Direct PubMed dump (no LLM).",
        query_en=(query or "").strip(),
        query_zh=None,
        keywords=[],
        synonyms={},
        required_phrases=[],
        exclude_terms=[],
        year=year,
        year_min=year_min,
        year_max=year_max,
        venue=venue,
    )

    source = PubMedSource(cache=cache, api_key=settings.api_keys.ncbi_api_key)

    progress = None
    task_id = None
    if not quiet:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

        console.print("[cyan]Fetching PubMed results...[/cyan]")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
        progress.start()
        task_id = progress.add_task("PubMed fetch", total=1)

    def _progress_cb(delta: int, total: Optional[int]) -> None:
        if progress is None or task_id is None:
            return
        if total is not None:
            progress.update(task_id, total=total)
        if delta:
            progress.advance(task_id, delta)

    try:
        papers = asyncio.run(
            source.search_all(
                intent=intent,
                batch_size=batch_size,
                max_results=max_results,
                progress_cb=_progress_cb,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    finally:
        if progress is not None:
            progress.stop()

    output_records = [
        {
            "title": p.title,
            "abstract": p.abstract,
            "year": p.year,
            "pmid": p.source_id,
            "doi": p.doi,
            "url": p.url,
            "venue": p.venue,
            "authors": p.authors,
        }
        for p in papers
    ]

    if output_format == "json":
        output_text = json.dumps(output_records, ensure_ascii=False, indent=2)
    else:
        output_text = "\n".join(
            json.dumps(record, ensure_ascii=False) for record in output_records
        )

    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output_text, encoding="utf-8")
        if not quiet:
            console.print(f"[green]Saved {len(output_records)} records to {path}[/green]")
    else:
        console.print(output_text)


@app.command()
def topics(
    query: Annotated[
        Optional[str],
        typer.Argument(help="Search query (optional if using --in)"),
    ] = None,
    input_file: Annotated[
        Optional[str],
        typer.Option("--in", "-i", help="Input JSON file with papers (from 'paper find --format json')"),
    ] = None,
    out: Annotated[
        str,
        typer.Option("--out", "-o", help="Output file path (default: topics.md)"),
    ] = "topics.md",
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: md or json"),
    ] = "md",
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Comma-separated list of sources: pubmed,openalex,scholar,arxiv,zotero"),
    ] = "pubmed,openalex,scholar",
    max_per_source: Annotated[
        int,
        typer.Option("--max-per-source", help="Maximum papers to fetch per source"),
    ] = 100,
    # Filter options
    year: Annotated[
        Optional[int],
        typer.Option("--year", "-y", help="Filter by exact publication year (e.g., 2025)"),
    ] = None,
    year_min: Annotated[
        Optional[int],
        typer.Option("--year-min", help="Filter by minimum publication year (inclusive)"),
    ] = None,
    year_max: Annotated[
        Optional[int],
        typer.Option("--year-max", help="Filter by maximum publication year (inclusive)"),
    ] = None,
    venue: Annotated[
        Optional[str],
        typer.Option("--venue", "-j", help="Filter by journal/venue name (e.g., 'Bioinformatics', 'Nature')"),
    ] = None,
    num_topics: Annotated[
        int,
        typer.Option("--num-topics", help="Target number of topics to identify (5-15)"),
    ] = 10,
    # Model options
    reasoning_model: Annotated[
        Optional[str],
        typer.Option("--reasoning-model", help="Model for topic analysis"),
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
    Analyze publication topics from search results or a JSON file.

    Two modes of operation:
    1. Direct search: paper topics "query" --year 2025 --venue Bioinformatics
    2. From file: paper topics --in papers.json

    Examples:
        paper topics "bioinformatics methods" --year 2025 --venue Bioinformatics
        paper topics --in papers.json --format json --out analysis.json
        paper find "RNA-seq" --show-all --format json > papers.json && paper topics --in papers.json
    """
    import asyncio
    import json
    from pathlib import Path

    from papercli.config import Settings
    from papercli.models import Paper, EvalResult

    # Validate input mode
    if not query and not input_file:
        console.print("[red]Error:[/red] Either provide a search query or use --in with a JSON file.")
        raise typer.Exit(1)

    if query and input_file:
        console.print("[red]Error:[/red] Cannot use both query and --in. Use one or the other.")
        raise typer.Exit(1)

    # Validate output format
    if output_format not in ("md", "json"):
        console.print("[red]Error:[/red] --format must be one of: md, json")
        raise typer.Exit(1)

    # Validate year filter combinations
    if year is not None and (year_min is not None or year_max is not None):
        console.print("[red]Error:[/red] Cannot use --year with --year-min/--year-max. Use one or the other.")
        raise typer.Exit(1)

    # Validate num_topics
    if not 3 <= num_topics <= 20:
        console.print("[red]Error:[/red] --num-topics must be between 3 and 20")
        raise typer.Exit(1)

    # Build settings
    settings = Settings(
        reasoning_model=reasoning_model,
        llm_base_url=llm_base_url,
        cache_path=cache_path,
        cache_enabled=not no_cache,
    )

    show_progress = not quiet

    try:
        asyncio.run(
            _run_topics(
                query=query,
                input_file=input_file,
                out=out,
                output_format=output_format,
                sources=sources,
                max_per_source=max_per_source,
                year=year,
                year_min=year_min,
                year_max=year_max,
                venue=venue,
                num_topics=num_topics,
                settings=settings,
                verbose=verbose,
                show_progress=show_progress,
            )
        )
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


async def _run_topics(
    query: str | None,
    input_file: str | None,
    out: str,
    output_format: str,
    sources: str,
    max_per_source: int,
    year: int | None,
    year_min: int | None,
    year_max: int | None,
    venue: str | None,
    num_topics: int,
    settings: "Settings",
    verbose: bool,
    show_progress: bool,
) -> None:
    """Run the topic analysis workflow."""
    import json
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from papercli.cache import Cache
    from papercli.llm import LLMClient
    from papercli.models import Paper, EvalResult
    from papercli.pipeline import run_pipeline_async, apply_filters
    from papercli.topics import (
        extract_keywords_stats,
        analyze_topics_llm,
        format_topics_markdown,
        format_topics_json,
    )

    papers: list[Paper] = []

    # Mode 1: Load from JSON file
    if input_file:
        if show_progress:
            console.print(f"[cyan]Loading papers from {input_file}...[/cyan]")

        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        data = json.loads(path.read_text(encoding="utf-8"))

        # Handle different JSON formats
        if isinstance(data, list):
            # List of papers or EvalResults
            for item in data:
                if "paper" in item:
                    # EvalResult format
                    papers.append(Paper.model_validate(item["paper"]))
                else:
                    # Direct Paper format
                    papers.append(Paper.model_validate(item))
        elif isinstance(data, dict) and "results" in data:
            # Results wrapper format
            for item in data["results"]:
                if "paper" in item:
                    papers.append(Paper.model_validate(item["paper"]))
                else:
                    papers.append(Paper.model_validate(item))

        if show_progress:
            console.print(f"[green]✓ Loaded {len(papers)} papers[/green]")

        # Apply filters if specified
        if year or year_min or year_max or venue:
            from papercli.models import QueryIntent
            filter_intent = QueryIntent(
                reasoning="",
                query_en="",
                year=year,
                year_min=year_min,
                year_max=year_max,
                venue=venue,
            )
            pre_filter = len(papers)
            papers = apply_filters(papers, filter_intent)
            if show_progress and pre_filter != len(papers):
                console.print(f"[dim]Filtered: {pre_filter} → {len(papers)} papers[/dim]")

    # Mode 2: Search and fetch papers
    else:
        if show_progress:
            console.print(f"[cyan]Searching for papers: {query}[/cyan]")

        # Parse sources
        source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]
        valid_sources = {"pubmed", "openalex", "scholar", "arxiv", "zotero"}
        for src in source_list:
            if src not in valid_sources:
                raise ValueError(f"Unknown source '{src}'. Valid: {', '.join(valid_sources)}")

        # Run the pipeline with show_all to get all papers (no LLM reranking)
        results = await run_pipeline_async(
            query=query,
            sources=source_list,
            top_n=max_per_source * len(source_list),  # Get all
            max_per_source=max_per_source,
            prefilter_k=max_per_source * len(source_list),
            settings=settings,
            verbose=verbose,
            quiet=not show_progress,
            show_all=True,  # Skip LLM reranking
            year=year,
            year_min=year_min,
            year_max=year_max,
            venue=venue,
        )

        papers = [r.paper for r in results]

    if not papers:
        console.print("[yellow]No papers found to analyze.[/yellow]")
        return

    if show_progress:
        console.print(f"\n[cyan]Analyzing {len(papers)} papers...[/cyan]\n")

    # Initialize components
    cache = Cache(settings.get_cache_path()) if settings.cache_enabled else None
    llm = LLMClient(settings)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not show_progress,
            transient=True,
        ) as progress:
            # Step 1: Extract keyword statistics
            task = progress.add_task("[cyan]Extracting keyword statistics...", total=None)
            stats = extract_keywords_stats(papers)
            progress.update(task, description="[green]✓ Keywords extracted")

            # Step 2: LLM topic analysis
            progress.update(task, description="[cyan]Analyzing topics with LLM...")
            analysis = await analyze_topics_llm(papers, llm, cache, num_topics=num_topics)
            progress.update(task, description="[green]✓ Topic analysis complete")

        # Format output
        if output_format == "md":
            output_text = format_topics_markdown(stats, analysis, venue_filter=venue, year_filter=year)
        else:
            output_data = format_topics_json(stats, analysis, venue_filter=venue, year_filter=year)
            output_text = json.dumps(output_data, ensure_ascii=False, indent=2)

        # Write output
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            output_text + ("\n" if not output_text.endswith("\n") else ""),
            encoding="utf-8",
        )

        if show_progress:
            console.print(f"[green]✓ Topic analysis saved to {out_path}[/green]")

            # Show summary
            console.print()
            console.print(f"[dim]Topics identified: {len(analysis.topics)}[/dim]")
            for i, topic in enumerate(analysis.topics[:5], 1):
                console.print(f"[dim]  {i}. {topic.name} ({topic.paper_count} papers)[/dim]")
            if len(analysis.topics) > 5:
                console.print(f"[dim]  ... and {len(analysis.topics) - 5} more[/dim]")

    finally:
        await llm.close()


@app.command()
def cite(
    input_file: Annotated[
        Optional[str],
        typer.Option("--in", "-i", help="Input text file (reads from stdin if not provided)"),
    ] = None,
    input_text: Annotated[
        Optional[str],
        typer.Option("--text", help="Input text to split into sentences"),
    ] = None,
    out: Annotated[
        str,
        typer.Option("--out", "-o", help="Output report path (default: report.md)"),
    ] = "report.md",
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Top citations to keep per sentence"),
    ] = 3,
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Comma-separated list of sources: pubmed,openalex,scholar,arxiv,zotero"),
    ] = "pubmed,openalex,scholar",
    max_per_source: Annotated[
        int,
        typer.Option("--max-per-source", help="Maximum papers to fetch per source"),
    ] = 50,
    prefilter_k: Annotated[
        int,
        typer.Option("--prefilter-k", help="Number of candidates to send to LLM for reranking"),
    ] = 30,
    reasoning_model: Annotated[
        Optional[str],
        typer.Option("--reasoning-model", help="Model for reasoning tasks (query rewriting, segmentation)"),
    ] = None,
    instinct_model: Annotated[
        Optional[str],
        typer.Option("--instinct-model", help="Model for instinct tasks (evaluation, reranking)"),
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
    Split a paragraph into sentences and find citations for each sentence.

    Examples:
        paper cite --text "Sentence one. Sentence two."
        paper cite --in notes.txt --out report.md --top-k 3
        cat notes.txt | paper cite --top-k 2
    """
    from pathlib import Path
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    from papercli.config import Settings
    from papercli.output import format_citation_report
    from papercli.pipeline import run_pipeline

    if top_k < 1:
        console.print("[red]Error:[/red] --top-k must be >= 1")
        raise typer.Exit(1)

    # Parse sources
    source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]
    valid_sources = {"pubmed", "openalex", "scholar", "arxiv", "zotero"}
    for src in source_list:
        if src not in valid_sources:
            console.print(f"[red]Error:[/red] Unknown source '{src}'. Valid: {', '.join(valid_sources)}")
            raise typer.Exit(1)

    if prefilter_k < top_k:
        if verbose and not quiet:
            console.print(
                f"[dim]Adjusting --prefilter-k from {prefilter_k} to {top_k} to match top-k[/dim]"
            )
        prefilter_k = top_k

    # Read input text
    try:
        text = _read_cite_input(input_file, input_text)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read input: {e}")
        raise typer.Exit(1)

    # Build settings (needed for LLM client)
    settings = Settings(
        reasoning_model=reasoning_model,
        instinct_model=instinct_model,
        llm_base_url=llm_base_url,
        cache_path=cache_path,
        cache_enabled=not no_cache,
    )

    # Segment text using LLM
    import asyncio
    from papercli.llm import LLMClient, LLMError

    llm = LLMClient(settings)

    show_progress = not quiet
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not show_progress,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Segmenting text with LLM...", total=None)
            segments = asyncio.run(
                split_segments_llm(text, llm, model=reasoning_model)
            )
            needs_cite_count = sum(1 for seg in segments if seg.needs_citation)
            progress.update(
                task, description=f"[green]✓ Segmented into {len(segments)} units ({needs_cite_count} need citations)"
            )
    except LLMError as e:
        console.print(f"[red]Error:[/red] LLM segmentation failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Segmentation failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    if not segments:
        console.print("[red]Error:[/red] No segments found in input")
        raise typer.Exit(1)

    # Run per-segment searches (skip segments that don't need citations)
    show_progress = not quiet
    sentence_results = []
    total = len(segments)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        TaskProgressColumn(),
        console=console,
        disable=not show_progress,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Finding citations...", total=total)

        for index, seg in enumerate(segments, 1):
            sentence = seg.text
            if show_progress:
                preview = sentence.strip()
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                progress.update(
                    task,
                    description=f"[cyan]Segment {index}/{total}: {preview}",
                )

            # Skip citation search for segments that don't need citations
            if not seg.needs_citation:
                sentence_results.append(
                    {
                        "sentence": sentence,
                        "results": [],
                        "recommended": None,
                        "error": None,
                        "needs_citation": False,
                        "citation_reason": seg.citation_reason,
                    }
                )
                progress.advance(task, 1)
                continue

            try:
                results = run_pipeline(
                    query=sentence,
                    sources=source_list,
                    top_n=top_k,
                    max_per_source=max_per_source,
                    prefilter_k=prefilter_k,
                    settings=settings,
                    verbose=verbose,
                    quiet=True,
                    show_all=False,
                )
                recommended = None
                if results:
                    recommended = max(results, key=lambda r: (r.meets_need, r.score))
                sentence_results.append(
                    {
                        "sentence": sentence,
                        "results": results,
                        "recommended": recommended,
                        "error": None,
                        "needs_citation": True,
                        "citation_reason": seg.citation_reason,
                    }
                )
            except Exception as e:
                error_text = str(e).strip() or repr(e)
                sentence_results.append(
                    {
                        "sentence": sentence,
                        "results": [],
                        "recommended": None,
                        "error": error_text,
                        "needs_citation": True,
                        "citation_reason": seg.citation_reason,
                    }
                )

            progress.advance(task, 1)

        progress.update(task, description="[green]✓ Citation report complete")

    report = format_citation_report(sentence_results, source_list, top_k)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + ("\n" if not report.endswith("\n") else ""), encoding="utf-8")

    if show_progress:
        console.print(f"[green]✓ Wrote citation report to {out_path}[/green]")


@app.command()
def check(
    sources: Annotated[
        str,
        typer.Option("--sources", "-s", help="Comma-separated list of sources to check: pubmed,openalex,scholar,arxiv,zotero"),
    ] = "pubmed,openalex,arxiv",
    reasoning_model: Annotated[
        Optional[str],
        typer.Option("--reasoning-model", help="Model for reasoning tasks (query rewriting, intent extraction)"),
    ] = None,
    instinct_model: Annotated[
        Optional[str],
        typer.Option("--instinct-model", help="Model for instinct tasks (evaluation, reranking)"),
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
        reasoning_model=reasoning_model,
        instinct_model=instinct_model,
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
    console.print(f"  Reasoning Model: [cyan]{settings.get_reasoning_model()}[/cyan]")
    console.print(f"  Instinct Model:  [cyan]{settings.get_instinct_model()}[/cyan]")
    console.print(f"  LLM Base URL:    [cyan]{settings.llm.base_url}[/cyan]")
    console.print()

    results = []

    # Check LLM API
    console.print("[bold]Checking LLM API...[/bold]")
    llm_ok, llm_msg, llm_time = await _check_llm(settings)
    results.append(("LLM (Reasoning)", settings.get_reasoning_model(), llm_ok, llm_msg, llm_time))

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

    if "zotero" in sources:
        ok, msg, t = await _check_zotero(settings)
        results.append(("Zotero", "api.zotero.org", ok, msg, t))

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
            model=settings.get_reasoning_model(),
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

        model = settings.get_reasoning_model()
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


async def _check_zotero(settings: "Settings") -> tuple[bool, str, float | None]:
    """Check Zotero API connectivity."""
    import os
    api_key = settings.get_zotero_api_key() or os.environ.get("ZOTERO_API_KEY")
    user_id = settings.get_zotero_user_id() or os.environ.get("ZOTERO_USER_ID")

    if not api_key:
        return False, "ZOTERO_API_KEY not configured", None
    if not user_id:
        return False, "ZOTERO_USER_ID not configured", None

    import time
    import httpx
    try:
        start = time.time()
        headers = {
            "Zotero-API-Key": api_key,
            "Zotero-API-Version": "3",
        }
        base_url = settings.zotero.base_url or "https://api.zotero.org"
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            url = f"{base_url}/users/{user_id}/items?limit=1"
            response = await client.get(url)
            response.raise_for_status()
        elapsed = (time.time() - start) * 1000
        return True, "Connected", elapsed
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return False, "Invalid API key or no access", None
        elif e.response.status_code == 404:
            return False, "User ID not found", None
        error_msg = str(e)[:50] if str(e) else "Connection failed"
        return False, error_msg, None
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
    reasoning_model: Annotated[
        Optional[str],
        typer.Option("--reasoning-model", help="Model for reasoning tasks (query rewriting, intent extraction)"),
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
        reasoning_model=reasoning_model,
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
    cache = Cache(settings.get_cache_path()) if settings.cache_enabled else None
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
        console.print(f"[dim]Reasoning model: {settings.get_reasoning_model()}[/dim]")
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
    image_dir: Annotated[
        Optional[str],
        typer.Option("--image-dir", "-i", help="Directory to download images (replaces CDN URLs with local paths)"),
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
        paper extract paper.pdf --image-dir ./images --out result.jsonl
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
                image_dir=image_dir,
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


@app.command("fetch-pdf")
def fetch_pdf(
    doi: Annotated[str, typer.Argument(help="DOI to fetch PDF for (e.g., 10.1038/nature12373)")],
    out_dir: Annotated[
        str,
        typer.Option("--out-dir", "-d", help="Directory to save PDF (default: current directory)"),
    ] = ".",
    filename: Annotated[
        Optional[str],
        typer.Option("--filename", "-f", help="Output filename (default: {doi_safe}.pdf)"),
    ] = None,
    skip_unpaywall: Annotated[
        bool,
        typer.Option("--skip-unpaywall", help="Skip Unpaywall lookup (use PMC only)"),
    ] = False,
    skip_pmc: Annotated[
        bool,
        typer.Option("--skip-pmc", help="Skip PMC fallback (use Unpaywall only)"),
    ] = False,
    no_download: Annotated[
        bool,
        typer.Option("--no-download", help="Only show PDF URL, don't download"),
    ] = False,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format for metadata: json or table"),
    ] = "table",
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
    Fetch PDF for a DOI using Unpaywall and PMC.

    Searches Unpaywall first for open access PDFs, then falls back to
    PubMed Central (PMC) if no direct PDF link is found.

    Requires UNPAYWALL_EMAIL environment variable or [unpaywall] email
    in config file for Unpaywall API access.

    Examples:
        paper fetch-pdf 10.1038/nature12373
        paper fetch-pdf "10.1038/s41586-023-06291-2" --out-dir ./pdfs
        paper fetch-pdf 10.1000/xyz123 --no-download --format json
    """
    import asyncio

    from papercli.config import Settings

    # Validate output format
    if output_format not in ("table", "json"):
        console.print("[red]Error:[/red] --format must be one of: table, json")
        raise typer.Exit(1)

    # Build settings
    settings = Settings()

    # Validate Unpaywall email early (unless skipping)
    if not skip_unpaywall:
        try:
            settings.get_unpaywall_email()
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Run the fetch
    try:
        asyncio.run(
            _run_fetch_pdf(
                doi=doi,
                out_dir=out_dir,
                filename=filename,
                skip_unpaywall=skip_unpaywall,
                skip_pmc=skip_pmc,
                no_download=no_download,
                output_format=output_format,
                settings=settings,
                verbose=verbose,
                quiet=quiet,
            )
        )
    except typer.Exit:
        # Preserve intended exit code and avoid printing noisy "Error: <code>".
        raise
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


async def _run_fetch_pdf(
    doi: str,
    out_dir: str,
    filename: str | None,
    skip_unpaywall: bool,
    skip_pmc: bool,
    no_download: bool,
    output_format: str,
    settings: "Settings",
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Run the PDF fetch workflow."""
    import json
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from papercli.pdf_fetch import (
        fetch_pdf_url,
        download_pdf,
        doi_to_filename,
        validate_doi,
        PDFFetchError,
    )

    show_progress = not quiet

    # Validate DOI
    try:
        normalized_doi = validate_doi(doi)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if verbose and show_progress:
        console.print(f"[dim]DOI: {normalized_doi}[/dim]")
        console.print(f"[dim]Unpaywall: {'skip' if skip_unpaywall else 'enabled'}[/dim]")
        console.print(f"[dim]PMC fallback: {'skip' if skip_pmc else 'enabled'}[/dim]")
        console.print()

    # Fetch PDF URL
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not show_progress,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Looking up PDF URL...", total=None)

        result = await fetch_pdf_url(
            doi=normalized_doi,
            settings=settings,
            skip_unpaywall=skip_unpaywall,
            skip_pmc=skip_pmc,
        )

        if result.pdf_url:
            progress.update(task, description=f"[green]✓ Found PDF via {result.source}")
        else:
            progress.update(task, description="[yellow]⚠ No direct PDF found")

    # Display result
    if output_format == "json":
        output = result.model_dump(exclude_none=True)
        console.print(json.dumps(output, indent=2))
    else:
        # Table format
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="dim")
        table.add_column("Value")

        table.add_row("DOI", result.doi)
        table.add_row("Source", result.source)

        if result.pdf_url:
            table.add_row("PDF URL", f"[green]{result.pdf_url}[/green]")
        if result.landing_url:
            table.add_row("Landing URL", result.landing_url)
        if result.oa_status:
            table.add_row("OA Status", result.oa_status)
        if result.license:
            table.add_row("License", result.license)
        if result.pmid:
            table.add_row("PMID", result.pmid)
        if result.pmcid:
            table.add_row("PMCID", result.pmcid)
        if result.error:
            table.add_row("Note", f"[yellow]{result.error}[/yellow]")

        console.print(table)

    # Download if we have a PDF URL and not --no-download
    if result.pdf_url and not no_download:
        out_path = Path(out_dir)
        if filename:
            pdf_path = out_path / filename
        else:
            pdf_path = out_path / doi_to_filename(normalized_doi)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not show_progress,
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]Downloading to {pdf_path}...", total=None)

            try:
                await download_pdf(result.pdf_url, pdf_path)
                progress.update(task, description=f"[green]✓ Downloaded to {pdf_path}")
                if show_progress:
                    console.print(f"\n[green]✓ PDF saved to {pdf_path}[/green]")
            except PDFFetchError as e:
                progress.update(task, description=f"[red]✗ Download failed: {e}[/red]")
                console.print(f"\n[red]Error downloading PDF:[/red] {e}")
                raise typer.Exit(2)

    elif not result.pdf_url:
        if result.landing_url:
            console.print(f"\n[yellow]No direct PDF available. Try visiting:[/yellow] {result.landing_url}")
        raise typer.Exit(1)


async def _run_extract(
    pdf_path: "Path",
    out: str | None,
    image_dir: str | None,
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
    from papercli.extract import (
        result_to_jsonl,
        write_jsonl,
        collect_all_image_urls,
        download_images,
        replace_urls_in_result,
    )

    show_progress = not quiet
    client = Doc2XClient(settings)

    try:
        if verbose and show_progress:
            console.print(f"[dim]Doc2X Base URL: {settings.doc2x.base_url}[/dim]")
            console.print(f"[dim]PDF: {pdf_path}[/dim]")
            if image_dir:
                console.print(f"[dim]Image directory: {image_dir}[/dim]")
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

            # Step 3: Download images if requested
            url_mapping: dict[str, str] = {}
            if image_dir:
                image_urls = collect_all_image_urls(result)
                if image_urls:
                    img_task = progress.add_task(
                        f"[cyan]Downloading {len(image_urls)} image(s)...",
                        total=len(image_urls),
                    )

                    def update_img_progress(downloaded: int, total: int) -> None:
                        progress.update(img_task, completed=downloaded)

                    url_mapping = await download_images(
                        urls=image_urls,
                        image_dir=Path(image_dir),
                        uid=uid,
                        on_progress=update_img_progress,
                    )
                    progress.update(
                        img_task,
                        completed=len(image_urls),
                        description=f"[green]✓ Downloaded {len(url_mapping)}/{len(image_urls)} image(s)",
                    )

                    # Replace URLs in result
                    result = replace_urls_in_result(result, url_mapping)

        # Step 4: Convert to JSONL
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


@app.command()
def slide(
    input_file: Annotated[
        Optional[str],
        typer.Option("--in", "-i", help="Input text file (reads from stdin if not provided)"),
    ] = None,
    out: Annotated[
        str,
        typer.Option("--out", "-o", help="Output PNG file path"),
    ] = "slide.png",
    style: Annotated[
        str,
        typer.Option("--style", "-s", help="Slide style: handdrawn, minimal, academic, dark, colorful"),
    ] = "handdrawn",
    bullets: Annotated[
        int,
        typer.Option("--bullets", "-b", help="Number of bullet points (1-8)"),
    ] = 5,
    aspect_ratio: Annotated[
        str,
        typer.Option("--aspect-ratio", help="Image aspect ratio: 16:9, 4:3, 1:1"),
    ] = "16:9",
    image_size: Annotated[
        str,
        typer.Option("--image-size", help="Image size: 1K, 2K, 4K"),
    ] = "1K",
    cache_path: Annotated[
        Optional[str],
        typer.Option("--cache-path", help="Path to SQLite cache file"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable caching"),
    ] = False,
    show_highlights: Annotated[
        bool,
        typer.Option("--show-highlights", help="Print extracted highlights to stdout"),
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
    Generate a visual slide summarizing article highlights.

    Reads text from a file or stdin, extracts key highlights using Gemini,
    and generates a styled single-page slide image.

    Styles:
    - handdrawn: Hand-drawn sketch style with marker strokes
    - minimal: Ultra-minimalist clean design
    - academic: Professional academic poster style
    - dark: Dark futuristic tech theme
    - colorful: Vibrant and energetic design

    Requires GEMINI_API_KEY environment variable or [gemini] section in config.

    Examples:
        paper slide --in article.txt --style handdrawn
        paper slide --in paper.txt --out summary.png --style academic
        cat article.txt | paper slide --style minimal
        paper slide --in text.txt --bullets 3 --image-size 2K
    """
    import asyncio
    import os
    from pathlib import Path

    from papercli.config import Settings
    from papercli.slide import VALID_STYLES

    def _mask_secret(value: str | None) -> str:
        """Mask secrets for debug output without leaking full values."""
        if not value:
            return "<unset>"
        v = value.strip()
        if not v:
            return "<empty>"
        if len(v) <= 8:
            return f"{v[0]}***{v[-1]} (len={len(v)})"
        return f"{v[:4]}…{v[-4:]} (len={len(v)})"

    # Validate style
    if style not in VALID_STYLES:
        console.print(
            f"[red]Error:[/red] Unknown style '{style}'. "
            f"Valid styles: {', '.join(sorted(VALID_STYLES))}"
        )
        raise typer.Exit(1)

    # Validate aspect ratio
    valid_aspects = {"16:9", "4:3", "1:1"}
    if aspect_ratio not in valid_aspects:
        console.print(
            f"[red]Error:[/red] Invalid aspect ratio '{aspect_ratio}'. "
            f"Valid options: {', '.join(sorted(valid_aspects))}"
        )
        raise typer.Exit(1)

    # Validate image size
    valid_sizes = {"1K", "2K", "4K"}
    if image_size not in valid_sizes:
        console.print(
            f"[red]Error:[/red] Invalid image size '{image_size}'. "
            f"Valid options: {', '.join(sorted(valid_sizes))}"
        )
        raise typer.Exit(1)

    # Validate bullets range
    if not 1 <= bullets <= 8:
        console.print("[red]Error:[/red] --bullets must be between 1 and 8")
        raise typer.Exit(1)

    # Build settings
    settings = Settings(
        cache_path=cache_path,
        cache_enabled=not no_cache,
    )

    # Validate Gemini API key early
    try:
        settings.get_gemini_api_key()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if verbose and not quiet:
        cfg_path = settings.get_config_file_used()
        console.print(f"[dim]Config file: {cfg_path if cfg_path else '<none>'}[/dim]")
        console.print(
            f"[dim]GEMINI_API_KEY env: {_mask_secret(os.environ.get('GEMINI_API_KEY'))}[/dim]"
        )
        console.print(
            f"[dim]Gemini api_key (effective): {_mask_secret(settings.gemini.api_key)}[/dim]"
        )
        console.print()

    # Read input text
    try:
        text = _read_slide_input(input_file)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read input: {e}")
        raise typer.Exit(1)

    if not text.strip():
        console.print("[red]Error:[/red] Input text is empty")
        raise typer.Exit(1)

    # Run the slide generation
    try:
        asyncio.run(
            _run_slide(
                text=text,
                out=out,
                style=style,
                bullets=bullets,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                settings=settings,
                show_highlights=show_highlights,
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


def _read_slide_input(input_file: str | None) -> str:
    """Read input text from file or stdin."""
    import sys
    from pathlib import Path

    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return path.read_text(encoding="utf-8")
    else:
        # Read from stdin
        if sys.stdin.isatty():
            raise ValueError(
                "No input provided. Use --in to specify a file or pipe text to stdin."
            )
        return sys.stdin.read()


async def _run_slide(
    text: str,
    out: str,
    style: str,
    bullets: int,
    aspect_ratio: str,
    image_size: str,
    settings: "Settings",
    show_highlights: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Run the slide generation workflow."""
    import json
    from pathlib import Path

    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text

    from papercli.cache import Cache
    from papercli.gemini import GeminiClient, GeminiError
    from papercli.slide import generate_slide

    show_progress = not quiet

    # Initialize components
    cache = Cache(settings.get_cache_path()) if settings.cache_enabled else None
    client = GeminiClient(settings)

    try:
        if verbose and show_progress:
            console.print(f"[dim]Gemini Base URL: {settings.gemini.base_url}[/dim]")
            console.print(f"[dim]Text Model: {settings.gemini.text_model}[/dim]")
            console.print(f"[dim]Image Model: {settings.gemini.image_model}[/dim]")
            console.print(f"[dim]Style: {style}[/dim]")
            console.print(f"[dim]Input length: {len(text)} chars[/dim]")
            console.print()

        # Progress tracking
        current_status = ""

        def on_progress(status: str) -> None:
            nonlocal current_status
            current_status = status

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not show_progress,
            transient=True,
        ) as progress:
            # Step 1: Generate slide
            task = progress.add_task("[cyan]Generating slide...", total=None)

            image_bytes, highlights = await generate_slide(
                text=text,
                style=style,
                client=client,
                cache=cache,
                num_bullets=bullets,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                progress_callback=lambda s: progress.update(task, description=f"[cyan]{s}"),
            )

            progress.update(task, description="[green]✓ Slide generated")

        # Write output file
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(image_bytes)

        if show_progress:
            console.print(f"[green]✓ Slide saved to {out_path}[/green]")
            console.print(f"[dim]  Size: {len(image_bytes):,} bytes[/dim]")

        # Show highlights if requested
        if show_highlights:
            console.print()
            content = Text()
            content.append("📌 ", style="bold")
            content.append(f"{highlights.title}\n", style="bold cyan")
            if highlights.subtitle:
                content.append(f"   {highlights.subtitle}\n", style="dim")
            content.append("\n")

            for i, bullet in enumerate(highlights.bullets, 1):
                content.append(f"  {i}. ", style="bold yellow")
                content.append(f"{bullet}\n", style="white")

            content.append("\n💡 ", style="bold")
            content.append("Takeaway: ", style="bold green")
            content.append(highlights.takeaway, style="italic")

            panel = Panel(
                content,
                title="[bold]Extracted Highlights[/bold]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
            console.print(panel)

    except GeminiError as e:
        # Display actionable error message
        if show_progress:
            console.print("[bold red]❌ Slide generation failed[/bold red]\n")
            console.print(f"[yellow]{e.args[0]}[/yellow]\n")
            if e.model or e.base_url:
                console.print("[dim]Configuration:[/dim]")
                if e.model:
                    console.print(f"  • Model: [cyan]{e.model}[/cyan]")
                if e.base_url:
                    console.print(f"  • Base URL: [cyan]{e.base_url}[/cyan]")
                console.print()
            if e.status_code:
                console.print(f"[dim]HTTP Status: {e.status_code}[/dim]")
            if verbose and e.raw_response:
                console.print("[dim]Raw response:[/dim]")
                console.print(f"  {e.raw_response[:500]}")
            if not verbose:
                console.print("[dim]Run with --verbose for more details.[/dim]")
        raise
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Review command group
# ---------------------------------------------------------------------------

review_app = typer.Typer(
    name="review",
    help="Review, critique, and lint academic papers.",
    no_args_is_help=True,
)
app.add_typer(review_app, name="review")


def _read_review_input(input_file: Optional[str], input_text: Optional[str]) -> str:
    """Read review input text from --in, --text, or stdin."""
    from pathlib import Path
    import sys

    if input_file and input_text:
        raise ValueError("Use only one of --in or --text, not both.")

    if input_text is not None:
        return input_text

    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return path.read_text(encoding="utf-8")

    if sys.stdin.isatty():
        raise ValueError("No input provided. Use --in, --text, or pipe text to stdin.")

    return sys.stdin.read()


@review_app.command("critique")
def review_critique(
    input_file: Annotated[
        Optional[str],
        typer.Option("--in", "-i", help="Input text file (reads from stdin if not provided)"),
    ] = None,
    input_text: Annotated[
        Optional[str],
        typer.Option("--text", help="Input text to review"),
    ] = None,
    out: Annotated[
        str,
        typer.Option("--out", "-o", help="Output report path (default: review.md)"),
    ] = "review.md",
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: md or json"),
    ] = "md",
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model for review (defaults to instinct_model)"),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option("--temperature", "-t", help="Sampling temperature (lower = more consistent)"),
    ] = 0.3,
    llm_base_url: Annotated[
        Optional[str],
        typer.Option("--llm-base-url", help="Base URL for OpenAI-compatible API"),
    ] = None,
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
    Generate a structured peer review for a biology/biomedical paper.

    Uses LLM to analyze the manuscript and produce a comprehensive review
    covering strengths, weaknesses, experimental design, statistics,
    reproducibility, and more.

    Examples:
        paper review critique --in manuscript.txt
        paper review critique --in paper.md --format json --out review.json
        cat abstract.txt | paper review critique --out feedback.md
    """
    import asyncio
    import json
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from papercli.config import Settings
    from papercli.llm import LLMClient, LLMError
    from papercli.review.critique import run_critique, format_review_markdown

    # Validate output format
    if output_format not in ("md", "json"):
        console.print("[red]Error:[/red] --format must be one of: md, json")
        raise typer.Exit(1)

    # Read input text
    try:
        text = _read_review_input(input_file, input_text)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read input: {e}")
        raise typer.Exit(1)

    if not text.strip():
        console.print("[red]Error:[/red] Input text is empty")
        raise typer.Exit(1)

    # Build settings
    settings = Settings(
        instinct_model=model,
        llm_base_url=llm_base_url,
    )

    show_progress = not quiet

    if verbose and show_progress:
        console.print(f"[dim]Model: {model or settings.get_instinct_model()}[/dim]")
        console.print(f"[dim]Input length: {len(text)} chars ({len(text.split())} words)[/dim]")
        console.print()

    # Run critique
    llm = LLMClient(settings)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not show_progress,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing manuscript...", total=None)

            review = asyncio.run(
                run_critique(
                    text=text,
                    llm_client=llm,
                    model=model or settings.get_instinct_model(),
                    temperature=temperature,
                )
            )

            progress.update(task, description="[green]✓ Review complete")

        # Format output
        if output_format == "json":
            output_text = json.dumps(review.model_dump(), ensure_ascii=False, indent=2)
        else:
            output_text = format_review_markdown(review)

        # Write output
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            output_text + ("\n" if not output_text.endswith("\n") else ""),
            encoding="utf-8",
        )

        if show_progress:
            console.print(f"[green]✓ Review saved to {out_path}[/green]")

            # Show summary
            console.print()
            console.print(f"[dim]Strengths: {len(review.strengths)}[/dim]")
            console.print(f"[dim]Weaknesses: {len(review.weaknesses)}[/dim]")
            console.print(f"[dim]Major concerns: {len(review.major_concerns)}[/dim]")
            console.print(f"[dim]Questions: {len(review.questions_for_authors)}[/dim]")
            if review.overall_recommendation:
                rec = review.overall_recommendation.replace("_", " ").title()
                console.print(f"[dim]Recommendation: {rec}[/dim]")

    except LLMError as e:
        if show_progress:
            console.print("[bold red]❌ Review failed[/bold red]\n")
            console.print(f"[yellow]{e.args[0]}[/yellow]\n")
            if e.model or e.base_url:
                console.print("[dim]Configuration:[/dim]")
                if e.model:
                    console.print(f"  • Model: [cyan]{e.model}[/cyan]")
                if e.base_url:
                    console.print(f"  • Base URL: [cyan]{e.base_url}[/cyan]")
            if not verbose:
                console.print("[dim]Run with --verbose for more details.[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        asyncio.run(llm.close())


@review_app.command("lint")
def review_lint(
    input_file: Annotated[
        Optional[str],
        typer.Option("--in", "-i", help="Input text file (reads from stdin if not provided)"),
    ] = None,
    input_text: Annotated[
        Optional[str],
        typer.Option("--text", help="Input text to lint"),
    ] = None,
    out: Annotated[
        Optional[str],
        typer.Option("--out", "-o", help="Output report path (default: stdout)"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: md or json"),
    ] = "md",
    rules: Annotated[
        Optional[str],
        typer.Option("--rules", "-r", help="Comma-separated list of rules to run (default: all)"),
    ] = None,
    exclude: Annotated[
        Optional[str],
        typer.Option("--exclude", "-x", help="Comma-separated list of rules to exclude"),
    ] = None,
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
    Run deterministic lint checks on paper text.

    Checks for:
    - figure-table-ref: Inconsistent Fig./Figure and Tab./Table references
    - units-format: Number-unit spacing and percentage formatting
    - term-consistency: Inconsistent biology term usage (RNA-seq, CRISPR-Cas9, etc.)
    - punctuation-mixed: Chinese/English punctuation mixing
    - case-title: Inconsistent heading case style

    Examples:
        paper review lint --in manuscript.md
        paper review lint --in paper.txt --format json
        paper review lint --in draft.md --rules figure-table-ref,units-format
        cat text.txt | paper review lint --exclude punctuation-mixed
    """
    import json
    import sys
    from pathlib import Path

    from papercli.review.lint import run_lint, format_lint_markdown, RULES

    # Validate output format
    if output_format not in ("md", "json"):
        console.print("[red]Error:[/red] --format must be one of: md, json")
        raise typer.Exit(1)

    # Parse rules
    rule_list = None
    if rules:
        rule_list = [r.strip() for r in rules.split(",") if r.strip()]
        # Validate rules
        invalid = [r for r in rule_list if r not in RULES]
        if invalid:
            console.print(f"[red]Error:[/red] Unknown rules: {', '.join(invalid)}")
            console.print(f"[dim]Available rules: {', '.join(RULES.keys())}[/dim]")
            raise typer.Exit(1)

    exclude_list = None
    if exclude:
        exclude_list = [r.strip() for r in exclude.split(",") if r.strip()]

    # Read input text
    try:
        text = _read_review_input(input_file, input_text)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read input: {e}")
        raise typer.Exit(1)

    if not text.strip():
        console.print("[red]Error:[/red] Input text is empty")
        raise typer.Exit(1)

    show_progress = not quiet

    if verbose and show_progress:
        console.print(f"[dim]Input length: {len(text)} chars ({len(text.splitlines())} lines)[/dim]")
        if rule_list:
            console.print(f"[dim]Rules: {', '.join(rule_list)}[/dim]")
        else:
            console.print(f"[dim]Rules: all ({', '.join(RULES.keys())})[/dim]")
        if exclude_list:
            console.print(f"[dim]Excluded: {', '.join(exclude_list)}[/dim]")
        console.print()

    # Run lint
    report = run_lint(text, rules=rule_list, exclude_rules=exclude_list)

    # Format output
    if output_format == "json":
        output_text = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
    else:
        output_text = format_lint_markdown(report)

    # Write output
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            output_text + ("\n" if not output_text.endswith("\n") else ""),
            encoding="utf-8",
        )
        if show_progress:
            console.print(f"[green]✓ Lint report saved to {out_path}[/green]")
    else:
        sys.stdout.write(output_text)
        if not output_text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    # Summary
    if show_progress and out:
        if report.issues:
            console.print()
            console.print(f"[dim]Found {len(report.issues)} issues:[/dim]")
            console.print(f"[dim]  Errors: {report.error_count}[/dim]")
            console.print(f"[dim]  Warnings: {report.warn_count}[/dim]")
            console.print(f"[dim]  Info: {report.info_count}[/dim]")
        else:
            console.print("[dim]No issues found![/dim]")

    # Exit with error code if there are errors
    if report.error_count > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

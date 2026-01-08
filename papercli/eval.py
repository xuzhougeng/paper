"""LLM-based reranking and evidence extraction."""

import asyncio
from typing import TYPE_CHECKING, Callable, Optional

from pydantic import BaseModel, Field

from papercli.models import EvalResult, Paper

if TYPE_CHECKING:
    from papercli.cache import Cache
    from papercli.llm import LLMClient

# Type for progress callback: (current_index, total) -> None
ProgressCallback = Callable[[int, int], None]

EVAL_SYSTEM_PROMPT = """You are an expert at evaluating academic paper relevance. Given a user's search query and a paper's title and abstract, you must:

1. Score the paper's relevance from 0-10 (10 = perfect match)
2. Determine if the paper meets the user's need (true/false)
3. Extract the most relevant quote from the title or abstract that supports why this paper is relevant
4. Identify which field the quote came from (title or abstract)
5. Provide a brief reason for your assessment

Be strict but fair. A paper should score high only if it directly addresses the user's query.
The evidence_quote must be an EXACT substring from the title or abstract - do not paraphrase."""


class PaperEvaluation(BaseModel):
    """LLM evaluation result for a single paper."""

    score: float = Field(..., ge=0, le=10, description="Relevance score 0-10")
    meets_need: bool = Field(..., description="Whether paper meets user's need")
    evidence_quote: str = Field(..., description="Exact quote from title/abstract")
    evidence_field: str = Field(..., description="'title' or 'abstract'")
    short_reason: str = Field(..., max_length=200, description="Brief explanation")


async def rerank_with_llm(
    query: str,
    papers: list[Paper],
    llm: "LLMClient",
    cache: Optional["Cache"] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[EvalResult]:
    """
    Rerank papers using LLM evaluation.

    Args:
        query: User's original query
        papers: List of papers to evaluate
        llm: LLM client
        cache: Optional cache
        progress_callback: Optional callback for progress updates (current, total)

    Returns:
        List of EvalResult with scores and evidence
    """
    if not papers:
        return []

    total = len(papers)
    eval_results = []

    # Process papers with progress updates
    # Use semaphore to limit concurrent requests (avoid rate limiting)
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent evaluations

    async def evaluate_with_semaphore(index: int, paper: Paper) -> tuple[int, EvalResult]:
        async with semaphore:
            result = await _evaluate_paper(query, paper, llm, cache)
            return index, result

    # Create tasks
    tasks = [
        evaluate_with_semaphore(i, paper)
        for i, paper in enumerate(papers)
    ]

    # Process results as they complete
    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            index, result = await coro
            eval_results.append((index, result))
        except Exception as e:
            # This shouldn't happen as _evaluate_paper handles exceptions
            eval_results.append((completed, EvalResult(
                paper=papers[completed],
                score=1.0,
                meets_need=False,
                evidence_quote=papers[completed].title,
                evidence_field="title",
                short_reason="Evaluation failed",
            )))

        completed += 1
        if progress_callback:
            progress_callback(completed, total)

    # Sort by original index to maintain order
    eval_results.sort(key=lambda x: x[0])
    return [result for _, result in eval_results]


async def _evaluate_paper(
    query: str,
    paper: Paper,
    llm: "LLMClient",
    cache: Optional["Cache"] = None,
) -> EvalResult:
    """Evaluate a single paper."""
    # Check cache
    if cache:
        cache_key = cache.hash_key("eval", query, paper.source_id, paper.title)
        cached = await cache.get(cache_key)
        if cached:
            return EvalResult(paper=paper, **cached)

    # Build evaluation prompt
    prompt = _build_eval_prompt(query, paper)

    try:
        evaluation = await llm.eval_completion(
            prompt=prompt,
            response_model=PaperEvaluation,
            system_prompt=EVAL_SYSTEM_PROMPT,
        )

        # Validate evidence_quote is actually in the paper
        evaluation = _validate_evidence(evaluation, paper)

        result = EvalResult(
            paper=paper,
            score=evaluation.score,
            meets_need=evaluation.meets_need,
            evidence_quote=evaluation.evidence_quote,
            evidence_field=evaluation.evidence_field,
            short_reason=evaluation.short_reason,
        )

        # Cache result
        if cache:
            cache_data = {
                "score": result.score,
                "meets_need": result.meets_need,
                "evidence_quote": result.evidence_quote,
                "evidence_field": result.evidence_field,
                "short_reason": result.short_reason,
            }
            await cache.set(cache_key, cache_data)

        return result

    except Exception as e:
        # Fallback on error
        return EvalResult(
            paper=paper,
            score=1.0,
            meets_need=False,
            evidence_quote=paper.title,
            evidence_field="title",
            short_reason=f"Evaluation error: {str(e)[:50]}",
        )


def _build_eval_prompt(query: str, paper: Paper) -> str:
    """Build the evaluation prompt for a paper."""
    abstract_text = paper.abstract or "(No abstract available)"

    return f"""Evaluate this paper's relevance to the user's query.

USER QUERY: "{query}"

PAPER TITLE: {paper.title}

PAPER ABSTRACT: {abstract_text}

Evaluate the paper and provide:
1. A relevance score (0-10)
2. Whether it meets the user's need (true/false)
3. The most relevant quote from the title OR abstract (must be exact text)
4. Which field the quote is from ("title" or "abstract")
5. A brief reason for your assessment"""


def _validate_evidence(evaluation: PaperEvaluation, paper: Paper) -> PaperEvaluation:
    """Validate and fix evidence quote if needed."""
    quote = evaluation.evidence_quote
    field = evaluation.evidence_field

    # Check if quote exists in the claimed field
    if field == "title":
        if quote.lower() in paper.title.lower():
            return evaluation
        # Try to find in abstract instead
        if paper.abstract and quote.lower() in paper.abstract.lower():
            return PaperEvaluation(
                score=evaluation.score,
                meets_need=evaluation.meets_need,
                evidence_quote=quote,
                evidence_field="abstract",
                short_reason=evaluation.short_reason,
            )
    elif field == "abstract":
        if paper.abstract and quote.lower() in paper.abstract.lower():
            return evaluation
        # Try to find in title instead
        if quote.lower() in paper.title.lower():
            return PaperEvaluation(
                score=evaluation.score,
                meets_need=evaluation.meets_need,
                evidence_quote=quote,
                evidence_field="title",
                short_reason=evaluation.short_reason,
            )

    # Quote not found - use title as fallback
    return PaperEvaluation(
        score=evaluation.score,
        meets_need=evaluation.meets_need,
        evidence_quote=paper.title[:100],
        evidence_field="title",
        short_reason=evaluation.short_reason,
    )

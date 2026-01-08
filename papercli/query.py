"""Query intent extraction and rewriting using LLM."""

from typing import TYPE_CHECKING, Optional

from papercli.models import QueryIntent

if TYPE_CHECKING:
    from papercli.cache import Cache
    from papercli.llm import LLMClient

INTENT_SYSTEM_PROMPT = """You are a search query expert. Given a user's natural language query about academic papers, extract the search intent and generate optimized search terms.

Your task:
1. Understand what the user is looking for
2. Generate an English search query optimized for academic databases
3. Identify key terms, synonyms, and related concepts
4. Identify any phrases that must appear (quoted phrases)
5. Identify any terms to exclude

Focus on scientific/academic terminology. Expand abbreviations when helpful.
If the query is in Chinese or another language, translate it to English for search."""


async def extract_intent(
    query: str,
    llm: "LLMClient",
    cache: Optional["Cache"] = None,
) -> QueryIntent:
    """
    Extract search intent from user query using LLM.

    Args:
        query: User's natural language query
        llm: LLM client
        cache: Optional cache for storing results

    Returns:
        QueryIntent with extracted search parameters
    """
    # Check cache
    if cache:
        cache_key = f"intent:{query}"
        cached = await cache.get(cache_key)
        if cached:
            return QueryIntent.model_validate(cached)

    prompt = f"""Analyze this search query and extract the search intent:

Query: "{query}"

Extract:
1. An optimized English search query for academic databases (query_en)
2. Key terms that should be searched (keywords)
3. Synonyms or related terms for important concepts (synonyms - as dict mapping term to list of synonyms)
4. Any exact phrases that must appear (required_phrases)
5. Any terms to exclude from results (exclude_terms)

If the original query is in Chinese, also provide query_zh with the Chinese version."""

    try:
        intent = await llm.intent_completion(
            prompt=prompt,
            response_model=QueryIntent,
            system_prompt=INTENT_SYSTEM_PROMPT,
        )
    except Exception:
        # Fallback: use query as-is
        intent = QueryIntent(
            query_en=query,
            keywords=query.split(),
        )

    # Cache result
    if cache:
        await cache.set(cache_key, intent.model_dump())

    return intent


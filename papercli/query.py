"""Query intent extraction and rewriting using LLM."""

from typing import TYPE_CHECKING, Optional

from papercli.models import QueryIntent

if TYPE_CHECKING:
    from papercli.cache import Cache
    from papercli.llm import LLMClient

INTENT_SYSTEM_PROMPT = """You are a search query expert specializing in academic literature search. Given a user's natural language query about academic papers, you must:

1. First, think step-by-step about what the user really wants to find
2. Analyze the domain, key concepts, and research context
3. Generate an optimized English search query for academic databases
4. Identify key terms, synonyms, abbreviations, and related concepts
5. Determine if any specific phrases must appear or terms should be excluded

Focus on scientific/academic terminology. Always expand abbreviations and provide common synonyms.
If the query is in Chinese or another language, translate it to English for search while preserving the original meaning."""


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

User Query: "{query}"

Please provide:

1. **reasoning**: Your step-by-step thinking process:
   - What is the user looking for?
   - What research domain/field is this?
   - What are the key concepts?
   - Are there any abbreviations to expand?
   - What related terms might help find relevant papers?

2. **query_en**: An optimized English search query for academic databases (concise but comprehensive)

3. **keywords**: Key terms that should be searched (list of important words/phrases)

4. **synonyms**: A dictionary mapping key terms to their synonyms/related terms/abbreviations
   Example: {{"CRISPR": ["Cas9", "gene editing", "genome editing"], "cancer": ["tumor", "carcinoma", "malignancy"]}}

5. **required_phrases**: Any exact phrases that must appear in results (usually empty unless user specified)

6. **exclude_terms**: Any terms to exclude from results (usually empty unless user wants to filter something out)

If the original query is in Chinese, also provide **query_zh** with the Chinese version."""

    try:
        intent = await llm.intent_completion(
            prompt=prompt,
            response_model=QueryIntent,
            system_prompt=INTENT_SYSTEM_PROMPT,
        )
    except Exception:
        # Fallback: use query as-is
        intent = QueryIntent(
            reasoning="Fallback mode: using original query as-is due to LLM error.",
            query_en=query,
            keywords=query.split(),
        )

    # Cache result
    if cache:
        await cache.set(cache_key, intent.model_dump())

    return intent

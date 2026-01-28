"""Query intent extraction and rewriting using LLM."""

from typing import TYPE_CHECKING, Optional

from papercli.models import PlatformQueryResult, QueryIntent

if TYPE_CHECKING:
    from papercli.cache import Cache
    from papercli.llm import LLMClient


# Supported platforms and their aliases
PLATFORM_ALIASES: dict[str, str] = {
    "pubmed": "pubmed",
    "scholar": "scholar",
    "google_scholar": "scholar",
    "wos": "wos",
    "web_of_science": "wos",
    "world_of_knowledge": "wos",
}

VALID_PLATFORMS = {"pubmed", "scholar", "wos"}


# Platform-specific system prompts for query generation
PLATFORM_SYSTEM_PROMPTS: dict[str, str] = {
    "pubmed": """You are an expert in PubMed/MEDLINE database searching.

Given the user's search intent, generate an optimized PubMed search query using proper syntax:
- Use Boolean operators: AND, OR, NOT (must be uppercase)
- Use parentheses for grouping: (term1 OR term2) AND term3
- Use field tags when helpful: [Title/Abstract], [MeSH Terms], [Author]
- Use quotation marks for exact phrases: "gene editing"
- Use truncation with asterisk: therap* (matches therapy, therapies, therapeutic)
- Keep the query readable and not overly complex (avoid nesting more than 2-3 levels)

Example good queries:
- ("CRISPR" OR "Cas9") AND "cancer" AND "therapy"[Title/Abstract]
- (obesity[MeSH] OR "metabolic syndrome") AND (treatment OR intervention)

Output a single search string that can be directly pasted into PubMed's search box.""",

    "scholar": """You are an expert in Google Scholar searching.

Given the user's search intent, generate an optimized Google Scholar search query:
- Keep it concise: Scholar works best with 3-6 key terms
- Use quotation marks for exact phrases: "machine learning"
- Use OR for alternatives: "deep learning" OR "neural network"
- Use minus sign to exclude: -review -meta-analysis
- Avoid complex Boolean logic (Scholar handles it poorly)
- Focus on the most distinctive and specific terms

Example good queries:
- "single cell RNA-seq" "trajectory inference" methods
- "protein structure prediction" AlphaFold OR ESMFold
- CRISPR "base editing" applications -review

Output a single search string optimized for Google Scholar.""",

    "wos": """You are an expert in Web of Science (WoS) database searching.

Given the user's search intent, generate an optimized Web of Science search query:
- Use Boolean operators: AND, OR, NOT
- Use parentheses for grouping
- Use field tags: TS= (Topic), TI= (Title), AU= (Author), SO= (Source/Journal)
- TS= searches Title, Abstract, Author Keywords, and Keywords Plus
- Use quotation marks for exact phrases
- Use wildcards: * (any characters), ? (single character)
- Use NEAR/x for proximity: term1 NEAR/5 term2 (within 5 words)

Example good queries:
- TS=("machine learning" AND (drug OR pharmaceutical) AND discovery)
- TI=(CRISPR AND cancer) AND TS=(therapy OR treatment)
- TS=("climate change" NEAR/3 adaptation) AND TS=(agriculture OR crop*)

Output a single search string that can be used in Web of Science Advanced Search.""",
}


INTENT_SYSTEM_PROMPT = """You are a search query expert specializing in academic literature search. Given a user's natural language query about academic papers, you must:

1. First, think step-by-step about what the user really wants to find
2. Analyze the domain, key concepts, and research context
3. Generate an optimized English search query for academic databases
4. Identify key terms, synonyms, abbreviations, and related concepts
5. Determine if any specific phrases must appear or terms should be excluded
6. Extract any publication year or journal/venue constraints mentioned in the query

Focus on scientific/academic terminology. Always expand abbreviations and provide common synonyms.
If the query is in Chinese or another language, translate it to English for search while preserving the original meaning.

Pay special attention to:
- Year mentions like "2025", "recent", "last 3 years", "published in 2024-2025"
- Journal/venue mentions like "in Nature", "published in Bioinformatics", "from Cell", "in PNAS\""""


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
   - Are there any year or journal/venue constraints?

2. **query_en**: An optimized English search query for academic databases (concise but comprehensive).
   NOTE: Do NOT include year or journal filters in this query string - those go in separate fields below.

3. **keywords**: Key terms that should be searched. IMPORTANT RULES:
   - ALL keywords MUST be in English (translate if the query is in another language)
   - Each keyword should be 1-3 words maximum (e.g., "gene discovery", "plant biology", "CRISPR")
   - Do NOT include full sentences or long phrases
   - Include 5-10 relevant scientific/academic terms
   Example: ["gene discovery", "plant genomics", "novel genes", "functional annotation", "gene prediction"]

4. **synonyms**: A dictionary mapping key terms to their synonyms/related terms/abbreviations
   Example: {{"CRISPR": ["Cas9", "gene editing", "genome editing"], "cancer": ["tumor", "carcinoma", "malignancy"]}}

5. **required_phrases**: Any exact phrases that must appear in results (usually empty unless user specified)

6. **exclude_terms**: Any terms to exclude from results (usually empty unless user wants to filter something out)

7. **year**: If user specifies an exact publication year (e.g., "papers from 2025", "2024 publications"), provide the year as an integer. Otherwise null.

8. **year_min** and **year_max**: If user specifies a year range (e.g., "last 3 years", "2020-2024", "since 2022"), provide the range bounds. Otherwise null.
   - For "recent" or "last N years", calculate from current year (2026).
   - If only minimum is specified (e.g., "since 2020"), set year_min only.

9. **venue**: If user specifies a journal or conference name (e.g., "in Nature", "published in Bioinformatics", "from Cell"), extract the venue name. Otherwise null.
   - Use the standard journal name (e.g., "Bioinformatics" not "bioinformatics journal")
   - Common examples: "Nature", "Science", "Cell", "PNAS", "Bioinformatics", "Nucleic Acids Research"

If the original query is in Chinese, also provide **query_zh** with the Chinese version."""

    # Let LLMError propagate for proper diagnostics
    intent = await llm.reasoning_completion(
        prompt=prompt,
        response_model=QueryIntent,
        system_prompt=INTENT_SYSTEM_PROMPT,
    )

    # Cache result
    if cache:
        await cache.set(cache_key, intent.model_dump())

    return intent


class _PlatformQueryLLMResponse(QueryIntent):
    """Internal model for LLM response when generating platform query.
    
    Extends QueryIntent with platform-specific fields.
    """
    platform_query: str
    notes: str = ""


async def generate_platform_query(
    query: str,
    platform: str,
    llm: "LLMClient",
    cache: Optional["Cache"] = None,
) -> PlatformQueryResult:
    """
    Generate a platform-specific search query from user's natural language query.

    Args:
        query: User's natural language query
        platform: Target platform (pubmed, scholar, wos)
        llm: LLM client
        cache: Optional cache for storing results

    Returns:
        PlatformQueryResult with the platform-specific query and intent
    """
    # Normalize platform name
    platform = PLATFORM_ALIASES.get(platform.lower(), platform.lower())
    if platform not in VALID_PLATFORMS:
        raise ValueError(
            f"Unknown platform '{platform}'. Valid platforms: {', '.join(sorted(VALID_PLATFORMS))}"
        )

    # Check cache first
    if cache:
        cache_key = f"platform_query:{platform}:{query}"
        cached = await cache.get(cache_key)
        if cached:
            return PlatformQueryResult.model_validate(cached)

    # First extract the intent
    intent = await extract_intent(query, llm, cache)

    # Build prompt for platform-specific query generation
    platform_system = PLATFORM_SYSTEM_PROMPTS[platform]
    
    # Build context from intent
    keywords_str = ", ".join(intent.keywords) if intent.keywords else "N/A"
    synonyms_str = ""
    if intent.synonyms:
        synonyms_str = "\n".join(
            f"  - {term}: {', '.join(syns)}" for term, syns in intent.synonyms.items()
        )
    else:
        synonyms_str = "  N/A"

    required_str = ", ".join(f'"{p}"' for p in intent.required_phrases) if intent.required_phrases else "None"
    exclude_str = ", ".join(intent.exclude_terms) if intent.exclude_terms else "None"

    prompt = f"""Based on the following search intent, generate an optimized search query for {platform.upper()}.

**User's Original Query:** "{query}"

**Extracted Intent:**
- English Query: {intent.query_en}
- Keywords: {keywords_str}
- Synonyms/Related Terms:
{synonyms_str}
- Required Phrases: {required_str}
- Exclude Terms: {exclude_str}

**Your Task:**
1. Generate a **platform_query**: A search string optimized for {platform.upper()} that can be directly copied and pasted into the search box.
2. Provide brief **notes** (1-3 sentences): Any tips for refining the search on this platform (e.g., useful filters, date ranges, or alternative strategies).

Remember to use proper {platform.upper()} syntax and keep the query practical (not overly complex)."""

    try:
        response = await llm.reasoning_completion(
            prompt=prompt,
            response_model=_PlatformQueryLLMResponse,
            system_prompt=platform_system,
        )
        
        result = PlatformQueryResult(
            platform=platform,
            platform_query=response.platform_query,
            notes=response.notes,
            intent=intent,
        )
    except Exception:
        # Fallback: construct a basic query from intent
        if intent.keywords:
            fallback_query = " AND ".join(f'"{kw}"' for kw in intent.keywords[:5])
        else:
            fallback_query = intent.query_en
        
        result = PlatformQueryResult(
            platform=platform,
            platform_query=fallback_query,
            notes="[Fallback mode] LLM generation failed. This is a basic query constructed from keywords.",
            intent=intent,
        )

    # Cache the result
    if cache:
        await cache.set(cache_key, result.model_dump())

    return result

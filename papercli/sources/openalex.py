"""OpenAlex search adapter."""

from typing import TYPE_CHECKING, Any, Optional

import httpx

from papercli.models import Paper, QueryIntent
from papercli.sources.base import BaseSource

if TYPE_CHECKING:
    from papercli.cache import Cache

OPENALEX_API = "https://api.openalex.org"


class OpenAlexSource(BaseSource):
    """OpenAlex search adapter."""

    name = "openalex"

    def __init__(self, cache: Optional["Cache"] = None, email: Optional[str] = None):
        self.cache = cache
        self.email = email  # For polite pool

    async def search(
        self,
        intent: QueryIntent,
        max_results: int = 20,
    ) -> list[Paper]:
        """Search OpenAlex for papers matching the query intent."""
        query = intent.query_en

        # Build filter components for year and venue
        filter_key = self._get_filter_key(intent)
        cache_key = f"openalex:{query}:{max_results}:{filter_key}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return [Paper.model_validate(p) for p in cached]

        headers = {}
        if self.email:
            headers["User-Agent"] = f"papercli/0.1.0 (mailto:{self.email})"

        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            params: dict[str, str | int] = {
                "search": query,
                "per_page": max_results,
                "select": "id,title,abstract_inverted_index,publication_year,authorships,primary_location,doi",
            }

            # Build filter string for year and venue
            filter_parts = self._build_filter(intent)
            if filter_parts:
                params["filter"] = ",".join(filter_parts)

            url = f"{OPENALEX_API}/works"
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            papers = self._parse_results(data.get("results", []))

        # Cache results
        if self.cache and papers:
            await self.cache.set(cache_key, [p.model_dump() for p in papers])

        return papers

    def _get_filter_key(self, intent: QueryIntent) -> str:
        """Generate cache key component for filters."""
        parts = []
        if intent.year:
            parts.append(f"y{intent.year}")
        if intent.year_min:
            parts.append(f"ymin{intent.year_min}")
        if intent.year_max:
            parts.append(f"ymax{intent.year_max}")
        if intent.venue:
            parts.append(f"v{intent.venue}")
        return "_".join(parts) if parts else "nofilter"

    def _build_filter(self, intent: QueryIntent) -> list[str]:
        """Build OpenAlex filter components from intent."""
        filters = []

        # Year filters
        if intent.year:
            filters.append(f"publication_year:{intent.year}")
        elif intent.year_min and intent.year_max:
            filters.append(f"publication_year:{intent.year_min}-{intent.year_max}")
        elif intent.year_min:
            filters.append(f"publication_year:>{intent.year_min - 1}")
        elif intent.year_max:
            filters.append(f"publication_year:<{intent.year_max + 1}")

        # Venue filter - use primary_location.source.display_name
        if intent.venue:
            # OpenAlex uses display_name for journal names
            filters.append(f"primary_location.source.display_name.search:{intent.venue}")

        return filters

    def _parse_results(self, results: list[dict[str, Any]]) -> list[Paper]:
        """Parse OpenAlex API results into Paper objects."""
        papers = []
        for work in results:
            try:
                paper = self._parse_work(work)
                if paper:
                    papers.append(paper)
            except Exception:
                continue
        return papers

    def _parse_work(self, work: dict[str, Any]) -> Optional[Paper]:
        """Parse a single OpenAlex work."""
        # ID
        openalex_id = work.get("id", "")
        if not openalex_id:
            return None
        # Extract just the ID part (e.g., "W2741809807" from "https://openalex.org/W2741809807")
        source_id = openalex_id.split("/")[-1] if "/" in openalex_id else openalex_id

        # Title
        title = work.get("title")
        if not title:
            return None

        # Abstract - reconstruct from inverted index
        abstract = None
        abstract_inverted_index = work.get("abstract_inverted_index")
        if abstract_inverted_index:
            abstract = self._reconstruct_abstract(abstract_inverted_index)

        # Year
        year = work.get("publication_year")

        # Authors
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            display_name = author.get("display_name")
            if display_name:
                authors.append(display_name)

        # DOI
        doi = work.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

        # Venue
        venue = None
        primary_location = work.get("primary_location")
        if primary_location:
            source = primary_location.get("source")
            if source:
                venue = source.get("display_name")

        # URL - prefer DOI, fallback to OpenAlex URL
        url = f"https://doi.org/{doi}" if doi else openalex_id

        return Paper(
            source="openalex",
            source_id=source_id,
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            url=url,
            doi=doi,
            venue=venue,
        )

    @staticmethod
    def reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
        """
        Reconstruct abstract text from OpenAlex inverted index.

        The inverted index maps words to their positions in the abstract.
        Example: {"The": [0, 5], "quick": [1], "brown": [2], ...}
        """
        if not inverted_index:
            return ""

        # Find max position to determine array size
        max_pos = 0
        for positions in inverted_index.values():
            if positions:
                max_pos = max(max_pos, max(positions))

        # Build word array
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                if pos < len(words):
                    words[pos] = word

        return " ".join(words)

    def _reconstruct_abstract(self, inverted_index: dict[str, list[int]]) -> str:
        """Instance method wrapper for reconstruct_abstract."""
        return self.reconstruct_abstract(inverted_index)


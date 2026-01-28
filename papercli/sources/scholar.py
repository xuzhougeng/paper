"""Google Scholar search adapter using SerpAPI."""

from typing import TYPE_CHECKING, Any, Optional

import httpx

from papercli.models import Paper, QueryIntent
from papercli.sources.base import BaseSource

if TYPE_CHECKING:
    from papercli.cache import Cache

SERPAPI_URL = "https://serpapi.com/search"


class ScholarSource(BaseSource):
    """Google Scholar search using SerpAPI."""

    name = "scholar"

    def __init__(self, cache: Optional["Cache"] = None, api_key: Optional[str] = None):
        self.cache = cache
        self._api_key = api_key

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from instance or environment."""
        if self._api_key:
            return self._api_key
        import os

        return os.environ.get("SERPAPI_API_KEY")

    async def search(
        self,
        intent: QueryIntent,
        max_results: int = 20,
    ) -> list[Paper]:
        """Search Google Scholar via SerpAPI."""
        api_key = self.api_key
        if not api_key:
            # Skip Scholar search if no API key
            return []

        query = intent.query_en

        # Build cache key including year filters
        filter_key = self._get_filter_key(intent)
        cache_key = f"scholar:{query}:{max_results}:{filter_key}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return [Paper.model_validate(p) for p in cached]

        async with httpx.AsyncClient(timeout=30.0) as client:
            params: dict[str, str | int] = {
                "engine": "google_scholar",
                "q": query,
                "api_key": api_key,
                "num": min(max_results, 20),  # SerpAPI max per page
            }

            # Add year range filters using SerpAPI's as_ylo/as_yhi
            if intent.year:
                params["as_ylo"] = intent.year
                params["as_yhi"] = intent.year
            else:
                if intent.year_min:
                    params["as_ylo"] = intent.year_min
                if intent.year_max:
                    params["as_yhi"] = intent.year_max

            response = await client.get(SERPAPI_URL, params=params)
            response.raise_for_status()

            data = response.json()
            papers = self._parse_results(data.get("organic_results", []))

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
        # Note: venue filtering not supported natively by Scholar
        return "_".join(parts) if parts else "nofilter"

    def _parse_results(self, results: list[dict[str, Any]]) -> list[Paper]:
        """Parse SerpAPI Scholar results into Paper objects."""
        papers = []
        for result in results:
            try:
                paper = self._parse_result(result)
                if paper:
                    papers.append(paper)
            except Exception:
                continue
        return papers

    def _parse_result(self, result: dict[str, Any]) -> Optional[Paper]:
        """Parse a single Scholar result."""
        # Title
        title = result.get("title")
        if not title:
            return None

        # ID (Scholar doesn't have persistent IDs, use position or result_id)
        source_id = result.get("result_id", "")
        if not source_id:
            # Generate from title hash
            import hashlib

            source_id = hashlib.md5(title.encode()).hexdigest()[:12]

        # Snippet as abstract (Scholar doesn't provide full abstracts)
        abstract = result.get("snippet")

        # Year - extract from publication_info
        year = None
        pub_info = result.get("publication_info", {})
        summary = pub_info.get("summary", "")
        if summary:
            import re

            # Look for 4-digit year in summary
            match = re.search(r"\b(19|20)\d{2}\b", summary)
            if match:
                year = int(match.group(0))

        # Authors - extract from publication_info
        authors = []
        authors_list = pub_info.get("authors", [])
        if authors_list:
            for author in authors_list:
                name = author.get("name")
                if name:
                    authors.append(name)
        elif summary:
            # Try to extract from summary (format: "Author1, Author2 - Journal, Year")
            parts = summary.split(" - ")
            if parts:
                author_part = parts[0]
                # Split by comma but be careful with "et al"
                if "," in author_part:
                    potential_authors = [a.strip() for a in author_part.split(",")]
                    authors = [a for a in potential_authors if a and len(a) < 50]

        # URL
        url = result.get("link")

        # Venue - from publication_info
        venue = None
        if summary and " - " in summary:
            parts = summary.split(" - ")
            if len(parts) >= 2:
                venue = parts[1].strip()
                # Remove year from venue
                import re

                venue = re.sub(r",?\s*(19|20)\d{2}$", "", venue).strip()

        # DOI - Scholar doesn't reliably provide DOI
        doi = None

        return Paper(
            source="scholar",
            source_id=source_id,
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            url=url,
            doi=doi,
            venue=venue,
        )


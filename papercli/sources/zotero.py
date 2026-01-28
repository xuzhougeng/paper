"""Zotero search adapter using the Zotero Web API."""

import re
from typing import TYPE_CHECKING, Any, Optional

import httpx

from papercli.models import Paper, QueryIntent
from papercli.sources.base import BaseSource

if TYPE_CHECKING:
    from papercli.cache import Cache
    from papercli.config import Settings

ZOTERO_API = "https://api.zotero.org"


class ZoteroSource(BaseSource):
    """Zotero search adapter using the Zotero Web API."""

    name = "zotero"

    def __init__(
        self,
        cache: Optional["Cache"] = None,
        settings: Optional["Settings"] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Zotero source.

        Args:
            cache: Optional cache for storing results
            settings: Settings object (preferred way to get config)
            api_key: Zotero API key (overrides settings)
            user_id: Zotero user ID (overrides settings)
            base_url: Zotero API base URL (overrides settings)
        """
        self.cache = cache
        self._settings = settings
        self._api_key = api_key
        self._user_id = user_id
        self._base_url = base_url

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from instance, settings, or environment."""
        if self._api_key:
            return self._api_key
        if self._settings and self._settings.zotero.api_key:
            return self._settings.zotero.api_key
        import os
        return os.environ.get("ZOTERO_API_KEY")

    @property
    def user_id(self) -> Optional[str]:
        """Get user ID from instance, settings, or environment."""
        if self._user_id:
            return self._user_id
        if self._settings and self._settings.zotero.user_id:
            return self._settings.zotero.user_id
        import os
        return os.environ.get("ZOTERO_USER_ID")

    @property
    def base_url(self) -> str:
        """Get base URL from instance, settings, or default."""
        if self._base_url:
            return self._base_url
        if self._settings and self._settings.zotero.base_url:
            return self._settings.zotero.base_url
        import os
        return os.environ.get("ZOTERO_BASE_URL", ZOTERO_API)

    @property
    def qmode(self) -> str:
        """Get query mode from settings or default."""
        if self._settings:
            return self._settings.zotero.qmode
        return "titleCreatorYear"

    @property
    def item_type(self) -> str:
        """Get item type filter from settings or default."""
        if self._settings:
            return self._settings.zotero.item_type
        return "-attachment"

    async def search(
        self,
        intent: QueryIntent,
        max_results: int = 20,
    ) -> list[Paper]:
        """Search Zotero library for papers matching the query intent."""
        api_key = self.api_key
        user_id = self.user_id

        if not api_key or not user_id:
            # Skip Zotero search if not configured
            return []

        query = self._build_query(intent)

        # Build cache key including filter info (filtering is done post-query)
        filter_key = self._get_filter_key(intent)
        cache_key = f"zotero:{user_id}:{query}:{max_results}:{filter_key}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return [Paper.model_validate(p) for p in cached]

        headers = {
            "Zotero-API-Key": api_key,
            "Zotero-API-Version": "3",
        }

        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            params = {
                "q": query,
                "qmode": self.qmode,
                "itemType": self.item_type,
                "limit": max_results,
                "sort": "relevance",
            }

            url = f"{self.base_url}/users/{user_id}/items"
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            papers = self._parse_results(data)

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
        # Note: Zotero doesn't support native filtering; post-filtering is applied in pipeline
        return "_".join(parts) if parts else "nofilter"

    def _build_query(self, intent: QueryIntent) -> str:
        """Build Zotero search query from intent."""
        query_parts = [intent.query_en]

        # Add required phrases
        for phrase in intent.required_phrases:
            query_parts.append(f'"{phrase}"')

        return " ".join(query_parts)

    def _parse_results(self, items: list[dict[str, Any]]) -> list[Paper]:
        """Parse Zotero API results into Paper objects."""
        papers = []
        for item in items:
            try:
                paper = self._parse_item(item)
                if paper:
                    papers.append(paper)
            except Exception:
                continue
        return papers

    def _parse_item(self, item: dict[str, Any]) -> Optional[Paper]:
        """Parse a single Zotero item into a Paper object."""
        data = item.get("data", {})

        # Key (required)
        key = data.get("key")
        if not key:
            return None

        # Title (required)
        title = data.get("title")
        if not title:
            return None

        # Abstract
        abstract = data.get("abstractNote") or None

        # DOI
        doi = data.get("DOI") or None
        if doi:
            # Normalize DOI (remove https://doi.org/ prefix if present)
            if doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            elif doi.startswith("http://doi.org/"):
                doi = doi[len("http://doi.org/"):]

        # URL - prefer DOI URL, fallback to item URL
        url = data.get("url")
        if doi:
            url = f"https://doi.org/{doi}"
        elif not url:
            # Try to construct a URL from the item key
            url = None

        # Venue (publication title)
        venue = data.get("publicationTitle") or data.get("journalAbbreviation") or None

        # Year - extract from date field
        year = self._extract_year(data.get("date"))

        # Authors - extract from creators
        authors = self._extract_authors(data.get("creators", []))

        return Paper(
            source="zotero",
            source_id=key,
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            url=url,
            doi=doi,
            venue=venue,
        )

    @staticmethod
    def _extract_year(date_str: Optional[str]) -> Optional[int]:
        """
        Extract year from Zotero date string.

        Handles various formats:
        - "2023"
        - "2023-01-15"
        - "2023 Jan"
        - "January 2023"
        - "2023/01/15"
        """
        if not date_str:
            return None

        # Try to find a 4-digit year
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                pass

        return None

    @staticmethod
    def _extract_authors(creators: list[dict[str, Any]]) -> list[str]:
        """
        Extract author names from Zotero creators list.

        Prioritizes authors, falls back to all creators if no authors found.
        """
        if not creators:
            return []

        # First, try to get only authors
        authors = []
        for creator in creators:
            if creator.get("creatorType") == "author":
                name = _format_creator_name(creator)
                if name:
                    authors.append(name)

        # If no authors found, use all creators
        if not authors:
            for creator in creators:
                name = _format_creator_name(creator)
                if name:
                    authors.append(name)

        return authors


def _format_creator_name(creator: dict[str, Any]) -> Optional[str]:
    """Format a creator's name from Zotero creator dict."""
    # Single-field name (e.g., institutional authors)
    if "name" in creator:
        return creator["name"]

    # Two-field name (firstName, lastName)
    first_name = creator.get("firstName", "").strip()
    last_name = creator.get("lastName", "").strip()

    if first_name and last_name:
        return f"{first_name} {last_name}"
    elif last_name:
        return last_name
    elif first_name:
        return first_name

    return None

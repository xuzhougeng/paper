"""arXiv search adapter using the arXiv API."""

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

import httpx

from papercli.models import Paper, QueryIntent
from papercli.sources.base import BaseSource

if TYPE_CHECKING:
    from papercli.cache import Cache

ARXIV_API = "http://export.arxiv.org/api/query"

# XML namespaces
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivSource(BaseSource):
    """arXiv search adapter."""

    name = "arxiv"

    def __init__(self, cache: Optional["Cache"] = None):
        self.cache = cache

    async def search(
        self,
        intent: QueryIntent,
        max_results: int = 20,
    ) -> list[Paper]:
        """Search arXiv for papers matching the query intent."""
        query = self._build_query(intent)

        # Build cache key including year filters
        filter_key = self._get_filter_key(intent)
        cache_key = f"arxiv:{query}:{max_results}:{filter_key}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return [Paper.model_validate(p) for p in cached]

        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            response = await client.get(ARXIV_API, params=params)
            response.raise_for_status()

            papers = self._parse_atom(response.text)

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
        # Note: venue filtering not applicable to arXiv
        return "_".join(parts) if parts else "nofilter"

    def _build_query(self, intent: QueryIntent) -> str:
        """Build arXiv API query from intent."""
        # arXiv uses: all, ti (title), abs (abstract), au (author), etc.
        query = intent.query_en

        # Search in title and abstract
        # arXiv query format: all:term OR ti:term OR abs:term
        terms = query.split()
        if len(terms) <= 3:
            # Short query - search everywhere
            base_query = f"all:{quote(query)}"
        else:
            # Longer query - search in title and abstract
            base_query = f"ti:{quote(query)} OR abs:{quote(query)}"

        # Add date filter if specified
        # arXiv uses submittedDate:[YYYYMMDD* TO YYYYMMDD*] format
        date_filter = self._build_date_filter(intent)
        if date_filter:
            return f"({base_query}) AND {date_filter}"

        return base_query

    def _build_date_filter(self, intent: QueryIntent) -> str | None:
        """Build arXiv date filter from intent."""
        if intent.year:
            # Exact year: January 1 to December 31
            return f"submittedDate:[{intent.year}0101 TO {intent.year}1231]"
        elif intent.year_min or intent.year_max:
            start = f"{intent.year_min}0101" if intent.year_min else "*"
            end = f"{intent.year_max}1231" if intent.year_max else "*"
            return f"submittedDate:[{start} TO {end}]"
        return None

    def _parse_atom(self, xml_text: str) -> list[Paper]:
        """Parse arXiv Atom feed response."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers

        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
            except Exception:
                continue

        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Paper]:
        """Parse a single arXiv entry."""
        # ID - extract arXiv ID from URL
        id_elem = entry.find(f"{ATOM_NS}id")
        if id_elem is None or not id_elem.text:
            return None

        arxiv_id = id_elem.text
        # Extract ID from URL like "http://arxiv.org/abs/2301.12345v1"
        match = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", arxiv_id)
        if match:
            source_id = match.group(1)
        else:
            source_id = arxiv_id.split("/")[-1]

        # Title
        title_elem = entry.find(f"{ATOM_NS}title")
        title = title_elem.text if title_elem is not None and title_elem.text else ""
        if not title:
            return None
        # Clean up title (remove newlines and extra spaces)
        title = " ".join(title.split())

        # Abstract (summary)
        summary_elem = entry.find(f"{ATOM_NS}summary")
        abstract = None
        if summary_elem is not None and summary_elem.text:
            abstract = " ".join(summary_elem.text.split())

        # Published date -> year
        year = None
        published_elem = entry.find(f"{ATOM_NS}published")
        if published_elem is not None and published_elem.text:
            # Format: 2023-01-15T12:00:00Z
            match = re.match(r"(\d{4})", published_elem.text)
            if match:
                year = int(match.group(1))

        # Authors
        authors = []
        for author in entry.findall(f"{ATOM_NS}author"):
            name_elem = author.find(f"{ATOM_NS}name")
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)

        # URL - prefer abstract page
        url = f"https://arxiv.org/abs/{source_id}"

        # DOI - arXiv entries may have a DOI link
        doi = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "doi":
                href = link.get("href", "")
                if "doi.org/" in href:
                    doi = href.split("doi.org/")[-1]
                    break

        # Primary category as venue
        venue = None
        primary_cat = entry.find(f"{ARXIV_NS}primary_category")
        if primary_cat is not None:
            venue = f"arXiv:{primary_cat.get('term', '')}"

        return Paper(
            source="arxiv",
            source_id=source_id,
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            url=url,
            doi=doi,
            venue=venue,
        )


"""PubMed search adapter using NCBI E-utilities."""

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Optional, Callable
from urllib.parse import quote

import httpx

from papercli.models import Paper, QueryIntent
from papercli.sources.base import BaseSource

if TYPE_CHECKING:
    from papercli.cache import Cache

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedSource(BaseSource):
    """PubMed search using NCBI E-utilities API."""

    name = "pubmed"

    def __init__(self, cache: Optional["Cache"] = None, api_key: Optional[str] = None):
        self.cache = cache
        self.api_key = api_key

    async def search(
        self,
        intent: QueryIntent,
        max_results: int = 20,
    ) -> list[Paper]:
        """Search PubMed for papers matching the query intent."""
        # Build search query - use English query for PubMed
        query = self._build_query(intent)

        # Build cache key including filter parameters
        year_filter = self._get_year_filter_key(intent)
        cache_key = f"pubmed:{query}:{max_results}:{year_filter}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return [Paper.model_validate(p) for p in cached]

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Search to get PMIDs (with date filters)
            pmids = await self._esearch(client, query, max_results, intent)
            if not pmids:
                return []

            # Step 2: Fetch paper details
            papers = await self._efetch(client, pmids)

        # Cache results
        if self.cache and papers:
            await self.cache.set(cache_key, [p.model_dump() for p in papers])

        return papers

    def _get_year_filter_key(self, intent: QueryIntent) -> str:
        """Generate cache key component for year filters."""
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

    def _build_query(self, intent: QueryIntent) -> str:
        """Build PubMed search query from intent."""
        parts = []

        # Main query - search in title and abstract
        main_query = intent.query_en.strip() if intent.query_en else ""
        if main_query:
            parts.append(f"({main_query}[Title/Abstract])")

        # Add required phrases
        for phrase in intent.required_phrases:
            parts.append(f'"{phrase}"[Title/Abstract]')

        # Add venue/journal filter if specified
        if intent.venue:
            parts.append(f'"{intent.venue}"[Journal]')

        # Combine with AND
        query = " AND ".join(parts)

        # Add exclusions with NOT
        for term in intent.exclude_terms:
            query += f" NOT {term}[Title/Abstract]"

        return query

    async def search_all(
        self,
        intent: QueryIntent,
        batch_size: int = 500,
        max_results: Optional[int] = None,
        progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> list[Paper]:
        """Search PubMed and retrieve all results using pagination."""
        fetch_chunk_size = 200
        query = self._build_query(intent)
        if not query:
            return []

        year_filter = self._get_year_filter_key(intent)
        cache_key = f"pubmed:all:{query}:{batch_size}:{max_results}:{year_filter}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                if isinstance(cached, dict):
                    papers = cached.get("papers", [])
                    total_cached = cached.get("total")
                else:
                    papers = cached
                    total_cached = len(papers)
                if progress_cb:
                    total_value = total_cached if total_cached is not None else len(papers)
                    progress_cb(0, total_value)
                    progress_cb(len(papers), total_value)
                return [Paper.model_validate(p) for p in papers]

        all_papers: list[Paper] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            retstart = 0
            total = None
            done = 0
            while True:
                pmids, count = await self._esearch_with_count(
                    client, query, batch_size, retstart, intent
                )
                if total is None:
                    total = count
                    if max_results is not None:
                        total = min(total, max_results)
                    if progress_cb:
                        progress_cb(0, total)
                if not pmids:
                    break

                # efetch URL can be too long if we pass too many IDs
                for i in range(0, len(pmids), fetch_chunk_size):
                    chunk = pmids[i : i + fetch_chunk_size]
                    papers = await self._efetch(client, chunk)
                    all_papers.extend(papers)

                retstart += batch_size
                advance = len(pmids)
                if max_results is not None and done + advance > max_results:
                    advance = max_results - done
                done += advance
                if progress_cb and advance > 0:
                    progress_cb(advance, total)
                if max_results is not None and len(all_papers) >= max_results:
                    all_papers = all_papers[:max_results]
                    break
                if total is not None and retstart >= total:
                    break

        if self.cache and all_papers:
            await self.cache.set(
                cache_key,
                {
                    "papers": [p.model_dump() for p in all_papers],
                    "total": total,
                },
            )

        return all_papers

    async def _esearch(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
        intent: QueryIntent,
    ) -> list[str]:
        """Use esearch to get PMIDs with optional date filtering."""
        if not query:
            return []
        params: dict[str, str | int] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        # Add date filters using E-utilities date parameters
        # These are more reliable than embedding in the query term
        if intent.year or intent.year_min or intent.year_max:
            params["datetype"] = "PDAT"  # Publication date

            if intent.year:
                # Exact year: set both min and max to same year
                params["mindate"] = str(intent.year)
                params["maxdate"] = str(intent.year)
            else:
                # Year range
                if intent.year_min:
                    params["mindate"] = str(intent.year_min)
                if intent.year_max:
                    params["maxdate"] = str(intent.year_max)

        url = f"{EUTILS_BASE}/esearch.fcgi"
        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    async def _esearch_with_count(
        self,
        client: httpx.AsyncClient,
        query: str,
        retmax: int,
        retstart: int,
        intent: QueryIntent,
    ) -> tuple[list[str], int]:
        """Use esearch to get PMIDs plus total count with optional date filtering."""
        if not query:
            return ([], 0)
        params: dict[str, str | int] = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retstart": retstart,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        if intent.year or intent.year_min or intent.year_max:
            params["datetype"] = "PDAT"
            if intent.year:
                params["mindate"] = str(intent.year)
                params["maxdate"] = str(intent.year)
            else:
                if intent.year_min:
                    params["mindate"] = str(intent.year_min)
                if intent.year_max:
                    params["maxdate"] = str(intent.year_max)

        url = f"{EUTILS_BASE}/esearch.fcgi"
        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        result = data.get("esearchresult", {})
        idlist = result.get("idlist", [])
        count_raw = result.get("count", 0)
        try:
            count = int(count_raw)
        except (TypeError, ValueError):
            count = 0
        return (idlist, count)

    async def _efetch(self, client: httpx.AsyncClient, pmids: list[str]) -> list[Paper]:
        """Fetch paper details using efetch."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{EUTILS_BASE}/efetch.fcgi"
        response = await client.get(url, params=params)
        response.raise_for_status()

        return self._parse_pubmed_xml(response.text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[Paper]:
        """Parse PubMed XML response into Paper objects."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers

        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception:
                continue

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[Paper]:
        """Parse a single PubMed article."""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        # PMID
        pmid_elem = medline.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            return None
        pmid = pmid_elem.text

        # Title
        title_elem = medline.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""
        if not title:
            return None

        # Abstract
        abstract_parts = []
        for abstract_text in medline.findall(".//AbstractText"):
            if abstract_text.text:
                label = abstract_text.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {abstract_text.text}")
                else:
                    abstract_parts.append(abstract_text.text)
        abstract = " ".join(abstract_parts) if abstract_parts else None

        # Year
        year = None
        pub_date = medline.find(".//PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass
            # Try MedlineDate if Year not found
            if year is None:
                medline_date = pub_date.find("MedlineDate")
                if medline_date is not None and medline_date.text:
                    # Extract year from strings like "2023 Jan-Feb"
                    import re

                    match = re.search(r"(\d{4})", medline_date.text)
                    if match:
                        year = int(match.group(1))

        # Authors
        authors = []
        for author in medline.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and last_name.text:
                name = last_name.text
                if fore_name is not None and fore_name.text:
                    name = f"{fore_name.text} {name}"
                authors.append(name)

        # Journal
        journal_elem = medline.find(".//Journal/Title")
        venue = journal_elem.text if journal_elem is not None else None

        # DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return Paper(
            source="pubmed",
            source_id=pmid,
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            url=url,
            doi=doi,
            venue=venue,
        )

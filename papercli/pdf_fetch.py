"""PDF fetching via Unpaywall and PMC APIs."""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import httpx
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from papercli.config import Settings

# NCBI E-utilities base URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# DOI regex pattern
DOI_PATTERN = re.compile(r"^10\.\d{4,}/[^\s]+$")


class PDFResult(BaseModel):
    """Result of PDF URL lookup."""

    doi: str
    pdf_url: Optional[str] = Field(default=None)  # Direct PDF URL if found
    landing_url: Optional[str] = Field(default=None)  # Fallback landing page
    source: Literal["unpaywall", "pmc", "none"] = Field(default="none")
    oa_status: Optional[str] = Field(default=None)  # gold/green/hybrid/bronze/closed
    license: Optional[str] = Field(default=None)
    pmid: Optional[str] = Field(default=None)
    pmcid: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


class PDFFetchError(Exception):
    """Error during PDF fetching."""

    def __init__(self, message: str, *, doi: str | None = None, source: str | None = None):
        super().__init__(message)
        self.doi = doi
        self.source = source


def validate_doi(doi: str) -> str:
    """
    Validate and normalize a DOI.

    Args:
        doi: DOI string (with or without URL prefix)

    Returns:
        Normalized DOI (just the 10.xxxx/... part)

    Raises:
        ValueError: If DOI format is invalid
    """
    # Strip common prefixes
    doi = doi.strip()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:", "DOI:"]:
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix) :]
            break

    doi = doi.strip()

    if not DOI_PATTERN.match(doi):
        raise ValueError(
            f"Invalid DOI format: '{doi}'. Expected format: 10.xxxx/... "
            "(e.g., 10.1038/nature12373)"
        )

    return doi


def doi_to_filename(doi: str) -> str:
    """Convert DOI to a safe filename."""
    # Replace / with _ and other unsafe chars
    safe = doi.replace("/", "_").replace(":", "_").replace("<", "_").replace(">", "_")
    safe = safe.replace('"', "_").replace("\\", "_").replace("|", "_").replace("?", "_")
    safe = safe.replace("*", "_")
    return f"{safe}.pdf"


class UnpaywallClient:
    """Client for Unpaywall API."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return self.settings.unpaywall.base_url.rstrip("/")

    @property
    def email(self) -> str:
        return self.settings.get_unpaywall_email()

    @property
    def timeout(self) -> float:
        return self.settings.unpaywall.timeout

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def lookup(self, doi: str) -> PDFResult:
        """
        Look up PDF URL for a DOI via Unpaywall.

        Args:
            doi: Normalized DOI string

        Returns:
            PDFResult with pdf_url if found, or error info
        """
        client = await self._get_client()
        url = f"{self.base_url}/{doi}"

        try:
            response = await client.get(url, params={"email": self.email})

            if response.status_code == 404:
                return PDFResult(
                    doi=doi,
                    source="none",
                    error="DOI not found in Unpaywall database",
                )

            response.raise_for_status()
            data = response.json()

        except httpx.HTTPStatusError as e:
            return PDFResult(
                doi=doi,
                source="none",
                error=f"Unpaywall API error: HTTP {e.response.status_code}",
            )
        except httpx.RequestError as e:
            return PDFResult(
                doi=doi,
                source="none",
                error=f"Network error contacting Unpaywall: {e}",
            )

        # Extract OA info
        oa_status = data.get("oa_status")
        is_oa = data.get("is_oa", False)

        # Get best OA location
        best_loc = data.get("best_oa_location") or {}
        pdf_url = best_loc.get("url_for_pdf")
        landing_url = best_loc.get("url") or data.get("doi_url")
        license_info = best_loc.get("license")

        if pdf_url:
            return PDFResult(
                doi=doi,
                pdf_url=pdf_url,
                landing_url=landing_url,
                source="unpaywall",
                oa_status=oa_status,
                license=license_info,
            )

        if is_oa and landing_url:
            return PDFResult(
                doi=doi,
                landing_url=landing_url,
                source="unpaywall",
                oa_status=oa_status,
                license=license_info,
                error="No direct PDF link available, only landing page",
            )

        return PDFResult(
            doi=doi,
            landing_url=landing_url,
            source="none",
            oa_status=oa_status,
            error="Not open access or no PDF available via Unpaywall",
        )


class PMCClient:
    """Client for NCBI PMC API (DOI → PMID → PMCID → PDF)."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def api_key(self) -> str | None:
        return self.settings.get_ncbi_api_key()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _add_api_key(self, params: dict) -> dict:
        """Add NCBI API key to params if available."""
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def doi_to_pmid(self, doi: str) -> str | None:
        """
        Convert DOI to PMID using ESearch.

        Args:
            doi: DOI string

        Returns:
            PMID if found, None otherwise
        """
        client = await self._get_client()
        url = f"{EUTILS_BASE}/esearch.fcgi"

        params = self._add_api_key({
            "db": "pubmed",
            "term": f"{doi}[doi]",
            "retmode": "json",
            "retmax": 1,
        })

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            return id_list[0] if id_list else None

        except (httpx.HTTPError, KeyError, IndexError):
            return None

    async def pmid_to_pmcid(self, pmid: str) -> str | None:
        """
        Convert PMID to PMCID using ELink.

        Args:
            pmid: PubMed ID

        Returns:
            PMCID (e.g., "PMC1234567") if found, None otherwise
        """
        client = await self._get_client()
        url = f"{EUTILS_BASE}/elink.fcgi"

        params = self._add_api_key({
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "json",
        })

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Navigate the nested structure to find PMC ID
            linksets = data.get("linksets", [])
            if not linksets:
                return None

            linksetdbs = linksets[0].get("linksetdbs", [])
            for lsdb in linksetdbs:
                if lsdb.get("dbto") == "pmc":
                    links = lsdb.get("links", [])
                    if links:
                        return f"PMC{links[0]}"

            return None

        except (httpx.HTTPError, KeyError, IndexError):
            return None

    def pmcid_to_pdf_url(self, pmcid: str) -> str:
        """
        Construct PMC PDF URL from PMCID.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            Direct PDF URL
        """
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"

    async def lookup(self, doi: str) -> PDFResult:
        """
        Look up PDF URL for a DOI via PubMed/PMC chain.

        Args:
            doi: Normalized DOI string

        Returns:
            PDFResult with pdf_url if found, or error info
        """
        # Step 1: DOI → PMID
        pmid = await self.doi_to_pmid(doi)
        if not pmid:
            return PDFResult(
                doi=doi,
                source="none",
                error="DOI not found in PubMed",
            )

        # Step 2: PMID → PMCID
        pmcid = await self.pmid_to_pmcid(pmid)
        if not pmcid:
            return PDFResult(
                doi=doi,
                pmid=pmid,
                source="none",
                error=f"Article (PMID:{pmid}) not available in PMC",
            )

        # Step 3: PMCID → PDF URL
        pdf_url = self.pmcid_to_pdf_url(pmcid)

        return PDFResult(
            doi=doi,
            pdf_url=pdf_url,
            pmid=pmid,
            pmcid=pmcid,
            source="pmc",
        )


async def fetch_pdf_url(
    doi: str,
    settings: "Settings",
    skip_unpaywall: bool = False,
    skip_pmc: bool = False,
) -> PDFResult:
    """
    Fetch PDF URL for a DOI using Unpaywall (primary) and PMC (fallback).

    Args:
        doi: DOI string (will be validated and normalized)
        settings: Configuration settings
        skip_unpaywall: Skip Unpaywall lookup
        skip_pmc: Skip PMC fallback

    Returns:
        PDFResult with pdf_url if found
    """
    # Validate DOI
    doi = validate_doi(doi)

    result = PDFResult(doi=doi, source="none")

    # Try Unpaywall first
    if not skip_unpaywall:
        unpaywall = UnpaywallClient(settings)
        try:
            result = await unpaywall.lookup(doi)
            if result.pdf_url:
                return result
        finally:
            await unpaywall.close()

    # Fall back to PMC if no direct PDF from Unpaywall
    if not skip_pmc and not result.pdf_url:
        pmc = PMCClient(settings)
        try:
            pmc_result = await pmc.lookup(doi)
            if pmc_result.pdf_url:
                # Merge info: keep Unpaywall metadata if available
                pmc_result.oa_status = result.oa_status or pmc_result.oa_status
                pmc_result.license = result.license or pmc_result.license
                pmc_result.landing_url = result.landing_url or pmc_result.landing_url
                return pmc_result
            elif not result.error:
                # Keep original error if Unpaywall had one
                result = pmc_result
        finally:
            await pmc.close()

    return result


async def download_pdf(
    pdf_url: str,
    out_path: Path,
    timeout: float = 60.0,
) -> Path:
    """
    Download PDF from URL to local file.

    Args:
        pdf_url: URL to download PDF from
        out_path: Path to save the PDF
        timeout: Request timeout in seconds

    Returns:
        Path to the downloaded file

    Raises:
        PDFFetchError: If download fails
    """
    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    user_agent = os.environ.get("PAPERCLI_PDF_USER_AGENT") or (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    headers = {
        # Some hosts (e.g., preprint servers) may block default http clients.
        "User-Agent": user_agent,
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        try:
            response = await client.get(pdf_url)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                # Some servers don't set correct content type, check magic bytes
                if not response.content.startswith(b"%PDF"):
                    raise PDFFetchError(
                        f"Downloaded content is not a PDF (content-type: {content_type})",
                        source="download",
                    )

            # Write to file
            out_path.write_bytes(response.content)
            return out_path

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            reason = (getattr(e.response, "reason_phrase", None) or "").strip()
            url = str(e.response.url)
            if status == 403:
                raise PDFFetchError(
                    "HTTP 403 Forbidden downloading PDF. The host may be blocking automated downloads "
                    f"(anti-bot). URL: {url}. "
                    "Try setting PAPERCLI_PDF_USER_AGENT to a browser User-Agent, or download via the landing page.",
                    source="download",
                ) from e
            raise PDFFetchError(
                f"HTTP error downloading PDF: {status}{(' ' + reason) if reason else ''} (URL: {url})",
                source="download",
            ) from e
        except httpx.RequestError as e:
            raise PDFFetchError(
                f"Network error downloading PDF: {e}",
                source="download",
            ) from e


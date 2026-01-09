"""Tests for PDF fetching via Unpaywall and PMC APIs."""

import pytest
import respx
from httpx import Response

from papercli.pdf_fetch import (
    PDFResult,
    UnpaywallClient,
    PMCClient,
    fetch_pdf_url,
    validate_doi,
    doi_to_filename,
    download_pdf,
    PDFFetchError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""

    class MockUnpaywallConfig:
        base_url = "https://api.unpaywall.org/v2"
        email = "test@example.com"
        timeout = 30.0

    class MockAPIKeysConfig:
        ncbi_api_key = None

    class MockSettings:
        unpaywall = MockUnpaywallConfig()
        api_keys = MockAPIKeysConfig()

        def get_unpaywall_email(self):
            return "test@example.com"

        def get_ncbi_api_key(self):
            return None

    return MockSettings()


@pytest.fixture
def mock_settings_with_ncbi_key():
    """Create mock settings with NCBI API key."""

    class MockUnpaywallConfig:
        base_url = "https://api.unpaywall.org/v2"
        email = "test@example.com"
        timeout = 30.0

    class MockAPIKeysConfig:
        ncbi_api_key = "test-ncbi-key"

    class MockSettings:
        unpaywall = MockUnpaywallConfig()
        api_keys = MockAPIKeysConfig()

        def get_unpaywall_email(self):
            return "test@example.com"

        def get_ncbi_api_key(self):
            return "test-ncbi-key"

    return MockSettings()


# =============================================================================
# DOI Validation Tests
# =============================================================================


class TestValidateDoi:
    """Tests for DOI validation and normalization."""

    def test_valid_doi(self):
        """Test valid DOI passes validation."""
        assert validate_doi("10.1038/nature12373") == "10.1038/nature12373"

    def test_doi_with_https_prefix(self):
        """Test DOI with https://doi.org/ prefix."""
        assert validate_doi("https://doi.org/10.1038/nature12373") == "10.1038/nature12373"

    def test_doi_with_http_prefix(self):
        """Test DOI with http://doi.org/ prefix."""
        assert validate_doi("http://doi.org/10.1038/nature12373") == "10.1038/nature12373"

    def test_doi_with_doi_prefix(self):
        """Test DOI with doi: prefix."""
        assert validate_doi("doi:10.1038/nature12373") == "10.1038/nature12373"

    def test_doi_with_whitespace(self):
        """Test DOI with surrounding whitespace."""
        assert validate_doi("  10.1038/nature12373  ") == "10.1038/nature12373"

    def test_complex_doi(self):
        """Test DOI with complex suffix."""
        assert validate_doi("10.1000/xyz123-456.789") == "10.1000/xyz123-456.789"

    def test_invalid_doi_no_prefix(self):
        """Test invalid DOI without 10. prefix."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("nature12373")

    def test_invalid_doi_no_slash(self):
        """Test invalid DOI without slash."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("10.1038")

    def test_invalid_doi_short_prefix(self):
        """Test invalid DOI with short registrant."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            validate_doi("10.10/abc")


class TestDoiToFilename:
    """Tests for DOI to filename conversion."""

    def test_simple_doi(self):
        """Test simple DOI conversion."""
        assert doi_to_filename("10.1038/nature12373") == "10.1038_nature12373.pdf"

    def test_doi_with_special_chars(self):
        """Test DOI with special characters."""
        result = doi_to_filename("10.1000/xyz:123<456>789")
        assert "/" not in result
        assert ":" not in result
        assert "<" not in result
        assert ">" not in result
        assert result.endswith(".pdf")


# =============================================================================
# Unpaywall Client Tests
# =============================================================================


class TestUnpaywallClient:
    """Tests for Unpaywall API client."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_with_pdf_url(self, mock_settings):
        """Test successful lookup with direct PDF URL."""
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(
                200,
                json={
                    "doi": "10.1038/nature12373",
                    "is_oa": True,
                    "oa_status": "gold",
                    "best_oa_location": {
                        "url_for_pdf": "https://example.com/paper.pdf",
                        "url": "https://example.com/paper",
                        "license": "cc-by",
                    },
                },
            )
        )

        client = UnpaywallClient(mock_settings)
        try:
            result = await client.lookup("10.1038/nature12373")

            assert result.doi == "10.1038/nature12373"
            assert result.pdf_url == "https://example.com/paper.pdf"
            assert result.landing_url == "https://example.com/paper"
            assert result.source == "unpaywall"
            assert result.oa_status == "gold"
            assert result.license == "cc-by"
            assert result.error is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_oa_without_pdf(self, mock_settings):
        """Test OA article without direct PDF link."""
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(
                200,
                json={
                    "doi": "10.1038/nature12373",
                    "is_oa": True,
                    "oa_status": "green",
                    "best_oa_location": {
                        "url": "https://repository.example.com/paper",
                    },
                },
            )
        )

        client = UnpaywallClient(mock_settings)
        try:
            result = await client.lookup("10.1038/nature12373")

            assert result.pdf_url is None
            assert result.landing_url == "https://repository.example.com/paper"
            assert result.source == "unpaywall"
            assert "No direct PDF link" in result.error
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_not_oa(self, mock_settings):
        """Test non-OA article."""
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(
                200,
                json={
                    "doi": "10.1038/nature12373",
                    "is_oa": False,
                    "oa_status": "closed",
                    "doi_url": "https://doi.org/10.1038/nature12373",
                },
            )
        )

        client = UnpaywallClient(mock_settings)
        try:
            result = await client.lookup("10.1038/nature12373")

            assert result.pdf_url is None
            assert result.source == "none"
            assert result.oa_status == "closed"
            assert "Not open access" in result.error
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_not_found(self, mock_settings):
        """Test DOI not found in Unpaywall."""
        respx.get("https://api.unpaywall.org/v2/10.9999/notfound").mock(
            return_value=Response(404)
        )

        client = UnpaywallClient(mock_settings)
        try:
            result = await client.lookup("10.9999/notfound")

            assert result.pdf_url is None
            assert result.source == "none"
            assert "not found" in result.error.lower()
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_server_error(self, mock_settings):
        """Test handling of server error."""
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(500)
        )

        client = UnpaywallClient(mock_settings)
        try:
            result = await client.lookup("10.1038/nature12373")

            assert result.pdf_url is None
            assert result.source == "none"
            assert "HTTP 500" in result.error
        finally:
            await client.close()


# =============================================================================
# PMC Client Tests
# =============================================================================


class TestPMCClient:
    """Tests for PMC API client."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_doi_to_pmid_success(self, mock_settings):
        """Test successful DOI to PMID conversion."""
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(
                200,
                json={
                    "esearchresult": {
                        "idlist": ["12345678"],
                    },
                },
            )
        )

        client = PMCClient(mock_settings)
        try:
            pmid = await client.doi_to_pmid("10.1038/nature12373")
            assert pmid == "12345678"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_doi_to_pmid_not_found(self, mock_settings):
        """Test DOI not found in PubMed."""
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(
                200,
                json={
                    "esearchresult": {
                        "idlist": [],
                    },
                },
            )
        )

        client = PMCClient(mock_settings)
        try:
            pmid = await client.doi_to_pmid("10.9999/notfound")
            assert pmid is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_pmid_to_pmcid_success(self, mock_settings):
        """Test successful PMID to PMCID conversion."""
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi").mock(
            return_value=Response(
                200,
                json={
                    "linksets": [
                        {
                            "linksetdbs": [
                                {
                                    "dbto": "pmc",
                                    "links": ["7654321"],
                                },
                            ],
                        },
                    ],
                },
            )
        )

        client = PMCClient(mock_settings)
        try:
            pmcid = await client.pmid_to_pmcid("12345678")
            assert pmcid == "PMC7654321"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_pmid_to_pmcid_not_in_pmc(self, mock_settings):
        """Test PMID not available in PMC."""
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi").mock(
            return_value=Response(
                200,
                json={
                    "linksets": [
                        {
                            "linksetdbs": [],
                        },
                    ],
                },
            )
        )

        client = PMCClient(mock_settings)
        try:
            pmcid = await client.pmid_to_pmcid("12345678")
            assert pmcid is None
        finally:
            await client.close()

    def test_pmcid_to_pdf_url(self, mock_settings):
        """Test PMCID to PDF URL conversion."""
        client = PMCClient(mock_settings)
        url = client.pmcid_to_pdf_url("PMC7654321")
        assert url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/pdf/"

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_full_chain_success(self, mock_settings):
        """Test full DOI → PMID → PMCID → PDF chain."""
        # Mock esearch (DOI → PMID)
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(
                200,
                json={"esearchresult": {"idlist": ["12345678"]}},
            )
        )

        # Mock elink (PMID → PMCID)
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi").mock(
            return_value=Response(
                200,
                json={
                    "linksets": [
                        {"linksetdbs": [{"dbto": "pmc", "links": ["7654321"]}]},
                    ],
                },
            )
        )

        client = PMCClient(mock_settings)
        try:
            result = await client.lookup("10.1038/nature12373")

            assert result.doi == "10.1038/nature12373"
            assert result.pmid == "12345678"
            assert result.pmcid == "PMC7654321"
            assert result.pdf_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/pdf/"
            assert result.source == "pmc"
            assert result.error is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_doi_not_in_pubmed(self, mock_settings):
        """Test DOI not found in PubMed."""
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(
                200,
                json={"esearchresult": {"idlist": []}},
            )
        )

        client = PMCClient(mock_settings)
        try:
            result = await client.lookup("10.9999/notfound")

            assert result.pdf_url is None
            assert result.source == "none"
            assert "not found in PubMed" in result.error
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_lookup_not_in_pmc(self, mock_settings):
        """Test article in PubMed but not in PMC."""
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(
                200,
                json={"esearchresult": {"idlist": ["12345678"]}},
            )
        )

        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi").mock(
            return_value=Response(
                200,
                json={"linksets": [{"linksetdbs": []}]},
            )
        )

        client = PMCClient(mock_settings)
        try:
            result = await client.lookup("10.1038/nature12373")

            assert result.pdf_url is None
            assert result.pmid == "12345678"
            assert result.pmcid is None
            assert result.source == "none"
            assert "not available in PMC" in result.error
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_key_added_to_requests(self, mock_settings_with_ncbi_key):
        """Test that NCBI API key is added to requests."""
        esearch_route = respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(200, json={"esearchresult": {"idlist": []}})
        )

        client = PMCClient(mock_settings_with_ncbi_key)
        try:
            await client.doi_to_pmid("10.1038/nature12373")

            # Check that api_key was in the request params
            assert esearch_route.called
            request = esearch_route.calls[0].request
            assert "api_key=test-ncbi-key" in str(request.url)
        finally:
            await client.close()


# =============================================================================
# Integration Tests for fetch_pdf_url
# =============================================================================


class TestFetchPdfUrl:
    """Tests for the main fetch_pdf_url function."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_unpaywall_success(self, mock_settings):
        """Test successful fetch via Unpaywall."""
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(
                200,
                json={
                    "doi": "10.1038/nature12373",
                    "is_oa": True,
                    "oa_status": "gold",
                    "best_oa_location": {
                        "url_for_pdf": "https://example.com/paper.pdf",
                    },
                },
            )
        )

        result = await fetch_pdf_url("10.1038/nature12373", mock_settings)

        assert result.pdf_url == "https://example.com/paper.pdf"
        assert result.source == "unpaywall"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fallback_to_pmc(self, mock_settings):
        """Test fallback to PMC when Unpaywall has no PDF."""
        # Unpaywall returns no PDF
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(
                200,
                json={
                    "doi": "10.1038/nature12373",
                    "is_oa": False,
                    "oa_status": "closed",
                },
            )
        )

        # PMC chain succeeds
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(200, json={"esearchresult": {"idlist": ["12345678"]}})
        )
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi").mock(
            return_value=Response(
                200,
                json={"linksets": [{"linksetdbs": [{"dbto": "pmc", "links": ["7654321"]}]}]},
            )
        )

        result = await fetch_pdf_url("10.1038/nature12373", mock_settings)

        assert result.pdf_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/pdf/"
        assert result.source == "pmc"
        # Should keep Unpaywall metadata
        assert result.oa_status == "closed"

    @pytest.mark.asyncio
    @respx.mock
    async def test_skip_unpaywall(self, mock_settings):
        """Test skipping Unpaywall and using PMC directly."""
        # PMC chain succeeds
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            return_value=Response(200, json={"esearchresult": {"idlist": ["12345678"]}})
        )
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi").mock(
            return_value=Response(
                200,
                json={"linksets": [{"linksetdbs": [{"dbto": "pmc", "links": ["7654321"]}]}]},
            )
        )

        result = await fetch_pdf_url(
            "10.1038/nature12373",
            mock_settings,
            skip_unpaywall=True,
        )

        assert result.pdf_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/pdf/"
        assert result.source == "pmc"

    @pytest.mark.asyncio
    @respx.mock
    async def test_skip_pmc(self, mock_settings):
        """Test skipping PMC fallback."""
        # Unpaywall returns no PDF
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(
                200,
                json={
                    "doi": "10.1038/nature12373",
                    "is_oa": False,
                    "oa_status": "closed",
                },
            )
        )

        result = await fetch_pdf_url(
            "10.1038/nature12373",
            mock_settings,
            skip_pmc=True,
        )

        assert result.pdf_url is None
        assert result.source == "none"

    @pytest.mark.asyncio
    @respx.mock
    async def test_doi_validation(self, mock_settings):
        """Test that DOI is validated and normalized."""
        respx.get("https://api.unpaywall.org/v2/10.1038/nature12373").mock(
            return_value=Response(
                200,
                json={
                    "doi": "10.1038/nature12373",
                    "is_oa": True,
                    "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"},
                },
            )
        )

        # Test with URL prefix
        result = await fetch_pdf_url("https://doi.org/10.1038/nature12373", mock_settings)
        assert result.doi == "10.1038/nature12373"

    @pytest.mark.asyncio
    async def test_invalid_doi_raises(self, mock_settings):
        """Test that invalid DOI raises ValueError."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            await fetch_pdf_url("invalid-doi", mock_settings)


# =============================================================================
# Download Tests
# =============================================================================


class TestDownloadPdf:
    """Tests for PDF download functionality."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_success(self, tmp_path):
        """Test successful PDF download."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        respx.get("https://example.com/paper.pdf").mock(
            return_value=Response(
                200,
                content=pdf_content,
                headers={"content-type": "application/pdf"},
            )
        )

        out_path = tmp_path / "downloaded.pdf"
        result = await download_pdf("https://example.com/paper.pdf", out_path)

        assert result == out_path
        assert out_path.exists()
        assert out_path.read_bytes() == pdf_content

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_creates_directory(self, tmp_path):
        """Test that download creates parent directories."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        respx.get("https://example.com/paper.pdf").mock(
            return_value=Response(
                200,
                content=pdf_content,
                headers={"content-type": "application/pdf"},
            )
        )

        out_path = tmp_path / "subdir" / "nested" / "downloaded.pdf"
        result = await download_pdf("https://example.com/paper.pdf", out_path)

        assert result == out_path
        assert out_path.exists()

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_accepts_octet_stream(self, tmp_path):
        """Test download accepts application/octet-stream content type."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        respx.get("https://example.com/paper.pdf").mock(
            return_value=Response(
                200,
                content=pdf_content,
                headers={"content-type": "application/octet-stream"},
            )
        )

        out_path = tmp_path / "downloaded.pdf"
        result = await download_pdf("https://example.com/paper.pdf", out_path)

        assert result == out_path
        assert out_path.exists()

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_checks_magic_bytes(self, tmp_path):
        """Test download validates PDF magic bytes if content-type is wrong."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        respx.get("https://example.com/paper.pdf").mock(
            return_value=Response(
                200,
                content=pdf_content,
                headers={"content-type": "text/html"},  # Wrong content type
            )
        )

        out_path = tmp_path / "downloaded.pdf"
        # Should succeed because content starts with %PDF
        result = await download_pdf("https://example.com/paper.pdf", out_path)
        assert result == out_path

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_rejects_non_pdf(self, tmp_path):
        """Test download rejects non-PDF content."""
        html_content = b"<html><body>Not a PDF</body></html>"
        respx.get("https://example.com/paper.pdf").mock(
            return_value=Response(
                200,
                content=html_content,
                headers={"content-type": "text/html"},
            )
        )

        out_path = tmp_path / "downloaded.pdf"
        with pytest.raises(PDFFetchError, match="not a PDF"):
            await download_pdf("https://example.com/paper.pdf", out_path)

    @pytest.mark.asyncio
    @respx.mock
    async def test_download_http_error(self, tmp_path):
        """Test download handles HTTP errors."""
        respx.get("https://example.com/paper.pdf").mock(
            return_value=Response(404)
        )

        out_path = tmp_path / "downloaded.pdf"
        with pytest.raises(PDFFetchError, match="HTTP error"):
            await download_pdf("https://example.com/paper.pdf", out_path)


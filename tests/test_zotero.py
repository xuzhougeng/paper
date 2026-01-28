"""Tests for Zotero source adapter."""

import pytest
import respx
from httpx import Response

from papercli.models import QueryIntent
from papercli.sources.zotero import ZoteroSource, _format_creator_name


# Sample Zotero API response
SAMPLE_ZOTERO_ITEMS = [
    {
        "key": "ABC12345",
        "version": 1,
        "library": {"type": "user", "id": 12345678},
        "data": {
            "key": "ABC12345",
            "version": 1,
            "itemType": "journalArticle",
            "title": "Deep Learning for Protein Structure Prediction",
            "creators": [
                {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
                {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"},
            ],
            "abstractNote": "We present a novel deep learning approach for protein folding.",
            "publicationTitle": "Nature",
            "date": "2023-06-15",
            "DOI": "10.1038/s41586-023-12345",
            "url": "https://www.nature.com/articles/s41586-023-12345",
            "tags": [{"tag": "deep learning"}, {"tag": "protein"}],
        },
    },
    {
        "key": "XYZ98765",
        "version": 1,
        "library": {"type": "user", "id": 12345678},
        "data": {
            "key": "XYZ98765",
            "version": 1,
            "itemType": "journalArticle",
            "title": "Machine Learning in Drug Discovery",
            "creators": [
                {"creatorType": "author", "name": "DeepMind Team"},
            ],
            "abstractNote": "This review covers ML applications in pharmaceutical research.",
            "publicationTitle": "Cell",
            "date": "2022",
            "DOI": "https://doi.org/10.1016/j.cell.2022.12345",
            "url": "",
            "tags": [],
        },
    },
]

MINIMAL_ZOTERO_ITEM = [
    {
        "key": "MIN00001",
        "data": {
            "key": "MIN00001",
            "itemType": "book",
            "title": "Minimal Book Entry",
            "creators": [],
            "date": "",
        },
    },
]


class TestZoteroParsing:
    """Tests for Zotero item parsing."""

    def test_parse_complete_item(self):
        """Test parsing a complete Zotero item."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        paper = source._parse_item(SAMPLE_ZOTERO_ITEMS[0])

        assert paper is not None
        assert paper.source == "zotero"
        assert paper.source_id == "ABC12345"
        assert paper.title == "Deep Learning for Protein Structure Prediction"
        assert paper.abstract == "We present a novel deep learning approach for protein folding."
        assert paper.year == 2023
        assert paper.authors == ["John Smith", "Jane Doe"]
        assert paper.doi == "10.1038/s41586-023-12345"
        assert paper.url == "https://doi.org/10.1038/s41586-023-12345"
        assert paper.venue == "Nature"

    def test_parse_item_with_single_field_name(self):
        """Test parsing item with single-field creator name (institutional author)."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        paper = source._parse_item(SAMPLE_ZOTERO_ITEMS[1])

        assert paper is not None
        assert paper.authors == ["DeepMind Team"]

    def test_parse_item_doi_normalization(self):
        """Test that DOI is normalized (URL prefix stripped)."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        paper = source._parse_item(SAMPLE_ZOTERO_ITEMS[1])

        assert paper is not None
        assert paper.doi == "10.1016/j.cell.2022.12345"

    def test_parse_minimal_item(self):
        """Test parsing item with minimal data."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        paper = source._parse_item(MINIMAL_ZOTERO_ITEM[0])

        assert paper is not None
        assert paper.source_id == "MIN00001"
        assert paper.title == "Minimal Book Entry"
        assert paper.abstract is None
        assert paper.authors == []
        assert paper.year is None
        assert paper.doi is None

    def test_parse_item_no_title(self):
        """Test that items without title are skipped."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        item = {"data": {"key": "NOTITLE1"}}
        paper = source._parse_item(item)
        assert paper is None

    def test_parse_item_no_key(self):
        """Test that items without key are skipped."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        item = {"data": {"title": "No Key Paper"}}
        paper = source._parse_item(item)
        assert paper is None


class TestYearExtraction:
    """Tests for year extraction from various date formats."""

    def test_year_full_date(self):
        """Test year extraction from full date."""
        assert ZoteroSource._extract_year("2023-06-15") == 2023

    def test_year_only(self):
        """Test year extraction from year only."""
        assert ZoteroSource._extract_year("2022") == 2022

    def test_year_month_year(self):
        """Test year extraction from 'Month Year' format."""
        assert ZoteroSource._extract_year("January 2021") == 2021

    def test_year_year_month(self):
        """Test year extraction from 'Year Month' format."""
        assert ZoteroSource._extract_year("2020 Jan") == 2020

    def test_year_slash_date(self):
        """Test year extraction from slash-separated date."""
        assert ZoteroSource._extract_year("2019/03/15") == 2019

    def test_year_empty(self):
        """Test year extraction from empty string."""
        assert ZoteroSource._extract_year("") is None

    def test_year_none(self):
        """Test year extraction from None."""
        assert ZoteroSource._extract_year(None) is None

    def test_year_no_year(self):
        """Test year extraction from string without year."""
        assert ZoteroSource._extract_year("n.d.") is None


class TestCreatorFormatting:
    """Tests for creator name formatting."""

    def test_format_two_field_name(self):
        """Test formatting with firstName and lastName."""
        creator = {"firstName": "John", "lastName": "Smith"}
        assert _format_creator_name(creator) == "John Smith"

    def test_format_single_field_name(self):
        """Test formatting with single name field (institutional)."""
        creator = {"name": "World Health Organization"}
        assert _format_creator_name(creator) == "World Health Organization"

    def test_format_last_name_only(self):
        """Test formatting with only lastName."""
        creator = {"lastName": "Anonymous"}
        assert _format_creator_name(creator) == "Anonymous"

    def test_format_first_name_only(self):
        """Test formatting with only firstName."""
        creator = {"firstName": "Prince"}
        assert _format_creator_name(creator) == "Prince"

    def test_format_empty_creator(self):
        """Test formatting empty creator dict."""
        assert _format_creator_name({}) is None


class TestAuthorExtraction:
    """Tests for author extraction from creators list."""

    def test_extract_authors_only(self):
        """Test extracting only authors from mixed creators."""
        creators = [
            {"creatorType": "author", "firstName": "John", "lastName": "Doe"},
            {"creatorType": "editor", "firstName": "Jane", "lastName": "Smith"},
            {"creatorType": "author", "firstName": "Bob", "lastName": "Wilson"},
        ]
        authors = ZoteroSource._extract_authors(creators)
        assert authors == ["John Doe", "Bob Wilson"]

    def test_extract_fallback_to_all_creators(self):
        """Test fallback to all creators when no authors present."""
        creators = [
            {"creatorType": "editor", "firstName": "Jane", "lastName": "Smith"},
            {"creatorType": "translator", "firstName": "Bob", "lastName": "Wilson"},
        ]
        authors = ZoteroSource._extract_authors(creators)
        assert authors == ["Jane Smith", "Bob Wilson"]

    def test_extract_empty_creators(self):
        """Test extraction from empty creators list."""
        assert ZoteroSource._extract_authors([]) == []


class TestZoteroSearch:
    """Tests for Zotero search functionality."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_returns_papers(self):
        """Test successful search returns parsed papers."""
        respx.get("https://api.zotero.org/users/12345678/items").mock(
            return_value=Response(200, json=SAMPLE_ZOTERO_ITEMS)
        )

        source = ZoteroSource(api_key="test-key", user_id="12345678")
        intent = QueryIntent(
            reasoning="User wants papers about deep learning.",
            query_en="deep learning protein",
            keywords=["deep learning", "protein"],
        )

        papers = await source.search(intent, max_results=10)

        assert len(papers) == 2
        assert papers[0].title == "Deep Learning for Protein Structure Prediction"
        assert papers[1].title == "Machine Learning in Drug Discovery"

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_with_required_phrases(self):
        """Test search query includes required phrases."""
        route = respx.get("https://api.zotero.org/users/12345678/items").mock(
            return_value=Response(200, json=[])
        )

        source = ZoteroSource(api_key="test-key", user_id="12345678")
        intent = QueryIntent(
            reasoning="User wants specific protein folding papers.",
            query_en="protein folding",
            keywords=["protein", "folding"],
            required_phrases=["AlphaFold"],
        )

        await source.search(intent, max_results=10)

        # Check that the query parameter includes the required phrase
        assert route.called
        request = route.calls.last.request
        assert "AlphaFold" in str(request.url)

    @pytest.mark.asyncio
    async def test_search_no_api_key(self):
        """Test search returns empty list when API key is missing."""
        source = ZoteroSource(api_key=None, user_id="12345678")
        intent = QueryIntent(
            reasoning="Test",
            query_en="test query",
            keywords=["test"],
        )

        papers = await source.search(intent)
        assert papers == []

    @pytest.mark.asyncio
    async def test_search_no_user_id(self):
        """Test search returns empty list when user ID is missing."""
        source = ZoteroSource(api_key="test-key", user_id=None)
        intent = QueryIntent(
            reasoning="Test",
            query_en="test query",
            keywords=["test"],
        )

        papers = await source.search(intent)
        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_empty_results(self):
        """Test search with no results."""
        respx.get("https://api.zotero.org/users/12345678/items").mock(
            return_value=Response(200, json=[])
        )

        source = ZoteroSource(api_key="test-key", user_id="12345678")
        intent = QueryIntent(
            reasoning="Test",
            query_en="nonexistent topic xyz",
            keywords=["nonexistent"],
        )

        papers = await source.search(intent)
        assert papers == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_api_error(self):
        """Test search handles API errors gracefully."""
        respx.get("https://api.zotero.org/users/12345678/items").mock(
            return_value=Response(500)
        )

        source = ZoteroSource(api_key="test-key", user_id="12345678")
        intent = QueryIntent(
            reasoning="Test",
            query_en="test query",
            keywords=["test"],
        )

        with pytest.raises(Exception):
            await source.search(intent)


class TestZoteroQueryBuilding:
    """Tests for Zotero query construction."""

    def test_simple_query(self):
        """Test building simple query."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        intent = QueryIntent(
            reasoning="User wants CRISPR papers.",
            query_en="CRISPR gene editing",
            keywords=["CRISPR", "gene", "editing"],
        )

        query = source._build_query(intent)
        assert "CRISPR gene editing" in query

    def test_query_with_required_phrases(self):
        """Test query with required phrases."""
        source = ZoteroSource(api_key="test", user_id="12345678")
        intent = QueryIntent(
            reasoning="User wants specific cancer therapy papers.",
            query_en="cancer therapy",
            keywords=["cancer", "therapy"],
            required_phrases=["immunotherapy", "CAR-T"],
        )

        query = source._build_query(intent)
        assert "cancer therapy" in query
        assert '"immunotherapy"' in query
        assert '"CAR-T"' in query

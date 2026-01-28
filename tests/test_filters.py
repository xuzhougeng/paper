"""Tests for year and venue filtering across sources and pipeline."""

import pytest

from papercli.models import Paper, QueryIntent


class TestPubMedFiltering:
    """Tests for PubMed year and venue filtering."""

    def test_build_query_with_venue(self):
        """Test that venue filter is included in PubMed query."""
        from papercli.sources.pubmed import PubMedSource

        source = PubMedSource()
        intent = QueryIntent(
            reasoning="User wants bioinformatics papers.",
            query_en="RNA-seq analysis",
            keywords=["RNA-seq", "analysis"],
            venue="Bioinformatics",
        )

        query = source._build_query(intent)
        assert '"Bioinformatics"[Journal]' in query
        assert "RNA-seq analysis" in query

    def test_build_query_without_venue(self):
        """Test that query without venue filter works."""
        from papercli.sources.pubmed import PubMedSource

        source = PubMedSource()
        intent = QueryIntent(
            reasoning="User wants RNA-seq papers.",
            query_en="RNA-seq analysis",
            keywords=["RNA-seq", "analysis"],
        )

        query = source._build_query(intent)
        assert "[Journal]" not in query
        assert "RNA-seq analysis" in query

    def test_year_filter_key_exact(self):
        """Test cache key generation for exact year filter."""
        from papercli.sources.pubmed import PubMedSource

        source = PubMedSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year=2025,
        )

        key = source._get_year_filter_key(intent)
        assert "y2025" in key

    def test_year_filter_key_range(self):
        """Test cache key generation for year range filter."""
        from papercli.sources.pubmed import PubMedSource

        source = PubMedSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year_min=2020,
            year_max=2025,
        )

        key = source._get_year_filter_key(intent)
        assert "ymin2020" in key
        assert "ymax2025" in key

    def test_year_filter_key_with_venue(self):
        """Test cache key generation with venue filter."""
        from papercli.sources.pubmed import PubMedSource

        source = PubMedSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year=2025,
            venue="Nature",
        )

        key = source._get_year_filter_key(intent)
        assert "y2025" in key
        assert "vNature" in key


class TestOpenAlexFiltering:
    """Tests for OpenAlex year and venue filtering."""

    def test_build_filter_exact_year(self):
        """Test building filter for exact year."""
        from papercli.sources.openalex import OpenAlexSource

        source = OpenAlexSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year=2025,
        )

        filters = source._build_filter(intent)
        assert "publication_year:2025" in filters

    def test_build_filter_year_range(self):
        """Test building filter for year range."""
        from papercli.sources.openalex import OpenAlexSource

        source = OpenAlexSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year_min=2020,
            year_max=2025,
        )

        filters = source._build_filter(intent)
        assert "publication_year:2020-2025" in filters

    def test_build_filter_year_min_only(self):
        """Test building filter with only min year."""
        from papercli.sources.openalex import OpenAlexSource

        source = OpenAlexSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year_min=2020,
        )

        filters = source._build_filter(intent)
        assert "publication_year:>2019" in filters

    def test_build_filter_venue(self):
        """Test building filter for venue."""
        from papercli.sources.openalex import OpenAlexSource

        source = OpenAlexSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            venue="Bioinformatics",
        )

        filters = source._build_filter(intent)
        assert any("Bioinformatics" in f for f in filters)

    def test_build_filter_combined(self):
        """Test building filter with year and venue."""
        from papercli.sources.openalex import OpenAlexSource

        source = OpenAlexSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year=2025,
            venue="Nature",
        )

        filters = source._build_filter(intent)
        assert len(filters) == 2
        assert "publication_year:2025" in filters

    def test_no_filters(self):
        """Test that empty filter list is returned when no filters."""
        from papercli.sources.openalex import OpenAlexSource

        source = OpenAlexSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
        )

        filters = source._build_filter(intent)
        assert filters == []


class TestScholarFiltering:
    """Tests for Google Scholar year filtering."""

    def test_filter_key_with_year(self):
        """Test cache key with year filter."""
        from papercli.sources.scholar import ScholarSource

        source = ScholarSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year=2025,
        )

        key = source._get_filter_key(intent)
        assert "y2025" in key

    def test_filter_key_with_range(self):
        """Test cache key with year range."""
        from papercli.sources.scholar import ScholarSource

        source = ScholarSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year_min=2020,
            year_max=2025,
        )

        key = source._get_filter_key(intent)
        assert "ymin2020" in key
        assert "ymax2025" in key


class TestArxivFiltering:
    """Tests for arXiv date filtering."""

    def test_build_date_filter_exact_year(self):
        """Test building date filter for exact year."""
        from papercli.sources.arxiv import ArxivSource

        source = ArxivSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year=2025,
        )

        date_filter = source._build_date_filter(intent)
        assert date_filter == "submittedDate:[20250101 TO 20251231]"

    def test_build_date_filter_year_range(self):
        """Test building date filter for year range."""
        from papercli.sources.arxiv import ArxivSource

        source = ArxivSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year_min=2020,
            year_max=2025,
        )

        date_filter = source._build_date_filter(intent)
        assert "20200101" in date_filter
        assert "20251231" in date_filter

    def test_build_date_filter_min_only(self):
        """Test building date filter with only min year."""
        from papercli.sources.arxiv import ArxivSource

        source = ArxivSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
            year_min=2020,
        )

        date_filter = source._build_date_filter(intent)
        assert "20200101" in date_filter
        assert "TO *" in date_filter

    def test_no_date_filter(self):
        """Test that None is returned when no date filter."""
        from papercli.sources.arxiv import ArxivSource

        source = ArxivSource()
        intent = QueryIntent(
            reasoning="",
            query_en="test",
        )

        date_filter = source._build_date_filter(intent)
        assert date_filter is None


class TestPipelineFiltering:
    """Tests for pipeline post-filtering."""

    @pytest.fixture
    def sample_papers_for_filtering(self):
        """Create sample papers with various years and venues."""
        return [
            Paper(
                source="pubmed",
                source_id="1",
                title="Paper 2025 Bioinformatics",
                year=2025,
                venue="Bioinformatics",
            ),
            Paper(
                source="pubmed",
                source_id="2",
                title="Paper 2024 Bioinformatics",
                year=2024,
                venue="Bioinformatics",
            ),
            Paper(
                source="pubmed",
                source_id="3",
                title="Paper 2025 Nature",
                year=2025,
                venue="Nature",
            ),
            Paper(
                source="pubmed",
                source_id="4",
                title="Paper 2023 Cell",
                year=2023,
                venue="Cell",
            ),
            Paper(
                source="pubmed",
                source_id="5",
                title="Paper no year",
                year=None,
                venue="Science",
            ),
            Paper(
                source="pubmed",
                source_id="6",
                title="Paper no venue",
                year=2025,
                venue=None,
            ),
        ]

    def test_filter_by_exact_year(self, sample_papers_for_filtering):
        """Test filtering by exact year."""
        from papercli.pipeline import apply_filters

        intent = QueryIntent(
            reasoning="",
            query_en="",
            year=2025,
        )

        filtered = apply_filters(sample_papers_for_filtering, intent)
        assert len(filtered) == 3  # Papers 1, 3, 6
        assert all(p.year == 2025 for p in filtered)

    def test_filter_by_year_range(self, sample_papers_for_filtering):
        """Test filtering by year range."""
        from papercli.pipeline import apply_filters

        intent = QueryIntent(
            reasoning="",
            query_en="",
            year_min=2024,
            year_max=2025,
        )

        filtered = apply_filters(sample_papers_for_filtering, intent)
        assert len(filtered) == 4  # Papers 1, 2, 3, 6
        assert all(p.year in (2024, 2025) for p in filtered)

    def test_filter_by_venue(self, sample_papers_for_filtering):
        """Test filtering by venue (case-insensitive)."""
        from papercli.pipeline import apply_filters

        intent = QueryIntent(
            reasoning="",
            query_en="",
            venue="bioinformatics",  # lowercase
        )

        filtered = apply_filters(sample_papers_for_filtering, intent)
        assert len(filtered) == 2  # Papers 1, 2
        assert all("bioinformatics" in p.venue.lower() for p in filtered)

    def test_filter_by_year_and_venue(self, sample_papers_for_filtering):
        """Test filtering by both year and venue."""
        from papercli.pipeline import apply_filters

        intent = QueryIntent(
            reasoning="",
            query_en="",
            year=2025,
            venue="Bioinformatics",
        )

        filtered = apply_filters(sample_papers_for_filtering, intent)
        assert len(filtered) == 1
        assert filtered[0].source_id == "1"

    def test_no_filters(self, sample_papers_for_filtering):
        """Test that all papers pass when no filters."""
        from papercli.pipeline import apply_filters

        intent = QueryIntent(
            reasoning="",
            query_en="",
        )

        filtered = apply_filters(sample_papers_for_filtering, intent)
        assert len(filtered) == len(sample_papers_for_filtering)

    def test_empty_papers_list(self):
        """Test filtering empty list."""
        from papercli.pipeline import apply_filters

        intent = QueryIntent(
            reasoning="",
            query_en="",
            year=2025,
        )

        filtered = apply_filters([], intent)
        assert filtered == []

    def test_partial_venue_match(self, sample_papers_for_filtering):
        """Test that partial venue match works."""
        from papercli.pipeline import apply_filters

        # Add a paper with longer venue name
        papers = sample_papers_for_filtering + [
            Paper(
                source="pubmed",
                source_id="7",
                title="Paper with full venue",
                year=2025,
                venue="Bioinformatics (Oxford, England)",
            ),
        ]

        intent = QueryIntent(
            reasoning="",
            query_en="",
            venue="Bioinformatics",
        )

        filtered = apply_filters(papers, intent)
        assert len(filtered) == 3  # Papers 1, 2, 7

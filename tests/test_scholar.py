"""Tests for Google Scholar (SerpAPI) parsing."""

import pytest

from papercli.sources.scholar import ScholarSource


SAMPLE_SERPAPI_RESULTS = [
    {
        "result_id": "abc123",
        "title": "Deep Learning for Medical Image Analysis",
        "snippet": "We present a comprehensive review of deep learning methods applied to medical imaging...",
        "link": "https://example.com/paper1",
        "publication_info": {
            "summary": "J Smith, A Doe - Nature Medicine, 2023",
            "authors": [
                {"name": "J Smith"},
                {"name": "A Doe"},
            ],
        },
    },
    {
        "title": "Minimal Paper Without ID",
        "snippet": "Short description.",
        "publication_info": {
            "summary": "Unknown Author - 2022",
        },
    },
]


class TestScholarParsing:
    """Tests for SerpAPI Scholar results parsing."""

    def test_parse_complete_result(self):
        """Test parsing a complete Scholar result."""
        source = ScholarSource()
        papers = source._parse_results(SAMPLE_SERPAPI_RESULTS)

        assert len(papers) == 2

        paper = papers[0]
        assert paper.source == "scholar"
        assert paper.source_id == "abc123"
        assert paper.title == "Deep Learning for Medical Image Analysis"
        assert "deep learning methods" in paper.abstract
        assert paper.url == "https://example.com/paper1"
        assert paper.year == 2023
        assert "J Smith" in paper.authors

    def test_parse_result_without_id(self):
        """Test parsing result without result_id (generates hash)."""
        source = ScholarSource()
        papers = source._parse_results(SAMPLE_SERPAPI_RESULTS)

        paper = papers[1]
        assert paper.source_id  # Should have generated ID
        assert len(paper.source_id) == 12  # MD5 hash prefix

    def test_year_extraction(self):
        """Test year extraction from publication summary."""
        source = ScholarSource()
        result = {
            "title": "Test Paper",
            "publication_info": {
                "summary": "Author Name - Some Journal, 2021",
            },
        }

        paper = source._parse_result(result)
        assert paper.year == 2021

    def test_venue_extraction(self):
        """Test venue extraction from publication summary."""
        source = ScholarSource()
        result = {
            "title": "Test Paper",
            "publication_info": {
                "summary": "J Doe - Nature Reviews, 2023",
            },
        }

        paper = source._parse_result(result)
        assert paper.venue == "Nature Reviews"

    def test_empty_results(self):
        """Test handling of empty results."""
        source = ScholarSource()
        papers = source._parse_results([])
        assert papers == []

    def test_result_without_title(self):
        """Test that results without title are skipped."""
        source = ScholarSource()
        result = {
            "snippet": "Some content but no title",
        }

        paper = source._parse_result(result)
        assert paper is None

    def test_authors_from_summary_fallback(self):
        """Test author extraction when authors list is missing."""
        source = ScholarSource()
        result = {
            "title": "Test Paper",
            "publication_info": {
                "summary": "John Smith, Jane Doe - Journal, 2023",
            },
        }

        paper = source._parse_result(result)
        assert len(paper.authors) >= 1


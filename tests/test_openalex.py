"""Tests for OpenAlex abstract reconstruction and parsing."""

import pytest

from papercli.sources.openalex import OpenAlexSource


class TestAbstractReconstruction:
    """Tests for OpenAlex inverted index to abstract conversion."""

    def test_simple_abstract(self):
        """Test basic abstract reconstruction."""
        inverted_index = {
            "The": [0],
            "quick": [1],
            "brown": [2],
            "fox": [3],
            "jumps": [4],
        }
        result = OpenAlexSource.reconstruct_abstract(inverted_index)
        assert result == "The quick brown fox jumps"

    def test_repeated_words(self):
        """Test abstract with repeated words."""
        inverted_index = {
            "The": [0, 5],
            "cat": [1, 6],
            "sat": [2],
            "on": [3],
            "the": [4],
            "mat": [7],
        }
        result = OpenAlexSource.reconstruct_abstract(inverted_index)
        # "The cat sat on the The cat mat"
        assert "The" in result
        assert "cat" in result
        assert "sat" in result

    def test_empty_index(self):
        """Test empty inverted index."""
        result = OpenAlexSource.reconstruct_abstract({})
        assert result == ""

    def test_none_index(self):
        """Test None inverted index."""
        result = OpenAlexSource.reconstruct_abstract(None)
        assert result == ""

    def test_scientific_abstract(self):
        """Test with realistic scientific terms."""
        inverted_index = {
            "We": [0],
            "present": [1],
            "a": [2, 8],
            "novel": [3],
            "method": [4],
            "for": [5],
            "protein": [6],
            "folding": [7],
            "using": [9],
            "deep": [10],
            "learning": [11],
        }
        result = OpenAlexSource.reconstruct_abstract(inverted_index)
        assert "novel method" in result
        assert "protein folding" in result
        assert "deep learning" in result


class TestOpenAlexParsing:
    """Tests for OpenAlex API response parsing."""

    def test_parse_work_complete(self):
        """Test parsing a complete work entry."""
        source = OpenAlexSource()
        work = {
            "id": "https://openalex.org/W2741809807",
            "title": "Test Paper Title",
            "abstract_inverted_index": {"Test": [0], "abstract": [1]},
            "publication_year": 2023,
            "authorships": [
                {"author": {"display_name": "John Doe"}},
                {"author": {"display_name": "Jane Smith"}},
            ],
            "doi": "https://doi.org/10.1234/test",
            "primary_location": {
                "source": {"display_name": "Nature"}
            },
        }

        paper = source._parse_work(work)

        assert paper is not None
        assert paper.source == "openalex"
        assert paper.source_id == "W2741809807"
        assert paper.title == "Test Paper Title"
        assert paper.abstract == "Test abstract"
        assert paper.year == 2023
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.doi == "10.1234/test"
        assert paper.venue == "Nature"

    def test_parse_work_minimal(self):
        """Test parsing work with minimal data."""
        source = OpenAlexSource()
        work = {
            "id": "https://openalex.org/W123",
            "title": "Minimal Paper",
        }

        paper = source._parse_work(work)

        assert paper is not None
        assert paper.source_id == "W123"
        assert paper.title == "Minimal Paper"
        assert paper.abstract is None
        assert paper.authors == []

    def test_parse_work_no_title(self):
        """Test that works without title are skipped."""
        source = OpenAlexSource()
        work = {
            "id": "https://openalex.org/W123",
        }

        paper = source._parse_work(work)
        assert paper is None

    def test_parse_work_no_id(self):
        """Test that works without ID are skipped."""
        source = OpenAlexSource()
        work = {
            "title": "No ID Paper",
        }

        paper = source._parse_work(work)
        assert paper is None


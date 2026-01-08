"""Tests for paper deduplication."""

import pytest

from papercli.models import Paper
from papercli.rank import deduplicate, _title_similarity


class TestTitleSimilarity:
    """Tests for title similarity calculation."""

    def test_identical_titles(self):
        """Test similarity of identical titles."""
        assert _title_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        """Test similarity of completely different titles."""
        assert _title_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        """Test similarity with partial word overlap."""
        # "hello world" vs "hello there" -> intersection={hello}, union={hello,world,there}
        sim = _title_similarity("hello world", "hello there")
        assert 0.3 < sim < 0.4  # 1/3 ≈ 0.33

    def test_empty_strings(self):
        """Test similarity with empty strings."""
        assert _title_similarity("", "") == 0.0
        assert _title_similarity("hello", "") == 0.0


class TestDeduplication:
    """Tests for paper deduplication logic."""

    def test_empty_list(self):
        """Test deduplication of empty list."""
        assert deduplicate([]) == []

    def test_no_duplicates(self):
        """Test list with no duplicates."""
        papers = [
            Paper(source="pubmed", source_id="1", title="Paper One", doi="10.1/one"),
            Paper(source="pubmed", source_id="2", title="Paper Two", doi="10.1/two"),
        ]
        result = deduplicate(papers)
        assert len(result) == 2

    def test_doi_duplicates(self):
        """Test deduplication by DOI."""
        papers = [
            Paper(source="pubmed", source_id="1", title="Paper A", doi="10.1/same"),
            Paper(source="openalex", source_id="W1", title="Paper A", doi="10.1/same"),
        ]
        result = deduplicate(papers)
        assert len(result) == 1
        assert result[0].doi == "10.1/same"

    def test_prefers_complete_metadata(self):
        """Test that deduplication prefers papers with more metadata."""
        papers = [
            Paper(source="pubmed", source_id="1", title="Test", doi="10.1/test"),
            Paper(
                source="openalex",
                source_id="W1",
                title="Test",
                doi="10.1/test",
                abstract="Full abstract here",
                authors=["John Doe"],
                year=2023,
            ),
        ]
        result = deduplicate(papers)
        assert len(result) == 1
        assert result[0].abstract == "Full abstract here"
        assert result[0].source == "openalex"

    def test_title_similarity_dedup(self):
        """Test deduplication by similar titles."""
        papers = [
            Paper(source="pubmed", source_id="1", title="machine learning for drug discovery"),
            Paper(source="openalex", source_id="W1", title="Machine Learning for Drug Discovery"),
        ]
        result = deduplicate(papers)
        # Should be deduplicated due to similar normalized titles
        assert len(result) == 1

    def test_different_titles_kept(self):
        """Test that sufficiently different titles are kept."""
        papers = [
            Paper(source="pubmed", source_id="1", title="Deep learning in medical imaging"),
            Paper(source="openalex", source_id="W1", title="Protein structure prediction methods"),
        ]
        result = deduplicate(papers)
        assert len(result) == 2

    def test_case_insensitive_doi(self):
        """Test that DOI matching is case-insensitive."""
        papers = [
            Paper(source="pubmed", source_id="1", title="Paper", doi="10.1/ABC"),
            Paper(source="openalex", source_id="W1", title="Paper", doi="10.1/abc"),
        ]
        result = deduplicate(papers)
        assert len(result) == 1


"""Tests for topic analysis functionality."""

import pytest

from papercli.models import Paper
from papercli.topics import (
    KeywordStats,
    Topic,
    TopicAnalysis,
    tokenize,
    extract_bigrams,
    extract_keywords_stats,
    format_topics_markdown,
    format_topics_json,
)


class TestTokenization:
    """Tests for text tokenization."""

    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        text = "Deep learning for protein folding"
        tokens = tokenize(text)
        assert "deep" in tokens
        assert "learning" in tokens
        assert "protein" in tokens
        assert "folding" in tokens

    def test_stopword_removal(self):
        """Test that stopwords are removed."""
        text = "The quick brown fox and the lazy dog"
        tokens = tokenize(text)
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_short_word_removal(self):
        """Test that short words are removed."""
        text = "A ML AI model is good"
        tokens = tokenize(text)
        assert "a" not in tokens
        assert "is" not in tokens
        # ML and AI are kept (length > 2 but may be filtered as stopwords)

    def test_case_normalization(self):
        """Test that tokens are lowercased."""
        text = "CRISPR Cas9 Gene Editing"
        tokens = tokenize(text)
        assert "crispr" in tokens
        assert "cas9" in tokens
        assert "gene" in tokens
        assert "editing" in tokens

    def test_punctuation_handling(self):
        """Test that punctuation is handled."""
        text = "RNA-seq, single-cell analysis!"
        tokens = tokenize(text)
        assert "rna" in tokens
        assert "seq" in tokens
        assert "single" in tokens
        assert "cell" in tokens


class TestBigramExtraction:
    """Tests for bigram extraction."""

    def test_basic_bigrams(self):
        """Test basic bigram extraction."""
        tokens = ["deep", "learning", "model"]
        bigrams = extract_bigrams(tokens)
        assert "deep learning" in bigrams
        assert "learning model" in bigrams
        assert len(bigrams) == 2

    def test_empty_tokens(self):
        """Test bigrams from empty list."""
        bigrams = extract_bigrams([])
        assert bigrams == []

    def test_single_token(self):
        """Test bigrams from single token."""
        bigrams = extract_bigrams(["word"])
        assert bigrams == []


class TestKeywordStatsExtraction:
    """Tests for keyword statistics extraction."""

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            Paper(
                source="pubmed",
                source_id="1",
                title="Deep Learning for Protein Structure Prediction",
                abstract="We present a deep learning model for predicting protein structures using transformers.",
            ),
            Paper(
                source="pubmed",
                source_id="2",
                title="Machine Learning in Drug Discovery",
                abstract="This paper explores machine learning approaches for drug target identification.",
            ),
            Paper(
                source="pubmed",
                source_id="3",
                title="Deep Learning Applications in Healthcare",
                abstract="We review deep learning applications in medical imaging and diagnosis.",
            ),
        ]

    def test_basic_stats(self, sample_papers):
        """Test basic keyword statistics."""
        stats = extract_keywords_stats(sample_papers)

        assert stats.total_papers == 3
        assert stats.papers_with_abstract == 3
        assert len(stats.top_keywords) > 0
        assert len(stats.top_bigrams) > 0

    def test_keyword_frequency(self, sample_papers):
        """Test that 'deep' appears frequently."""
        stats = extract_keywords_stats(sample_papers)

        # Find 'deep' in keywords
        deep_count = next((count for kw, count in stats.top_keywords if kw == "deep"), 0)
        assert deep_count >= 2  # Should appear at least twice

    def test_bigram_frequency(self, sample_papers):
        """Test that 'deep learning' is a top bigram."""
        stats = extract_keywords_stats(sample_papers)

        # Check if 'deep learning' is in top bigrams
        bigram_names = [phrase for phrase, _ in stats.top_bigrams]
        assert "deep learning" in bigram_names

    def test_empty_papers(self):
        """Test with empty paper list."""
        stats = extract_keywords_stats([])

        assert stats.total_papers == 0
        assert stats.papers_with_abstract == 0
        assert stats.top_keywords == []
        assert stats.top_bigrams == []

    def test_papers_without_abstract(self):
        """Test papers without abstracts."""
        papers = [
            Paper(
                source="pubmed",
                source_id="1",
                title="Title Only Paper",
                abstract=None,
            ),
        ]
        stats = extract_keywords_stats(papers)

        assert stats.total_papers == 1
        assert stats.papers_with_abstract == 0
        # Should still extract keywords from title
        assert len(stats.top_keywords) > 0


class TestTopicModels:
    """Tests for topic-related Pydantic models."""

    def test_topic_model(self):
        """Test Topic model creation."""
        topic = Topic(
            name="Deep Learning",
            keywords=["neural networks", "transformers", "attention"],
            summary="Research on deep learning methods.",
            paper_count=10,
            representative_papers=["Paper A", "Paper B"],
        )

        assert topic.name == "Deep Learning"
        assert len(topic.keywords) == 3
        assert topic.paper_count == 10

    def test_topic_analysis_model(self):
        """Test TopicAnalysis model creation."""
        analysis = TopicAnalysis(
            topics=[
                Topic(
                    name="Topic A",
                    keywords=["kw1", "kw2"],
                    summary="Summary A",
                    paper_count=5,
                ),
                Topic(
                    name="Topic B",
                    keywords=["kw3", "kw4"],
                    summary="Summary B",
                    paper_count=3,
                ),
            ],
            overall_summary="This is the overall summary.",
            methodology_trends=["Trend 1", "Trend 2"],
            emerging_themes=["Theme 1"],
        )

        assert len(analysis.topics) == 2
        assert analysis.overall_summary == "This is the overall summary."
        assert len(analysis.methodology_trends) == 2
        assert len(analysis.emerging_themes) == 1


class TestTopicsFormatting:
    """Tests for topic analysis output formatting."""

    @pytest.fixture
    def sample_stats(self):
        """Create sample KeywordStats."""
        return KeywordStats(
            top_keywords=[("learning", 10), ("deep", 8), ("model", 5)],
            top_bigrams=[("deep learning", 6), ("machine learning", 4)],
            total_papers=20,
            papers_with_abstract=18,
        )

    @pytest.fixture
    def sample_analysis(self):
        """Create sample TopicAnalysis."""
        return TopicAnalysis(
            topics=[
                Topic(
                    name="Deep Learning Methods",
                    keywords=["neural networks", "transformers"],
                    summary="Research on deep learning architectures.",
                    paper_count=8,
                    representative_papers=["Paper A", "Paper B"],
                ),
                Topic(
                    name="Drug Discovery",
                    keywords=["drug target", "screening"],
                    summary="Applications in pharmaceutical research.",
                    paper_count=5,
                    representative_papers=["Paper C"],
                ),
            ],
            overall_summary="The papers cover machine learning in biomedicine.",
            methodology_trends=["Transformer architectures", "Transfer learning"],
            emerging_themes=["Foundation models"],
        )

    def test_markdown_format(self, sample_stats, sample_analysis):
        """Test markdown output format."""
        output = format_topics_markdown(sample_stats, sample_analysis)

        # Check structure
        assert "# Topic Analysis Report" in output
        assert "## Overview" in output
        assert "## Summary" in output
        assert "## Identified Topics" in output
        assert "## Keyword Statistics" in output

        # Check content
        assert "Total papers analyzed**: 20" in output
        assert "Deep Learning Methods" in output
        assert "Drug Discovery" in output
        assert "Transformer architectures" in output

    def test_markdown_with_filters(self, sample_stats, sample_analysis):
        """Test markdown with filter info."""
        output = format_topics_markdown(
            sample_stats,
            sample_analysis,
            venue_filter="Bioinformatics",
            year_filter=2025,
        )

        assert "Year filter**: 2025" in output
        assert "Venue filter**: Bioinformatics" in output

    def test_json_format(self, sample_stats, sample_analysis):
        """Test JSON output format."""
        output = format_topics_json(sample_stats, sample_analysis)

        # Check structure
        assert "metadata" in output
        assert "topics" in output
        assert "overall_summary" in output
        assert "keyword_stats" in output

        # Check content
        assert output["metadata"]["total_papers"] == 20
        assert len(output["topics"]) == 2
        assert output["topics"][0]["name"] == "Deep Learning Methods"

    def test_json_with_filters(self, sample_stats, sample_analysis):
        """Test JSON with filter info."""
        output = format_topics_json(
            sample_stats,
            sample_analysis,
            venue_filter="Nature",
            year_filter=2024,
        )

        assert output["metadata"]["venue_filter"] == "Nature"
        assert output["metadata"]["year_filter"] == 2024


class TestQueryIntentWithFilters:
    """Tests for QueryIntent with new filter fields."""

    def test_query_intent_with_year(self):
        """Test QueryIntent with year filter."""
        from papercli.models import QueryIntent

        intent = QueryIntent(
            reasoning="Looking for 2025 papers.",
            query_en="bioinformatics",
            year=2025,
        )

        assert intent.year == 2025
        assert intent.year_min is None
        assert intent.year_max is None

    def test_query_intent_with_year_range(self):
        """Test QueryIntent with year range."""
        from papercli.models import QueryIntent

        intent = QueryIntent(
            reasoning="Looking for recent papers.",
            query_en="bioinformatics",
            year_min=2020,
            year_max=2025,
        )

        assert intent.year is None
        assert intent.year_min == 2020
        assert intent.year_max == 2025

    def test_query_intent_with_venue(self):
        """Test QueryIntent with venue filter."""
        from papercli.models import QueryIntent

        intent = QueryIntent(
            reasoning="Looking for Nature papers.",
            query_en="genomics",
            venue="Nature",
        )

        assert intent.venue == "Nature"

    def test_query_intent_defaults(self):
        """Test that new fields have None defaults."""
        from papercli.models import QueryIntent

        intent = QueryIntent(
            reasoning="Basic query.",
            query_en="test",
        )

        assert intent.year is None
        assert intent.year_min is None
        assert intent.year_max is None
        assert intent.venue is None

    def test_query_intent_serialization(self):
        """Test that QueryIntent serializes correctly."""
        from papercli.models import QueryIntent

        intent = QueryIntent(
            reasoning="Test",
            query_en="test query",
            year=2025,
            venue="Cell",
        )

        data = intent.model_dump()
        assert data["year"] == 2025
        assert data["venue"] == "Cell"

        # Test deserialization
        restored = QueryIntent.model_validate(data)
        assert restored.year == 2025
        assert restored.venue == "Cell"

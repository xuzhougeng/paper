"""Tests for arXiv Atom feed parsing."""

import pytest

from papercli.sources.arxiv import ArxivSource


SAMPLE_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>
      Attention Is All You Need: A Comprehensive Survey
    </title>
    <summary>
      We provide a comprehensive survey of transformer models
      and their applications in natural language processing.
    </summary>
    <published>2023-01-15T12:00:00Z</published>
    <author>
      <name>Alice Johnson</name>
    </author>
    <author>
      <name>Bob Williams</name>
    </author>
    <link href="http://arxiv.org/abs/2301.12345v1" rel="alternate" type="text/html"/>
    <link title="doi" href="http://dx.doi.org/10.1234/arxiv.2301.12345" rel="related"/>
    <arxiv:primary_category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2302.99999v2</id>
    <title>Minimal Paper</title>
    <summary>Short abstract.</summary>
    <published>2023-02-20T08:00:00Z</published>
    <author>
      <name>Single Author</name>
    </author>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
</feed>"""


class TestArxivParsing:
    """Tests for arXiv Atom feed parsing."""

    def test_parse_complete_entry(self):
        """Test parsing a complete arXiv entry."""
        source = ArxivSource()
        papers = source._parse_atom(SAMPLE_ARXIV_ATOM)

        assert len(papers) == 2

        paper = papers[0]
        assert paper.source == "arxiv"
        assert paper.source_id == "2301.12345"
        assert paper.title == "Attention Is All You Need: A Comprehensive Survey"
        assert "comprehensive survey" in paper.abstract
        assert "natural language processing" in paper.abstract
        assert paper.year == 2023
        assert paper.authors == ["Alice Johnson", "Bob Williams"]
        assert paper.url == "https://arxiv.org/abs/2301.12345"
        assert paper.venue == "arXiv:cs.CL"

    def test_parse_minimal_entry(self):
        """Test parsing entry with minimal data."""
        source = ArxivSource()
        papers = source._parse_atom(SAMPLE_ARXIV_ATOM)

        paper = papers[1]
        assert paper.source_id == "2302.99999"
        assert paper.title == "Minimal Paper"
        assert paper.abstract == "Short abstract."
        assert paper.authors == ["Single Author"]
        assert paper.venue == "arXiv:cs.LG"

    def test_parse_invalid_xml(self):
        """Test handling of invalid XML."""
        source = ArxivSource()
        papers = source._parse_atom("not valid xml")
        assert papers == []

    def test_parse_empty_feed(self):
        """Test handling of empty feed."""
        source = ArxivSource()
        empty_feed = '''<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>'''
        papers = source._parse_atom(empty_feed)
        assert papers == []

    def test_title_whitespace_normalization(self):
        """Test that title whitespace is normalized."""
        source = ArxivSource()
        papers = source._parse_atom(SAMPLE_ARXIV_ATOM)

        # Title should have normalized whitespace (no newlines or extra spaces)
        paper = papers[0]
        assert "\n" not in paper.title
        assert "  " not in paper.title


class TestArxivQueryBuilding:
    """Tests for arXiv query construction."""

    def test_short_query(self):
        """Test query building with few terms."""
        from papercli.models import QueryIntent

        source = ArxivSource()
        intent = QueryIntent(
            reasoning="User wants papers about neural networks.",
            query_en="neural networks",
            keywords=["neural", "networks"],
        )

        query = source._build_query(intent)
        assert "all:" in query

    def test_long_query(self):
        """Test query building with many terms."""
        from papercli.models import QueryIntent

        source = ArxivSource()
        intent = QueryIntent(
            reasoning="User wants papers about deep learning for NLP applications.",
            query_en="deep learning for natural language processing applications",
            keywords=["deep", "learning", "NLP"],
        )

        query = source._build_query(intent)
        # Long queries should search ti: and abs:
        assert "ti:" in query or "abs:" in query


"""Tests for PubMed XML parsing."""

import pytest

from papercli.sources.pubmed import PubMedSource


SAMPLE_PUBMED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">12345678</PMID>
      <Article PubModel="Print">
        <Journal>
          <Title>Nature</Title>
          <PubDate>
            <Year>2023</Year>
            <Month>Jun</Month>
          </PubDate>
        </Journal>
        <ArticleTitle>Deep Learning for Protein Structure Prediction</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">We present a novel deep learning approach.</AbstractText>
          <AbstractText Label="METHODS">Our method uses transformer architecture.</AbstractText>
          <AbstractText Label="RESULTS">We achieved state-of-the-art accuracy.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1038/s41586-023-12345</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


MINIMAL_PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999</PMID>
      <Article>
        <ArticleTitle>Minimal Test Paper</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class TestPubMedParsing:
    """Tests for PubMed XML parsing."""

    def test_parse_complete_article(self):
        """Test parsing a complete PubMed article."""
        source = PubMedSource()
        papers = source._parse_pubmed_xml(SAMPLE_PUBMED_XML)

        assert len(papers) == 1
        paper = papers[0]

        assert paper.source == "pubmed"
        assert paper.source_id == "12345678"
        assert paper.title == "Deep Learning for Protein Structure Prediction"
        assert "BACKGROUND: We present a novel deep learning approach" in paper.abstract
        assert "METHODS: Our method uses transformer architecture" in paper.abstract
        assert paper.year == 2023
        assert paper.authors == ["John Smith", "Jane Doe"]
        assert paper.doi == "10.1038/s41586-023-12345"
        assert paper.venue == "Nature"
        assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"

    def test_parse_minimal_article(self):
        """Test parsing article with minimal data."""
        source = PubMedSource()
        papers = source._parse_pubmed_xml(MINIMAL_PUBMED_XML)

        assert len(papers) == 1
        paper = papers[0]

        assert paper.source_id == "99999"
        assert paper.title == "Minimal Test Paper"
        assert paper.abstract is None
        assert paper.year is None
        assert paper.authors == []

    def test_parse_invalid_xml(self):
        """Test handling of invalid XML."""
        source = PubMedSource()
        papers = source._parse_pubmed_xml("not valid xml")
        assert papers == []

    def test_parse_empty_xml(self):
        """Test handling of empty article set."""
        source = PubMedSource()
        papers = source._parse_pubmed_xml("<PubmedArticleSet></PubmedArticleSet>")
        assert papers == []


class TestPubMedQueryBuilding:
    """Tests for PubMed query construction."""

    def test_simple_query(self):
        """Test building simple query."""
        from papercli.models import QueryIntent

        source = PubMedSource()
        intent = QueryIntent(
            query_en="CRISPR gene editing",
            keywords=["CRISPR", "gene", "editing"],
        )

        query = source._build_query(intent)
        assert "CRISPR gene editing" in query
        assert "[Title/Abstract]" in query

    def test_query_with_required_phrases(self):
        """Test query with required phrases."""
        from papercli.models import QueryIntent

        source = PubMedSource()
        intent = QueryIntent(
            query_en="machine learning",
            keywords=["machine", "learning"],
            required_phrases=["deep learning"],
        )

        query = source._build_query(intent)
        assert '"deep learning"' in query

    def test_query_with_exclusions(self):
        """Test query with excluded terms."""
        from papercli.models import QueryIntent

        source = PubMedSource()
        intent = QueryIntent(
            query_en="cancer therapy",
            keywords=["cancer", "therapy"],
            exclude_terms=["review"],
        )

        query = source._build_query(intent)
        assert "NOT review" in query


"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    from papercli.models import Paper

    return [
        Paper(
            source="pubmed",
            source_id="12345",
            title="Deep Learning for Drug Discovery",
            abstract="We present a novel deep learning approach for predicting drug-target interactions.",
            year=2023,
            authors=["John Smith", "Jane Doe"],
            doi="10.1234/test1",
            url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            venue="Nature Medicine",
        ),
        Paper(
            source="openalex",
            source_id="W54321",
            title="Machine Learning in Healthcare",
            abstract="This review covers applications of machine learning in clinical settings.",
            year=2022,
            authors=["Alice Johnson"],
            doi="10.1234/test2",
            url="https://doi.org/10.1234/test2",
            venue="Lancet Digital Health",
        ),
        Paper(
            source="arxiv",
            source_id="2301.12345",
            title="Transformer Models for Protein Structure",
            abstract="We introduce a transformer-based model for protein structure prediction.",
            year=2023,
            authors=["Bob Williams", "Carol Brown"],
            url="https://arxiv.org/abs/2301.12345",
            venue="arXiv:cs.LG",
        ),
    ]


@pytest.fixture
def sample_query_intent():
    """Create a sample QueryIntent for testing."""
    from papercli.models import QueryIntent

    return QueryIntent(
        reasoning="User is looking for papers about applying deep learning to drug discovery.",
        query_en="deep learning drug discovery",
        keywords=["deep learning", "drug", "discovery"],
        synonyms={"drug": ["pharmaceutical", "compound"]},
        required_phrases=[],
        exclude_terms=[],
    )


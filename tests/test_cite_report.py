"""Tests for citation report formatting and sentence splitting."""

from papercli.cli import split_sentences
from papercli.models import EvalResult, Paper
from papercli.output import format_citation_report


def test_split_sentences_basic():
    text = "第一句。第二句！\n第三句？Fourth sentence! Fifth sentence? Sixth sentence."
    sentences = split_sentences(text)
    assert sentences == [
        "第一句。",
        "第二句！",
        "第三句？",
        "Fourth sentence!",
        "Fifth sentence?",
        "Sixth sentence.",
    ]


def test_format_citation_report_includes_recommended():
    paper = Paper(
        source="pubmed",
        source_id="pmid:123",
        title="Example Paper Title",
        abstract="Example abstract sentence for evidence.",
        year=2024,
        authors=["A. Author", "B. Author"],
        doi="10.1000/example",
        venue="Journal of Examples",
        url="https://example.org/paper",
    )
    result = EvalResult(
        paper=paper,
        score=9.2,
        meets_need=True,
        evidence_quote="Example abstract sentence for evidence.",
        evidence_field="abstract",
        short_reason="Directly supports the sentence.",
    )

    report = format_citation_report(
        [
            {
                "sentence": "This is a test sentence.",
                "results": [result],
                "recommended": result,
                "error": None,
            }
        ],
        sources=["pubmed"],
        top_k=1,
    )

    assert "# Citation Report" in report
    assert "## Sentence 1" in report
    assert "This is a test sentence." in report
    assert "Example Paper Title" in report
    assert "https://doi.org/10.1000/example" in report
    assert "Example abstract sentence for evidence." in report
    assert "Candidates (top-1)" in report

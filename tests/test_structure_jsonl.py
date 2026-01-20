"""Tests for second-pass structuring of page-level JSONL."""

from papercli.structure import (
    PageRecord,
    structure_from_pages,
    structured_paper_to_markdown,
    extract_doi,
    extract_date,
    extract_keywords,
    extract_journal,
    extract_authors,
)


def test_structure_extracts_title_and_major_sections_and_media():
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text=(
                "# Test Paper Title\n\n"
                "## SUMMARY\n"
                "This is the abstract.\n\n"
                "## INTRODUCTION\n"
                "Intro text.\n\n"
                "## RESULTS\n"
                "Results lead-in.\n"
            ),
        ),
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=1,
            page_no=2,
            text=(
                "<!-- figureText: Alt text for Figure 1 -->\n"
                '<img src="https://example.com/fig1.jpg"/>\n'
                "Figure 1. Main figure legend\n"
                "(A) Panel A.\n"
                "(B) Panel B.\n"
            ),
        ),
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=2,
            page_no=3,
            text=(
                "## STAR★METHODS\n\n"
                "KEY RESOURCES TABLE\n"
                "<!-- Media -->\n"
                "<table><tr><td>A</td></tr></table>\n"
                "Some methods text.\n"
            ),
        ),
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=3,
            page_no=4,
            text=(
                '<img src="https://example.com/supp-fig.jpg"/>\n'
                "<!-- Meanless: (legend on next page) -->\n"
            ),
        ),
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=4,
            page_no=5,
            text=(
                "Figure S2. Supplementary legend, related to Figure 1\n"
                "More legend text.\n"
            ),
        ),
    ]

    structured = structure_from_pages(pages)

    assert structured.title == "Test Paper Title"
    assert structured.abstract and "This is the abstract" in structured.abstract
    assert structured.results and "Results lead-in" in structured.results
    assert structured.methods and "KEY RESOURCES TABLE" in structured.methods

    # The HTML table should be extracted into tables and not pollute methods text.
    assert structured.main_tables
    assert structured.main_tables[0].html and "<table>" in structured.main_tables[0].html

    # Main figure should have image URL + alt text.
    assert structured.main_figures
    fig1 = next(f for f in structured.main_figures if f.figure_id == "Figure 1")
    assert fig1.image_urls == ["https://example.com/fig1.jpg"]
    assert fig1.alt_text and "Alt text" in fig1.alt_text

    # Supplementary figure should have image URL attached cross-page.
    assert structured.supp_figures
    s2 = next(f for f in structured.supp_figures if f.figure_id == "Figure S2")
    assert "Supplementary legend" in s2.caption
    assert "https://example.com/supp-fig.jpg" in s2.image_urls

    # Appendix should include supplementary legends by default.
    assert structured.appendix and "Figure S2" in structured.appendix

    md = structured_paper_to_markdown(structured)

    # YAML front matter should be present at the beginning
    assert md.startswith("---\n")
    assert "title: Test Paper Title" in md
    assert "doc2x_uid:" in md
    assert "source_path:" in md
    assert "page_count: 5" in md
    # Abstract should appear in YAML as block scalar
    assert "abstract: |" in md

    # Body content assertions (after YAML front matter)
    assert "# Test Paper Title" in md
    assert "## Abstract" in md
    assert "This is the abstract" in md
    assert "## Main Figures" in md
    assert "Figure 1" in md
    assert "<https://example.com/fig1.jpg>" in md
    assert "## Supplementary Figures" in md
    assert "Figure S2" in md
    assert "<https://example.com/supp-fig.jpg>" in md


def test_yaml_front_matter_with_metadata():
    """Test that YAML front matter includes extracted metadata."""
    pages = [
        PageRecord(
            doc2x_uid="test-uid-123",
            source_path="/path/to/paper.pdf",
            page_index=0,
            page_no=1,
            text=(
                "# A Novel Method for Single-Cell Analysis\n\n"
                "John Smith, Jane Doe, Bob Johnson\n\n"
                "DOI: 10.1234/example.2024.001\n"
                "Published: 2024-03-15\n"
                "Journal: Nature Methods\n"
                "Keywords: single-cell, RNA-seq, bioinformatics\n\n"
                "## ABSTRACT\n"
                "This paper presents a novel method for analyzing single-cell data.\n\n"
                "## INTRODUCTION\n"
                "Introduction text here.\n"
            ),
        ),
    ]

    structured = structure_from_pages(pages)

    # Check extracted metadata
    assert structured.doi == "10.1234/example.2024.001"
    assert structured.date == "2024-03-15"
    assert structured.journal == "Nature Methods"
    assert "single-cell" in structured.keywords
    assert "RNA-seq" in structured.keywords
    assert "bioinformatics" in structured.keywords

    # Check authors extraction (best-effort)
    # Note: Author extraction is heuristic, so we check if at least some names are found
    assert len(structured.authors) >= 1

    md = structured_paper_to_markdown(structured)

    # YAML front matter checks
    assert md.startswith("---\n")
    assert "title:" in md
    assert "doi: " in md and "10.1234/example.2024.001" in md
    assert "date: " in md and "2024-03-15" in md
    assert "journal: " in md and "Nature Methods" in md
    assert "keywords:" in md
    assert "  - single-cell" in md or "  - \"single-cell\"" in md
    assert "abstract: |" in md

    # Verify YAML front matter ends with ---
    yaml_end_idx = md.find("---\n", 4)  # Find the second ---
    assert yaml_end_idx > 0


def test_extract_doi():
    """Test DOI extraction from pages."""
    # Labeled DOI
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="Some text\nDOI: 10.1038/nature12373\nMore text",
        ),
    ]
    assert extract_doi(pages) == "10.1038/nature12373"

    # DOI without label
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="Some text https://doi.org/10.1101/2024.01.01.123456 more text",
        ),
    ]
    assert extract_doi(pages) == "10.1101/2024.01.01.123456"

    # No DOI
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="No DOI here",
        ),
    ]
    assert extract_doi(pages) is None


def test_extract_date():
    """Test date extraction from pages."""
    # Labeled date
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="Published: 2024-03-15",
        ),
    ]
    assert extract_date(pages) == "2024-03-15"

    # Month DD, YYYY format
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="Accepted: January 5, 2024",
        ),
    ]
    assert extract_date(pages) == "2024-01-05"

    # No date
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="No date here",
        ),
    ]
    assert extract_date(pages) is None


def test_extract_keywords():
    """Test keywords extraction from pages."""
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="Keywords: machine learning, deep learning, neural networks",
        ),
    ]
    keywords = extract_keywords(pages)
    assert "machine learning" in keywords
    assert "deep learning" in keywords
    assert "neural networks" in keywords

    # Semicolon separated
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="KEY WORDS: RNA-seq; single-cell; bioinformatics",
        ),
    ]
    keywords = extract_keywords(pages)
    assert len(keywords) == 3

    # No keywords
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="No keywords here",
        ),
    ]
    assert extract_keywords(pages) == []


def test_extract_journal():
    """Test journal extraction from pages."""
    # Explicit label
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="Journal: Nature Methods",
        ),
    ]
    assert extract_journal(pages) == "Nature Methods"

    # Preprint server
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="This preprint was posted on bioRxiv",
        ),
    ]
    assert extract_journal(pages) == "bioRxiv"

    # No journal
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text="No journal info",
        ),
    ]
    assert extract_journal(pages) is None


def test_extract_authors():
    """Test author extraction from pages."""
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text=(
                "# Test Paper Title\n\n"
                "John Smith, Jane Doe, Bob Johnson\n\n"
                "Department of Computer Science\n"
                "## ABSTRACT\n"
                "Abstract text.\n"
            ),
        ),
    ]
    authors = extract_authors(pages, "Test Paper Title")
    # Should find at least some author names
    assert len(authors) >= 1
    # Should not include affiliation
    assert not any("Department" in a for a in authors)


def test_json_output_excludes_metadata_fields():
    """Test that JSON output excludes metadata fields by default."""
    pages = [
        PageRecord(
            doc2x_uid="uid",
            source_path="/paper.pdf",
            page_index=0,
            page_no=1,
            text=(
                "# Test Title\n\n"
                "DOI: 10.1234/test\n"
                "Keywords: test, example\n"
                "## ABSTRACT\n"
                "Abstract text.\n"
            ),
        ),
    ]

    structured = structure_from_pages(pages)

    # Metadata fields should be populated
    assert structured.doi == "10.1234/test"
    assert structured.keywords == ["test", "example"]

    # But model_dump() should exclude them by default
    data = structured.model_dump()
    assert "doi" not in data
    assert "keywords" not in data
    assert "authors" not in data
    assert "journal" not in data
    assert "date" not in data

    # Core fields should still be present
    assert "title" in data
    assert "abstract" in data
    assert "doc2x_uid" in data



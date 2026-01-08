"""Tests for second-pass structuring of page-level JSONL."""

from papercli.structure import PageRecord, structure_from_pages, structured_paper_to_markdown


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
    assert "# Test Paper Title" in md
    assert "## Abstract" in md
    assert "This is the abstract" in md
    assert "## Main Figures" in md
    assert "Figure 1" in md
    assert "<https://example.com/fig1.jpg>" in md
    assert "## Supplementary Figures" in md
    assert "Figure S2" in md
    assert "<https://example.com/supp-fig.jpg>" in md



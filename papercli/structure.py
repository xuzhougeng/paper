"""Second-pass structuring for page-level JSONL produced by `paper extract`.

This module turns Doc2X/PaperCLI's page-level JSONL into database-friendly fields:
- title / abstract / methods / results / references / appendix
- main vs supplementary figures/tables (with basic cross-page legend association)

The output is intentionally "best-effort" and should be treated as a starting point
for downstream normalization and database ingestion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Models
# =============================================================================


@dataclass(frozen=True)
class PageRecord:
    doc2x_uid: str
    source_path: str
    page_index: int
    page_no: int
    text: str
    raw_page: dict[str, Any] | None = None


class ExtractedFigure(BaseModel):
    figure_id: str = Field(..., description="Figure identifier, e.g. 'Figure 1' or 'Figure S2'")
    caption: str = Field(default="", description="Full legend/caption text")
    page_index: int | None = Field(default=None, description="0-based page index where legend starts")
    page_no: int | None = Field(default=None, description="1-based page number where legend starts")
    image_urls: list[str] = Field(default_factory=list, description="Associated image URLs (if available)")
    alt_text: str | None = Field(default=None, description="FigureText/alt-text extracted from Doc2X comments")
    is_supplementary: bool = Field(default=False, description="Whether this is a supplementary figure (S*)")


class ExtractedTable(BaseModel):
    table_id: str = Field(..., description="Table identifier, e.g. 'Table 1', 'Table S1', or synthetic id")
    caption: str = Field(default="", description="Caption/title text near the table")
    page_index: int | None = Field(default=None, description="0-based page index where table appears")
    page_no: int | None = Field(default=None, description="1-based page number where table appears")
    image_urls: list[str] = Field(default_factory=list, description="Associated image URLs (if table is an image)")
    html: str | None = Field(default=None, description="Raw HTML table (if present)")
    text: str | None = Field(default=None, description="Fallback text content (if present)")
    is_supplementary: bool = Field(default=False, description="Whether this is a supplementary table (S*)")


class StructuredPaper(BaseModel):
    doc2x_uid: str
    source_path: str
    page_count: int

    title: str | None = None
    abstract: str | None = None
    methods: str | None = None
    results: str | None = None
    references: str | None = None
    appendix: str | None = None

    main_figures: list[ExtractedFigure] = Field(default_factory=list)
    main_tables: list[ExtractedTable] = Field(default_factory=list)
    supp_figures: list[ExtractedFigure] = Field(default_factory=list)
    supp_tables: list[ExtractedTable] = Field(default_factory=list)

    # Optional: other major sections we found (introduction, discussion, etc.)
    other_sections: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # Metadata fields for YAML front matter (excluded from JSON output by default)
    authors: list[str] = Field(default_factory=list, exclude=True)
    keywords: list[str] = Field(default_factory=list, exclude=True)
    journal: str | None = Field(default=None, exclude=True)
    date: str | None = Field(default=None, exclude=True)
    doi: str | None = Field(default=None, exclude=True)


# =============================================================================
# Regex helpers
# =============================================================================


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_COMMENT_LINE_RE = re.compile(r"^\s*<!--[\s\S]*?-->\s*$")
_IMG_RE = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*/?>', re.IGNORECASE)
_FIGURETEXT_RE = re.compile(r"<!--\s*figureText:\s*([\s\S]*?)\s*-->", re.IGNORECASE)
_TABLE_BLOCK_RE = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
_LEGEND_NEXT_RE = re.compile(r"\blegend\s+on\s+next\s+page\b", re.IGNORECASE)

_FIGURE_LEGEND_RE = re.compile(
    r"^(Figure|Fig\.?)\s+(S?\d+[A-Za-z]?)\s*[\.:]?\s*(.*)$", re.IGNORECASE
)
_TABLE_LEGEND_RE = re.compile(r"^Table\s+(S?\d+[A-Za-z]?)\s*[\.:]?\s*(.*)$", re.IGNORECASE)

# DOI pattern: 10.XXXX/... (standard DOI format)
_DOI_RE = re.compile(r"\b(10\.\d{4,}/[^\s\]>)\"']+)", re.IGNORECASE)
# DOI with explicit label: "DOI:" or "doi:" followed by DOI
_DOI_LABEL_RE = re.compile(r"(?:doi|DOI)\s*[:=]?\s*(10\.\d{4,}/[^\s\]>)\"']+)", re.IGNORECASE)

# Date patterns
_DATE_ISO_RE = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b")
_DATE_MDY_RE = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_DATE_DMY_RE = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r",?\s+\d{4})\b",
    re.IGNORECASE,
)
# Date with label (Published/Accepted/Received)
_DATE_LABEL_RE = re.compile(
    r"(?:Published|Accepted|Received|Date|Online)\s*[:;]?\s*"
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r",?\s+\d{4})",
    re.IGNORECASE,
)

# Keywords pattern
_KEYWORDS_RE = re.compile(
    r"(?:Keywords?|KEY\s*WORDS?|Index\s+Terms?)\s*[:;]?\s*(.+)",
    re.IGNORECASE,
)

# Journal/source patterns
_JOURNAL_RE = re.compile(
    r"(?:Journal|Published\s+in|Appeared\s+in)\s*[:;]?\s*(.+)",
    re.IGNORECASE,
)
_PREPRINT_RE = re.compile(
    r"\b(bioRxiv|medRxiv|arXiv|ChemRxiv|SSRN|PsyArXiv|OSF\s+Preprints?)\b",
    re.IGNORECASE,
)

# Affiliation/institution markers (for filtering author lines)
_AFFILIATION_MARKERS = re.compile(
    r"(?:University|Institute|Department|College|School|Laboratory|Lab|Hospital|Center|Centre|"
    r"Faculty|Division|Unit|Research|@|\.edu|\.org|\.ac\.|\.gov)",
    re.IGNORECASE,
)

# Major section headings that mark end of author block
_SECTION_HEADING_MARKERS = {
    "ABSTRACT", "SUMMARY", "INTRODUCTION", "BACKGROUND", "KEYWORDS", "KEY WORDS",
    "METHODS", "MATERIALS", "RESULTS", "DISCUSSION", "CONCLUSION", "CONCLUSIONS",
    "REFERENCES", "ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS", "FUNDING", "CONFLICTS",
}


# =============================================================================
# Metadata extraction (best-effort heuristics)
# =============================================================================


def extract_doi(pages: list[PageRecord], *, max_pages: int = 5) -> str | None:
    """
    Best-effort DOI extraction from early pages.

    Prioritizes DOIs with explicit labels (e.g., "DOI: 10.xxx/...").
    """
    texts = [p.text for p in pages[:max_pages]]

    # First try: labeled DOI (more reliable)
    for text in texts:
        for line in text.splitlines():
            m = _DOI_LABEL_RE.search(line)
            if m:
                doi = m.group(1).rstrip(".,;)")
                return doi

    # Second try: any DOI pattern
    for text in texts:
        m = _DOI_RE.search(text)
        if m:
            doi = m.group(1).rstrip(".,;)")
            return doi

    return None


def extract_date(pages: list[PageRecord], *, max_pages: int = 5) -> str | None:
    """
    Best-effort date extraction from early pages.

    Prioritizes dates with labels (Published/Accepted/Received).
    Attempts to normalize to YYYY-MM-DD format.
    """
    texts = [p.text for p in pages[:max_pages]]

    # First try: labeled date
    for text in texts:
        for line in text.splitlines():
            m = _DATE_LABEL_RE.search(line)
            if m:
                return _normalize_date(m.group(1))

    # Second try: ISO date
    for text in texts:
        m = _DATE_ISO_RE.search(text)
        if m:
            return _normalize_date(m.group(1))

    # Third try: Month DD, YYYY
    for text in texts:
        m = _DATE_MDY_RE.search(text)
        if m:
            return _normalize_date(m.group(1))

    # Fourth try: DD Month YYYY
    for text in texts:
        m = _DATE_DMY_RE.search(text)
        if m:
            return _normalize_date(m.group(1))

    return None


def _normalize_date(date_str: str) -> str:
    """Attempt to normalize date to YYYY-MM-DD format."""
    import calendar

    date_str = date_str.strip()

    # Already ISO-like: YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", date_str)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    # Month DD, YYYY or Month DD YYYY
    months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str)
    if m:
        month_name, day, year = m.groups()
        month_num = months.get(month_name.lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"

    # DD Month YYYY
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", date_str)
    if m:
        day, month_name, year = m.groups()
        month_num = months.get(month_name.lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"

    # Can't normalize, return as-is
    return date_str


def extract_keywords(pages: list[PageRecord], *, max_pages: int = 5) -> list[str]:
    """
    Best-effort keywords extraction from early pages.

    Looks for "Keywords:" or similar labels and splits by comma/semicolon.
    """
    texts = [p.text for p in pages[:max_pages]]

    for text in texts:
        for line in text.splitlines():
            m = _KEYWORDS_RE.match(line.strip())
            if m:
                raw = m.group(1).strip()
                # Split by comma or semicolon, clean up
                keywords = re.split(r"[;,]", raw)
                keywords = [k.strip().strip(".").strip() for k in keywords if k.strip()]
                # Filter out empty or very long "keywords" (likely parsing errors)
                keywords = [k for k in keywords if k and len(k) < 100]
                if keywords:
                    return keywords

    return []


def extract_journal(pages: list[PageRecord], *, max_pages: int = 5) -> str | None:
    """
    Best-effort journal/source extraction from early pages.

    Looks for explicit "Journal:" labels or preprint server mentions.
    """
    texts = [p.text for p in pages[:max_pages]]

    # First try: explicit label
    for text in texts:
        for line in text.splitlines():
            m = _JOURNAL_RE.match(line.strip())
            if m:
                journal = m.group(1).strip().rstrip(".,;")
                if journal and len(journal) < 200:
                    return journal

    # Second try: preprint server mention
    for text in texts:
        m = _PREPRINT_RE.search(text)
        if m:
            return m.group(1)

    return None


def extract_authors(
    pages: list[PageRecord],
    title: str | None,
    *,
    max_pages: int = 2,
) -> list[str]:
    """
    Best-effort author extraction from early pages.

    Strategy: Find lines between the title and the first major section heading,
    filter out affiliations/emails, split by common separators.
    """
    if not pages:
        return []

    # Combine text from early pages
    combined = "\n".join(p.text for p in pages[:max_pages])
    lines = combined.splitlines()

    # Find title line index (if title is provided)
    title_idx = -1
    if title:
        title_lower = title.lower().strip()
        for i, line in enumerate(lines):
            # Check if line contains the title (allowing for markdown heading)
            line_clean = re.sub(r"^#+\s*", "", line).strip().lower()
            if title_lower in line_clean or line_clean in title_lower:
                title_idx = i
                break

    # Start searching from after the title (or from beginning if not found)
    start_idx = title_idx + 1 if title_idx >= 0 else 0

    # Collect potential author lines
    author_candidates: list[str] = []
    for i in range(start_idx, min(len(lines), start_idx + 50)):
        line = lines[i].strip()
        if not line:
            continue

        # Stop at section heading
        line_upper = re.sub(r"^#+\s*", "", line).upper().strip()
        if line_upper in _SECTION_HEADING_MARKERS:
            break

        # Stop at obvious non-author content
        if _HEADING_RE.match(line) and not _is_likely_author_line(line):
            # Check if it's a major heading
            heading_text = re.sub(r"^#+\s*", "", line).upper().strip()
            if any(marker in heading_text for marker in _SECTION_HEADING_MARKERS):
                break

        # Skip lines that look like affiliations/institutions
        if _AFFILIATION_MARKERS.search(line):
            continue

        # Skip lines that look like email addresses or URLs
        if "@" in line or "http" in line.lower():
            continue

        # Skip lines that are too long (likely paragraphs, not author names)
        if len(line) > 300:
            continue

        # Skip lines that start with numbers (likely affiliations or dates)
        if re.match(r"^\d+\s", line):
            continue

        # Skip markdown images/links
        if line.startswith("!") or line.startswith("<"):
            continue

        # This might be an author line
        if _is_likely_author_line(line):
            author_candidates.append(line)

    # Parse author names from candidates
    authors: list[str] = []
    for candidate in author_candidates:
        # Remove superscript markers (1,2,*,† etc.)
        cleaned = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰\*†‡§\d]+,?", "", candidate)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Split by common separators
        parts = re.split(r"\s*[,;]\s*|\s+and\s+|\s*&\s*", cleaned)
        for part in parts:
            name = part.strip().strip(",").strip()
            # Basic name validation: at least 2 chars, not all digits
            if name and len(name) >= 2 and not name.isdigit():
                # Skip if it looks like an institution
                if not _AFFILIATION_MARKERS.search(name):
                    authors.append(name)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_authors: list[str] = []
    for a in authors:
        a_lower = a.lower()
        if a_lower not in seen:
            seen.add(a_lower)
            unique_authors.append(a)

    return unique_authors


def _is_likely_author_line(line: str) -> bool:
    """Check if a line is likely to contain author names."""
    # Remove heading markers
    text = re.sub(r"^#+\s*", "", line).strip()

    if not text:
        return False

    # Too short or too long
    if len(text) < 3 or len(text) > 500:
        return False

    # Contains typical author-like patterns:
    # - Multiple capitalized words
    # - Comma-separated names
    # - Contains "and" between names

    # Check for capitalized words (names typically start with capitals)
    words = text.split()
    cap_words = sum(1 for w in words if w and w[0].isupper())
    if cap_words >= 2:
        return True

    # Contains comma suggesting multiple names
    if "," in text and cap_words >= 1:
        return True

    return False


# =============================================================================
# Public API
# =============================================================================


def structure_from_jsonl_path(jsonl_path: str) -> StructuredPaper:
    pages = read_page_jsonl(jsonl_path)
    return structure_from_pages(pages)


def read_page_jsonl(jsonl_path: str) -> list[PageRecord]:
    """Read page-level JSONL (one JSON object per line)."""
    pages: list[PageRecord] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no}: expected JSON object, got {type(obj).__name__}")

            missing = [k for k in ("doc2x_uid", "source_path", "page_index", "page_no", "text") if k not in obj]
            if missing:
                raise ValueError(f"Line {line_no}: missing required keys: {', '.join(missing)}")

            if not isinstance(obj["doc2x_uid"], str):
                raise ValueError(f"Line {line_no}: doc2x_uid must be string")
            if not isinstance(obj["source_path"], str):
                raise ValueError(f"Line {line_no}: source_path must be string")
            if not isinstance(obj["page_index"], int):
                raise ValueError(f"Line {line_no}: page_index must be int")
            if not isinstance(obj["page_no"], int):
                raise ValueError(f"Line {line_no}: page_no must be int")
            if not isinstance(obj["text"], str):
                raise ValueError(f"Line {line_no}: text must be string")

            raw_page = obj.get("raw_page")
            if raw_page is not None and not isinstance(raw_page, dict):
                raw_page = None

            pages.append(
                PageRecord(
                    doc2x_uid=obj["doc2x_uid"],
                    source_path=obj["source_path"],
                    page_index=obj["page_index"],
                    page_no=obj["page_no"],
                    text=obj["text"],
                    raw_page=raw_page,
                )
            )

    pages.sort(key=lambda p: p.page_index)
    return pages


def structure_from_pages(pages: list[PageRecord]) -> StructuredPaper:
    if not pages:
        raise ValueError("No pages found in JSONL")

    uid = pages[0].doc2x_uid
    source_path = pages[0].source_path
    warnings: list[str] = []

    if any(p.doc2x_uid != uid for p in pages):
        warnings.append("Multiple doc2x_uid detected across pages; using first page's uid.")
    if any(p.source_path != source_path for p in pages):
        warnings.append("Multiple source_path detected across pages; using first page's source_path.")

    title = extract_title(pages)
    section_map = extract_major_sections(pages)

    figures, tables, appendix_text, media_warnings = extract_media(pages)
    warnings.extend(media_warnings)

    main_figures = [f for f in figures if not f.is_supplementary]
    supp_figures = [f for f in figures if f.is_supplementary]
    main_tables = [t for t in tables if not t.is_supplementary]
    supp_tables = [t for t in tables if t.is_supplementary]

    # Populate primary fields requested for DB ingestion
    abstract = section_map.pop("abstract", None)
    methods = section_map.pop("methods", None)
    results = section_map.pop("results", None)
    references = section_map.pop("references", None)
    appendix = section_map.pop("appendix", None) or appendix_text

    # Keep other sections (introduction/discussion/etc.) for completeness
    other_sections = {k: v for k, v in section_map.items() if v and k not in ("title",)}

    # Extract metadata for YAML front matter (best-effort heuristics)
    doi = extract_doi(pages)
    date = extract_date(pages)
    keywords = extract_keywords(pages)
    journal = extract_journal(pages)
    authors = extract_authors(pages, title)

    return StructuredPaper(
        doc2x_uid=uid,
        source_path=source_path,
        page_count=len(pages),
        title=title,
        abstract=_clean_text_block(abstract),
        methods=_clean_text_block(methods),
        results=_clean_text_block(results),
        references=_clean_text_block(references),
        appendix=_clean_text_block(appendix),
        main_figures=main_figures,
        main_tables=main_tables,
        supp_figures=supp_figures,
        supp_tables=supp_tables,
        other_sections=other_sections,
        warnings=warnings,
        # Metadata fields
        authors=authors,
        keywords=keywords,
        journal=journal,
        date=date,
        doi=doi,
    )


def structured_paper_to_markdown(paper: StructuredPaper) -> str:
    """
    Render a StructuredPaper into a human-readable Markdown document.

    This is a convenience formatter for notes/review. The JSON output remains the
    preferred format for database ingestion.

    The output includes a YAML front matter block with metadata fields
    (title, author, abstract, keywords, journal, date, doi, etc.) for
    compatibility with static site generators and note-taking tools like Obsidian.
    """

    def section(title: str, body: str | None) -> str:
        if not body or not body.strip():
            return ""
        return f"## {title}\n\n{body.strip()}\n\n"

    lines: list[str] = []

    # ==========================================================================
    # YAML front matter
    # ==========================================================================
    lines.append("---\n")

    title = paper.title.strip() if paper.title else "Untitled"
    lines.append(f"title: {_yaml_quote_string(title)}\n")

    # Author list (YAML list format)
    if paper.authors:
        lines.append("author:\n")
        for author in paper.authors:
            lines.append(f"  - {_yaml_quote_string(author)}\n")

    # Abstract as block scalar
    if paper.abstract:
        abstract_clean = paper.abstract.strip().replace("\r\n", "\n").replace("\r", "\n")
        lines.append("abstract: |\n")
        for line in abstract_clean.splitlines():
            lines.append(f"  {line}\n")

    # Keywords list (YAML list format)
    if paper.keywords:
        lines.append("keywords:\n")
        for kw in paper.keywords:
            lines.append(f"  - {_yaml_quote_string(kw)}\n")

    # Journal
    if paper.journal:
        lines.append(f"journal: {_yaml_quote_string(paper.journal)}\n")

    # Date
    if paper.date:
        lines.append(f"date: {_yaml_quote_string(paper.date)}\n")

    # DOI
    if paper.doi:
        lines.append(f"doi: {_yaml_quote_string(paper.doi)}\n")

    # Additional metadata for traceability
    lines.append(f"doc2x_uid: {_yaml_quote_string(paper.doc2x_uid)}\n")
    lines.append(f"source_path: {_yaml_quote_string(paper.source_path)}\n")
    lines.append(f"page_count: {paper.page_count}\n")

    lines.append("---\n")
    lines.append("\n")

    # ==========================================================================
    # Markdown body
    # ==========================================================================
    lines.append(f"# {title}\n")
    lines.append("\n")

    if paper.warnings:
        lines.append("## Warnings\n\n")
        for w in paper.warnings:
            lines.append(f"- {w}\n")
        lines.append("\n")

    lines.append(section("Abstract", paper.abstract))
    lines.append(section("Methods", paper.methods))
    lines.append(section("Results", paper.results))
    lines.append(section("References", paper.references))
    lines.append(section("Appendix", paper.appendix))

    if paper.other_sections:
        lines.append("## Other Sections\n\n")
        for k, v in sorted(paper.other_sections.items()):
            if not v or not v.strip():
                continue
            lines.append(f"### {k.title()}\n\n{v.strip()}\n\n")

    def format_fig_list(title_: str, figs: list[ExtractedFigure]) -> str:
        if not figs:
            return ""
        out: list[str] = [f"## {title_}\n\n"]
        for f in figs:
            where = f"(p{f.page_no})" if f.page_no else ""
            out.append(f"- **{f.figure_id}** {where}\n")
            if f.caption:
                out.append(f"  - caption: {f.caption.strip()}\n")
            if f.alt_text:
                out.append(f"  - alt_text: {f.alt_text.strip()}\n")
            if f.image_urls:
                out.append("  - images:\n")
                for u in f.image_urls:
                    out.append(f"    - <{u}>\n")
        out.append("\n")
        return "".join(out)

    def format_table_list(title_: str, tabs: list[ExtractedTable]) -> str:
        if not tabs:
            return ""
        out: list[str] = [f"## {title_}\n\n"]
        for t in tabs:
            where = f"(p{t.page_no})" if t.page_no else ""
            out.append(f"- **{t.table_id}** {where}\n")
            if t.caption:
                out.append(f"  - caption: {t.caption.strip()}\n")
            if t.image_urls:
                out.append("  - images:\n")
                for u in t.image_urls:
                    out.append(f"    - <{u}>\n")
            if t.html:
                out.append("  - html:\n\n")
                out.append(t.html.strip() + "\n\n")
        out.append("\n")
        return "".join(out)

    lines.append(format_fig_list("Main Figures", paper.main_figures))
    lines.append(format_table_list("Main Tables", paper.main_tables))
    lines.append(format_fig_list("Supplementary Figures", paper.supp_figures))
    lines.append(format_table_list("Supplementary Tables", paper.supp_tables))

    text = "".join(lines)
    # Ensure a trailing newline for POSIX-friendly output.
    if not text.endswith("\n"):
        text += "\n"
    return text


# =============================================================================
# Section extraction
# =============================================================================


_MAJOR_SECTION_ALIASES: dict[str, set[str]] = {
    "abstract": {"ABSTRACT", "SUMMARY"},
    "introduction": {"INTRODUCTION", "BACKGROUND"},
    "results": {"RESULTS", "RESULT"},
    "methods": {
        "METHODS",
        "MATERIALS AND METHODS",
        "MATERIALS METHODS",
        "EXPERIMENTAL PROCEDURES",
        "STAR METHODS",
    },
    "discussion": {"DISCUSSION"},
    "conclusion": {"CONCLUSION", "CONCLUSIONS"},
    "references": {"REFERENCES", "BIBLIOGRAPHY", "LITERATURE CITED"},
    "appendix": {
        "APPENDIX",
        "SUPPLEMENTARY INFORMATION",
        "SUPPLEMENTAL INFORMATION",
        "SUPPLEMENTARY MATERIAL",
        "SUPPLEMENTAL MATERIAL",
    },
}


def extract_title(pages: list[PageRecord], *, max_pages: int = 3) -> str | None:
    """Best-effort title extraction from early pages (first # heading)."""
    for page in pages[: max(1, max_pages)]:
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            m = _HEADING_RE.match(line)
            if not m:
                continue
            level = len(m.group(1))
            if level != 1:
                continue
            candidate = m.group(2).strip()
            if not candidate:
                continue
            # Ignore very short/placeholder headings
            if len(candidate) < 6:
                continue
            return candidate
    return None


def extract_major_sections(pages: list[PageRecord]) -> dict[str, str]:
    """
    Aggregate major section text based on markdown headings.

    Returns:
        Dict of {section_key: text} for keys like: abstract, methods, results, ...
    """
    collected: dict[str, list[str]] = {}
    current_key: str | None = None

    for page in pages:
        for raw_line in page.text.splitlines():
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            m = _HEADING_RE.match(stripped)
            if m:
                heading_text = m.group(2).strip()
                norm = normalize_heading(heading_text)
                key = match_major_section(norm)
                if key:
                    current_key = key
                    collected.setdefault(current_key, [])
                    continue

                # Non-major heading: keep it inside the current section as a subheading.
                if current_key:
                    collected[current_key].append(stripped)
                continue

            # Filter obvious noise lines when aggregating section text.
            if _COMMENT_LINE_RE.match(stripped):
                continue
            if "<img" in stripped.lower():
                continue
            if "<table" in stripped.lower():
                # Keep tables separately; don't pollute section text with long HTML blobs.
                continue
            if stripped == "---":
                continue

            if current_key:
                collected[current_key].append(line)

    out: dict[str, str] = {}
    for key, lines in collected.items():
        text = "\n".join(lines).strip()
        if text:
            out[key] = _collapse_blank_lines(text)
    return out


def normalize_heading(heading: str) -> str:
    """Normalize a heading for robust matching (handles 'STAR★METHODS', punctuation, numbering)."""
    h = heading.strip()
    # Remove leading numbering like "1.", "1.2", "2 -"
    h = re.sub(r"^\s*\d+(?:\.\d+)*\s*[-\.)]?\s*", "", h)
    h = h.upper()
    # Replace non-alnum with spaces
    h = re.sub(r"[^A-Z0-9]+", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def match_major_section(normalized_heading: str) -> str | None:
    for key, aliases in _MAJOR_SECTION_ALIASES.items():
        if normalized_heading in aliases:
            return key
    return None


# =============================================================================
# Media extraction (figures/tables)
# =============================================================================


@dataclass
class _Legend:
    kind: str  # "figure" | "table"
    ident: str
    caption: str
    page_index: int
    page_no: int


@dataclass
class _Image:
    url: str
    page_index: int
    page_no: int
    alt_text: str | None
    legend_on_next_page: bool
    attached_kind: str | None = None  # "figure" | "table"
    attached_id: str | None = None


def extract_media(pages: list[PageRecord]) -> tuple[list[ExtractedFigure], list[ExtractedTable], str | None, list[str]]:
    warnings: list[str] = []

    figure_legends: list[_Legend] = []
    table_legends: list[_Legend] = []
    images: list[_Image] = []
    html_tables: list[ExtractedTable] = []

    for page in pages:
        lines = page.text.splitlines()

        # 1) Legends (Figure/Table lines)
        for legend in _extract_legends_from_lines(lines, page):
            if legend.kind == "figure":
                figure_legends.append(legend)
            else:
                table_legends.append(legend)

        # 2) Images (img tags) + nearby figureText + legend-next hint
        images.extend(_extract_images_from_lines(lines, page))

        # 3) HTML tables
        html_tables.extend(_extract_html_tables_from_text(page.text, page))

    # Build figure map from legends
    figures_by_id: dict[str, ExtractedFigure] = {}
    for leg in figure_legends:
        figures_by_id.setdefault(
            leg.ident,
            ExtractedFigure(
                figure_id=leg.ident,
                caption=leg.caption,
                page_index=leg.page_index,
                page_no=leg.page_no,
                is_supplementary=_is_supplementary_id(leg.ident),
            ),
        )

    # Build table map from legends
    tables_by_id: dict[str, ExtractedTable] = {}
    for leg in table_legends:
        tables_by_id.setdefault(
            leg.ident,
            ExtractedTable(
                table_id=leg.ident,
                caption=leg.caption,
                page_index=leg.page_index,
                page_no=leg.page_no,
                is_supplementary=_is_supplementary_id(leg.ident),
            ),
        )

    # Attach images that have an immediately-following legend on the same page
    # (We detect this inside _extract_images_from_lines and store attached_id when possible.)
    pending_images: list[_Image] = []
    for img in images:
        if img.attached_id and img.attached_kind == "figure":
            fig = figures_by_id.get(img.attached_id)
            if fig is None:
                fig = ExtractedFigure(
                    figure_id=img.attached_id,
                    caption="",
                    page_index=img.page_index,
                    page_no=img.page_no,
                    is_supplementary=_is_supplementary_id(img.attached_id),
                )
                figures_by_id[img.attached_id] = fig
            fig.image_urls.append(img.url)
            if img.alt_text and not fig.alt_text:
                fig.alt_text = img.alt_text
        elif img.attached_id and img.attached_kind == "table":
            tab = tables_by_id.get(img.attached_id)
            if tab is None:
                tab = ExtractedTable(
                    table_id=img.attached_id,
                    caption="",
                    page_index=img.page_index,
                    page_no=img.page_no,
                    is_supplementary=_is_supplementary_id(img.attached_id),
                )
                tables_by_id[img.attached_id] = tab
            tab.image_urls.append(img.url)
        elif img.legend_on_next_page:
            pending_images.append(img)
        else:
            # Unassigned image; keep it as a figure with synthetic id so it's not lost.
            synthetic_id = f"Image@p{img.page_no}"
            suffix = 1
            while synthetic_id in figures_by_id:
                suffix += 1
                synthetic_id = f"Image@p{img.page_no}#{suffix}"
            figures_by_id[synthetic_id] = ExtractedFigure(
                figure_id=synthetic_id,
                caption="",
                page_index=img.page_index,
                page_no=img.page_no,
                image_urls=[img.url],
                alt_text=img.alt_text,
                is_supplementary=False,
            )

    # Cross-page association: attach pending images to the next legend (figure/table) in document order.
    sorted_any_legends = sorted(
        [*figure_legends, *table_legends], key=lambda l: (l.page_index, l.page_no)
    )
    if pending_images and not sorted_any_legends:
        warnings.append("Found images with 'legend on next page' but no legends were detected.")
    for img in pending_images:
        target = next((l for l in sorted_any_legends if l.page_index >= img.page_index), None)
        if not target:
            warnings.append(
                f"Found image on page {img.page_no} with 'legend on next page' but no following legend."
            )
            continue
        if target.kind == "figure":
            fig = figures_by_id.get(target.ident)
            if fig is None:
                fig = ExtractedFigure(
                    figure_id=target.ident,
                    caption=target.caption,
                    page_index=target.page_index,
                    page_no=target.page_no,
                    is_supplementary=_is_supplementary_id(target.ident),
                )
                figures_by_id[target.ident] = fig
            fig.image_urls.append(img.url)
            if img.alt_text and not fig.alt_text:
                fig.alt_text = img.alt_text
        else:
            tab = tables_by_id.get(target.ident)
            if tab is None:
                tab = ExtractedTable(
                    table_id=target.ident,
                    caption=target.caption,
                    page_index=target.page_index,
                    page_no=target.page_no,
                    is_supplementary=_is_supplementary_id(target.ident),
                )
                tables_by_id[target.ident] = tab
            tab.image_urls.append(img.url)

    # Merge HTML tables into tables list (keep them even if no explicit 'Table 1' id exists).
    for t in html_tables:
        tables_by_id.setdefault(t.table_id, t)

    figures = sorted(figures_by_id.values(), key=lambda f: (_figure_sort_key(f.figure_id), f.page_no or 0))
    tables = sorted(tables_by_id.values(), key=lambda t: (_table_sort_key(t.table_id), t.page_no or 0))

    appendix_text = _build_appendix_from_supp_legs(figure_legends, table_legends)
    return figures, tables, appendix_text, warnings


def _extract_legends_from_lines(lines: list[str], page: PageRecord) -> Iterable[_Legend]:
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or _COMMENT_LINE_RE.match(line) or line == "---":
            i += 1
            continue

        m_fig = _FIGURE_LEGEND_RE.match(line)
        m_tab = _TABLE_LEGEND_RE.match(line)
        if not m_fig and not m_tab:
            i += 1
            continue

        kind = "figure" if m_fig else "table"
        if m_fig:
            ident = _normalize_figure_id(m_fig.group(2))
            first = f"{ident}. {m_fig.group(3).strip()}".strip()
        else:
            ident = _normalize_table_id(m_tab.group(1))
            first = f"{ident}. {m_tab.group(2).strip()}".strip()

        cap_lines: list[str] = [first]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].rstrip("\n")
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                cap_lines.append("")
                j += 1
                continue
            if _COMMENT_LINE_RE.match(nxt_stripped):
                j += 1
                continue
            if nxt_stripped == "---":
                break
            if _FIGURE_LEGEND_RE.match(nxt_stripped) or _TABLE_LEGEND_RE.match(nxt_stripped):
                break
            if _HEADING_RE.match(nxt_stripped):
                # A heading usually indicates we've left the legend block.
                break
            cap_lines.append(nxt)
            j += 1

        caption = _collapse_blank_lines("\n".join(cap_lines).strip())
        yield _Legend(kind=kind, ident=ident, caption=caption, page_index=page.page_index, page_no=page.page_no)
        i = j


def _extract_images_from_lines(lines: list[str], page: PageRecord) -> list[_Image]:
    imgs: list[_Image] = []

    # Pre-scan for figureText comments so we can associate the nearest one above an image.
    figuretext_by_line: dict[int, str] = {}
    for idx, line in enumerate(lines):
        m = _FIGURETEXT_RE.search(line)
        if m:
            # Keep it short-ish; figureText can be extremely long/noisy.
            txt = re.sub(r"\s+", " ", m.group(1)).strip()
            if txt:
                figuretext_by_line[idx] = txt[:2000]

    for idx, line in enumerate(lines):
        m = _IMG_RE.search(line)
        if not m:
            continue

        url = m.group(1).strip()
        if not url:
            continue

        # Find nearest figureText comment within 5 lines above
        alt_text = None
        for back in range(idx, max(-1, idx - 6), -1):
            if back in figuretext_by_line:
                alt_text = figuretext_by_line[back]
                break

        # Detect "legend on next page" hint within the next few lines
        legend_on_next = False
        for fwd in range(idx, min(len(lines), idx + 6)):
            if _LEGEND_NEXT_RE.search(lines[fwd]):
                legend_on_next = True
                break

        # If the next non-empty, non-comment line is a figure/table legend, attach directly.
        attached_kind: str | None = None
        attached_id: str | None = None
        for fwd in range(idx + 1, min(len(lines), idx + 25)):
            nxt = lines[fwd].strip()
            if not nxt or _COMMENT_LINE_RE.match(nxt):
                continue
            m_fig = _FIGURE_LEGEND_RE.match(nxt)
            if m_fig:
                attached_kind = "figure"
                attached_id = _normalize_figure_id(m_fig.group(2))
            m_tab = _TABLE_LEGEND_RE.match(nxt)
            if m_tab and not attached_id:
                attached_kind = "table"
                attached_id = _normalize_table_id(m_tab.group(1))
            break

        imgs.append(
            _Image(
                url=url,
                page_index=page.page_index,
                page_no=page.page_no,
                alt_text=alt_text,
                legend_on_next_page=legend_on_next,
                attached_kind=attached_kind,
                attached_id=attached_id,
            )
        )

    return imgs


def _extract_html_tables_from_text(text: str, page: PageRecord) -> list[ExtractedTable]:
    out: list[ExtractedTable] = []
    for idx, m in enumerate(_TABLE_BLOCK_RE.finditer(text)):
        html = m.group(1).strip()
        if not html:
            continue

        # Best-effort caption: nearest previous non-empty, non-comment line within this page.
        caption = _nearest_caption_before_table(text, m.start())
        table_id = _infer_table_id_from_caption(caption) or f"Table@p{page.page_no}#{idx + 1}"
        out.append(
            ExtractedTable(
                table_id=table_id,
                caption=caption or "",
                page_index=page.page_index,
                page_no=page.page_no,
                html=html,
                is_supplementary=_is_supplementary_id(table_id),
            )
        )
    return out


def _nearest_caption_before_table(text: str, table_start: int, *, max_chars: int = 1500) -> str | None:
    # Look backwards from the table start, split into lines, find the last "normal" line.
    window = text[max(0, table_start - max_chars) : table_start]
    lines = window.splitlines()
    for raw in reversed(lines):
        s = raw.strip()
        if not s:
            continue
        if _COMMENT_LINE_RE.match(s) or s == "---":
            continue
        # Skip obvious boilerplate markers
        if s.lower() in {"media", "footnote"}:
            continue
        return s[:300]
    return None


def _infer_table_id_from_caption(caption: str | None) -> str | None:
    if not caption:
        return None
    m = _TABLE_LEGEND_RE.match(caption.strip())
    if m:
        return _normalize_table_id(m.group(1))
    # Some PDFs use "KEY RESOURCES TABLE" etc. Keep as synthetic.
    return None


def _build_appendix_from_supp_legs(fig_legs: list[_Legend], tab_legs: list[_Legend]) -> str | None:
    parts: list[str] = []
    for leg in fig_legs:
        if _is_supplementary_id(leg.ident):
            parts.append(leg.caption.strip())
    for leg in tab_legs:
        if _is_supplementary_id(leg.ident):
            parts.append(leg.caption.strip())
    joined = "\n\n---\n\n".join(p for p in parts if p)
    return joined.strip() if joined.strip() else None


def _normalize_figure_id(num: str) -> str:
    n = num.strip()
    # normalize "s1" -> "S1"
    if n.lower().startswith("s"):
        n = "S" + n[1:]
    return f"Figure {n}"


def _normalize_table_id(num: str) -> str:
    n = num.strip()
    if n.lower().startswith("s"):
        n = "S" + n[1:]
    return f"Table {n}"


def _is_supplementary_id(ident: str) -> bool:
    return bool(re.search(r"\b(?:Table|Figure)\s+S\d+", ident, re.IGNORECASE))


def _figure_sort_key(figure_id: str) -> tuple[int, int]:
    # Main figures first, then supplementary, then synthetic.
    m = re.match(r"^Figure\s+(S?)(\d+)", figure_id, re.IGNORECASE)
    if not m:
        return (2, 10**9)
    is_supp = 1 if m.group(1) else 0
    return (is_supp, int(m.group(2)))


def _table_sort_key(table_id: str) -> tuple[int, int]:
    m = re.match(r"^Table\s+(S?)(\d+)", table_id, re.IGNORECASE)
    if not m:
        return (2, 10**9)
    is_supp = 1 if m.group(1) else 0
    return (is_supp, int(m.group(2)))


# =============================================================================
# Text cleanup helpers
# =============================================================================


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_text_block(text: str | None) -> str | None:
    if not text:
        return None
    # Remove stray carriage returns and excessive blank lines
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _collapse_blank_lines(cleaned)
    return cleaned.strip() if cleaned.strip() else None


def _yaml_quote_string(s: str) -> str:
    """
    Quote a string for YAML if it contains special characters.

    Returns the string quoted with double quotes if it contains characters
    that could cause YAML parsing issues, otherwise returns it as-is.
    """
    if not s:
        return '""'

    # Characters that require quoting in YAML
    # - Leading/trailing whitespace
    # - Colons followed by space
    # - Hash/pound signs
    # - Various special YAML indicators
    needs_quote = (
        s != s.strip()
        or s.startswith(("'", '"', "-", "[", "{", ">", "|", "*", "&", "!", "%", "@", "`"))
        or ": " in s
        or " #" in s
        or s.startswith("#")
        or "\n" in s
        or s.lower() in ("true", "false", "yes", "no", "null", "~")
        or re.match(r"^\d+(\.\d+)?$", s)  # Looks like a number
    )

    if needs_quote:
        # Escape double quotes and backslashes
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    return s



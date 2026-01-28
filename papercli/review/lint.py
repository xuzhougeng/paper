"""Deterministic lint rules for paper text."""

import re
from typing import Callable

from papercli.review.models import LintIssue, LintReport


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

RULES: dict[str, Callable[[str, list[str]], list[LintIssue]]] = {}


def register_rule(rule_id: str):
    """Decorator to register a lint rule."""

    def decorator(func: Callable[[str, list[str]], list[LintIssue]]):
        RULES[rule_id] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _find_line_col(text: str, pos: int) -> tuple[int, int]:
    """Convert character offset to 1-based line and column."""
    lines = text[:pos].split("\n")
    line = len(lines)
    col = len(lines[-1]) + 1 if lines else 1
    return line, col


def _get_context(text: str, start: int, end: int, context_chars: int = 40) -> str:
    """Get context around a span."""
    ctx_start = max(0, start - context_chars)
    ctx_end = min(len(text), end + context_chars)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return prefix + text[ctx_start:ctx_end] + suffix


# ---------------------------------------------------------------------------
# Lint rules
# ---------------------------------------------------------------------------


@register_rule("figure-table-ref")
def check_figure_table_ref(text: str, lines: list[str]) -> list[LintIssue]:
    """Check for inconsistent Figure/Fig. and Table references."""
    issues = []

    # Find all figure references
    fig_patterns = [
        (r"\bFig\.?\s*\d+", "Fig."),
        (r"\bFigure\s+\d+", "Figure"),
    ]

    fig_refs: dict[str, list[int]] = {"Fig.": [], "Figure": []}
    for pattern, style in fig_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Normalize to check actual case used
            matched = m.group()
            if matched.lower().startswith("figure"):
                fig_refs["Figure"].append(m.start())
            else:
                fig_refs["Fig."].append(m.start())

    # If both styles are used, warn about inconsistency
    if fig_refs["Fig."] and fig_refs["Figure"]:
        # Find the minority style
        if len(fig_refs["Fig."]) < len(fig_refs["Figure"]):
            minority = "Fig."
            majority = "Figure"
            positions = fig_refs["Fig."]
        else:
            minority = "Figure"
            majority = "Fig."
            positions = fig_refs["Figure"]

        for pos in positions:
            line, col = _find_line_col(text, pos)
            issues.append(
                LintIssue(
                    rule_id="figure-table-ref",
                    severity="warn",
                    message=f"Inconsistent figure reference style: '{minority}' used but '{majority}' is more common in this document",
                    line=line,
                    column=col,
                    span_start=pos,
                    suggestion=f"Consider using '{majority}' consistently",
                )
            )

    # Check table references similarly
    table_patterns = [
        (r"\bTab\.?\s*\d+", "Tab."),
        (r"\bTable\s+\d+", "Table"),
    ]

    table_refs: dict[str, list[int]] = {"Tab.": [], "Table": []}
    for pattern, style in table_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            matched = m.group()
            if matched.lower().startswith("table"):
                table_refs["Table"].append(m.start())
            else:
                table_refs["Tab."].append(m.start())

    if table_refs["Tab."] and table_refs["Table"]:
        if len(table_refs["Tab."]) < len(table_refs["Table"]):
            minority = "Tab."
            majority = "Table"
            positions = table_refs["Tab."]
        else:
            minority = "Table"
            majority = "Tab."
            positions = table_refs["Table"]

        for pos in positions:
            line, col = _find_line_col(text, pos)
            issues.append(
                LintIssue(
                    rule_id="figure-table-ref",
                    severity="warn",
                    message=f"Inconsistent table reference style: '{minority}' used but '{majority}' is more common",
                    line=line,
                    column=col,
                    span_start=pos,
                    suggestion=f"Consider using '{majority}' consistently",
                )
            )

    return issues


@register_rule("units-format")
def check_units_format(text: str, lines: list[str]) -> list[LintIssue]:
    """Check for common unit formatting issues."""
    issues = []

    # Check for missing space between number and unit
    # Common units in biology: μL, mL, L, μg, mg, g, kg, μM, mM, M, nm, μm, mm, cm, m
    unit_pattern = r"(\d)([μµu]?[LlMmgGnNcCkK][LlMmgGs]?(?![a-zA-Z]))"

    for m in re.finditer(unit_pattern, text):
        # Check if there's no space
        full_match = m.group()
        if not re.match(r"\d\s", full_match):
            line, col = _find_line_col(text, m.start())
            issues.append(
                LintIssue(
                    rule_id="units-format",
                    severity="info",
                    message=f"Consider adding space between number and unit: '{full_match}'",
                    line=line,
                    column=col,
                    span_start=m.start(),
                    span_end=m.end(),
                    context=_get_context(text, m.start(), m.end()),
                    suggestion=f"{m.group(1)} {m.group(2)}",
                )
            )

    # Check for percentage without space (or with space, depending on style)
    # This is style-dependent, so just flag for awareness
    pct_no_space = re.finditer(r"(\d)%", text)
    pct_with_space = re.finditer(r"(\d)\s+%", text)

    no_space_count = len(list(re.finditer(r"\d%", text)))
    with_space_count = len(list(re.finditer(r"\d\s+%", text)))

    # If mixed usage, warn about the minority
    if no_space_count > 0 and with_space_count > 0:
        if no_space_count < with_space_count:
            for m in re.finditer(r"(\d)%", text):
                line, col = _find_line_col(text, m.start())
                issues.append(
                    LintIssue(
                        rule_id="units-format",
                        severity="info",
                        message="Inconsistent percentage formatting (no space before %)",
                        line=line,
                        column=col,
                        span_start=m.start(),
                        span_end=m.end(),
                    )
                )
        else:
            for m in re.finditer(r"(\d)\s+%", text):
                line, col = _find_line_col(text, m.start())
                issues.append(
                    LintIssue(
                        rule_id="units-format",
                        severity="info",
                        message="Inconsistent percentage formatting (space before %)",
                        line=line,
                        column=col,
                        span_start=m.start(),
                        span_end=m.end(),
                    )
                )

    return issues


@register_rule("term-consistency")
def check_term_consistency(text: str, lines: list[str]) -> list[LintIssue]:
    """Check for inconsistent term usage (case and hyphenation)."""
    issues = []

    # Common biology term variations to check
    term_groups = [
        # (canonical, variants to flag)
        ("RNA-seq", ["RNAseq", "RNA seq", "Rnaseq", "rnaseq", "RNA-Seq"]),
        ("scRNA-seq", ["scRNAseq", "scRNA seq", "single-cell RNA-seq", "sc-RNA-seq"]),
        ("CRISPR-Cas9", ["CRISPR/Cas9", "CRISPRCas9", "Crispr-Cas9", "crispr-cas9"]),
        ("ChIP-seq", ["ChIPseq", "ChIP seq", "Chipseq"]),
        ("ATAC-seq", ["ATACseq", "ATAC seq"]),
        ("qRT-PCR", ["qRT PCR", "qrtpcr", "RT-qPCR", "rtqpcr"]),
        ("PCR", ["Pcr", "pcr"]),
        ("DNA", ["Dna", "dna"]),
        ("RNA", ["Rna", "rna"]),
        ("mRNA", ["Mrna", "mrna", "MRNA"]),
        ("miRNA", ["Mirna", "mirna", "MIRNA", "MiRNA"]),
        ("siRNA", ["Sirna", "sirna", "SIRNA", "SiRNA"]),
        ("in vitro", ["in-vitro", "invitro"]),
        ("in vivo", ["in-vivo", "invivo"]),
        ("wild-type", ["wildtype", "wild type", "WT"]),  # WT is often acceptable
        ("knock-out", ["knockout", "knock out"]),
        ("knock-in", ["knockin", "knock in"]),
    ]

    text_lower = text.lower()

    for canonical, variants in term_groups:
        canonical_count = len(
            re.findall(re.escape(canonical), text, re.IGNORECASE)
        ) - len(re.findall(re.escape(canonical), text))

        for variant in variants:
            # Skip if variant is the same as canonical (case-insensitive check artifact)
            if variant.lower() == canonical.lower():
                continue

            for m in re.finditer(re.escape(variant), text):
                # Make sure it's not part of a larger word
                start, end = m.start(), m.end()
                if start > 0 and text[start - 1].isalnum():
                    continue
                if end < len(text) and text[end].isalnum():
                    continue

                line, col = _find_line_col(text, start)
                issues.append(
                    LintIssue(
                        rule_id="term-consistency",
                        severity="warn",
                        message=f"Inconsistent term: '{m.group()}' found, consider using '{canonical}'",
                        line=line,
                        column=col,
                        span_start=start,
                        span_end=end,
                        context=_get_context(text, start, end),
                        suggestion=canonical,
                    )
                )

    return issues


@register_rule("punctuation-mixed")
def check_punctuation_mixed(text: str, lines: list[str]) -> list[LintIssue]:
    """Check for mixed Chinese/English punctuation issues."""
    issues = []

    # Chinese punctuation that might be misused in English text
    chinese_punct = {
        "，": ",",  # Chinese comma
        "。": ".",  # Chinese period
        "；": ";",  # Chinese semicolon
        "：": ":",  # Chinese colon
        "？": "?",  # Chinese question mark
        "！": "!",  # Chinese exclamation
        "（": "(",  # Chinese left paren
        "）": ")",  # Chinese right paren
        """: '"',  # Chinese left quote
        """: '"',  # Chinese right quote
        "'": "'",  # Chinese left single quote
        "'": "'",  # Chinese right single quote
    }

    for ch_punct, en_punct in chinese_punct.items():
        for m in re.finditer(re.escape(ch_punct), text):
            # Check if surrounding text is mostly English
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 20)
            context = text[start:end]

            # Simple heuristic: if context has mostly ASCII letters, flag it
            ascii_letters = sum(1 for c in context if c.isascii() and c.isalpha())
            total_letters = sum(1 for c in context if c.isalpha())

            if total_letters > 0 and ascii_letters / total_letters > 0.7:
                line, col = _find_line_col(text, m.start())
                issues.append(
                    LintIssue(
                        rule_id="punctuation-mixed",
                        severity="warn",
                        message=f"Chinese punctuation '{ch_punct}' in English context",
                        line=line,
                        column=col,
                        span_start=m.start(),
                        span_end=m.end(),
                        context=_get_context(text, m.start(), m.end()),
                        suggestion=en_punct,
                    )
                )

    # Check for missing space after punctuation in English text
    # Pattern: punctuation followed by letter without space
    for m in re.finditer(r"([,;:])([A-Za-z])", text):
        line, col = _find_line_col(text, m.start())
        issues.append(
            LintIssue(
                rule_id="punctuation-mixed",
                severity="info",
                message=f"Missing space after '{m.group(1)}'",
                line=line,
                column=col,
                span_start=m.start(),
                span_end=m.end(),
                context=_get_context(text, m.start(), m.end()),
                suggestion=f"{m.group(1)} {m.group(2)}",
            )
        )

    return issues


@register_rule("case-title")
def check_case_title(text: str, lines: list[str]) -> list[LintIssue]:
    """Check for inconsistent title/heading case style."""
    issues = []

    # Look for markdown headings
    heading_pattern = r"^(#{1,6})\s+(.+)$"

    title_case_count = 0
    sentence_case_count = 0
    heading_info: list[tuple[int, str, str, bool]] = []  # (line_num, level, text, is_title_case)

    for i, line in enumerate(lines, 1):
        m = re.match(heading_pattern, line)
        if m:
            level = m.group(1)
            heading_text = m.group(2).strip()

            # Skip very short headings or those with special formatting
            if len(heading_text) < 3:
                continue

            # Detect if title case or sentence case
            words = heading_text.split()
            if len(words) < 2:
                continue

            # Title case: most words capitalized (excluding small words)
            small_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            capitalized = sum(1 for w in words if w[0].isupper() or w.lower() in small_words)
            is_title_case = capitalized / len(words) > 0.7

            if is_title_case:
                title_case_count += 1
            else:
                sentence_case_count += 1

            heading_info.append((i, level, heading_text, is_title_case))

    # If there's a mix of styles, flag the minority
    if title_case_count > 0 and sentence_case_count > 0:
        minority_is_title = title_case_count < sentence_case_count

        for line_num, level, heading_text, is_title_case in heading_info:
            if is_title_case == minority_is_title:
                expected = "sentence case" if minority_is_title else "title case"
                issues.append(
                    LintIssue(
                        rule_id="case-title",
                        severity="info",
                        message=f"Inconsistent heading case: this heading uses {'title case' if is_title_case else 'sentence case'}, but {expected} is more common in this document",
                        line=line_num,
                        context=heading_text,
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Main lint runner
# ---------------------------------------------------------------------------


def run_lint(
    text: str,
    rules: list[str] | None = None,
    exclude_rules: list[str] | None = None,
) -> LintReport:
    """
    Run lint rules on text.

    Args:
        text: The text to lint
        rules: List of rule IDs to run (default: all rules)
        exclude_rules: List of rule IDs to exclude

    Returns:
        LintReport with all issues found
    """
    lines = text.split("\n")

    # Determine which rules to run
    if rules:
        active_rules = [r for r in rules if r in RULES]
    else:
        active_rules = list(RULES.keys())

    if exclude_rules:
        active_rules = [r for r in active_rules if r not in exclude_rules]

    # Run rules
    all_issues: list[LintIssue] = []
    for rule_id in active_rules:
        rule_func = RULES[rule_id]
        issues = rule_func(text, lines)
        all_issues.extend(issues)

    # Sort by line number
    all_issues.sort(key=lambda x: (x.line or 0, x.column or 0))

    return LintReport(
        issues=all_issues,
        rules_checked=active_rules,
        total_lines=len(lines),
        total_chars=len(text),
    )


def format_lint_markdown(report: LintReport) -> str:
    """Format LintReport as readable Markdown."""
    lines = ["# Lint Report", ""]

    # Summary
    lines.append("## Summary")
    lines.append(f"- **Total issues:** {len(report.issues)}")
    lines.append(f"  - Errors: {report.error_count}")
    lines.append(f"  - Warnings: {report.warn_count}")
    lines.append(f"  - Info: {report.info_count}")
    lines.append(f"- **Lines checked:** {report.total_lines}")
    lines.append(f"- **Rules checked:** {', '.join(report.rules_checked)}")
    lines.append("")

    if not report.issues:
        lines.append("No issues found!")
        return "\n".join(lines)

    # Group by rule
    by_rule: dict[str, list[LintIssue]] = {}
    for issue in report.issues:
        by_rule.setdefault(issue.rule_id, []).append(issue)

    lines.append("## Issues")
    lines.append("")

    for rule_id, issues in by_rule.items():
        lines.append(f"### {rule_id} ({len(issues)} issues)")
        lines.append("")

        for issue in issues:
            severity_icon = {"error": "❌", "warn": "⚠️", "info": "ℹ️"}.get(
                issue.severity, "•"
            )
            loc = f"Line {issue.line}" if issue.line else "Unknown location"
            if issue.column:
                loc += f":{issue.column}"

            lines.append(f"- {severity_icon} **{loc}**: {issue.message}")
            if issue.context:
                lines.append(f"  - Context: `{issue.context}`")
            if issue.suggestion:
                lines.append(f"  - Suggestion: `{issue.suggestion}`")
        lines.append("")

    return "\n".join(lines)

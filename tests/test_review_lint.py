"""Tests for paper review lint functionality."""

import pytest

from papercli.review.lint import (
    run_lint,
    check_figure_table_ref,
    check_units_format,
    check_term_consistency,
    check_punctuation_mixed,
    check_case_title,
    format_lint_markdown,
    RULES,
)
from papercli.review.models import LintIssue, LintReport


class TestFigureTableRef:
    """Tests for figure/table reference consistency."""

    def test_consistent_figure_refs_no_issue(self):
        """Consistent Figure references should not produce issues."""
        text = "As shown in Figure 1, the results are clear. Figure 2 shows more data."
        issues = check_figure_table_ref(text, text.split("\n"))
        assert len(issues) == 0

    def test_consistent_fig_refs_no_issue(self):
        """Consistent Fig. references should not produce issues."""
        text = "As shown in Fig. 1, the results are clear. Fig. 2 shows more data."
        issues = check_figure_table_ref(text, text.split("\n"))
        assert len(issues) == 0

    def test_mixed_figure_refs_produces_warning(self):
        """Mixed Figure/Fig. references should produce warnings."""
        text = "Figure 1 shows X. Fig. 2 shows Y. Figure 3 shows Z."
        issues = check_figure_table_ref(text, text.split("\n"))
        # Fig. 2 is the minority (1 vs 2), so it should be flagged
        assert len(issues) == 1
        assert issues[0].rule_id == "figure-table-ref"
        assert "Fig." in issues[0].message

    def test_table_refs_inconsistency(self):
        """Mixed Table/Tab. references should produce warnings."""
        text = "Table 1 shows X. Tab. 2 shows Y. Table 3 shows Z."
        issues = check_figure_table_ref(text, text.split("\n"))
        assert len(issues) == 1
        assert "Tab." in issues[0].message


class TestUnitsFormat:
    """Tests for units formatting."""

    def test_number_unit_spacing(self):
        """Check for missing space between number and unit."""
        text = "We used 10mL of buffer and 5μg of protein."
        issues = check_units_format(text, text.split("\n"))
        # Should flag both 10mL and 5μg
        assert len(issues) >= 1
        assert any("space" in i.message.lower() for i in issues)

    def test_percentage_consistency(self):
        """Check for inconsistent percentage formatting."""
        text = "The recovery was 50% in group A and 60 % in group B."
        issues = check_units_format(text, text.split("\n"))
        # Should flag inconsistent percentage formatting
        assert len(issues) >= 1


class TestTermConsistency:
    """Tests for biology term consistency."""

    def test_rnaseq_variants(self):
        """Check for RNA-seq variants."""
        text = "We performed RNA-seq analysis. The RNAseq data showed..."
        issues = check_term_consistency(text, text.split("\n"))
        assert len(issues) >= 1
        assert any("RNAseq" in i.message for i in issues)
        assert any(i.suggestion == "RNA-seq" for i in issues)

    def test_crispr_variants(self):
        """Check for CRISPR-Cas9 variants."""
        text = "We used CRISPR-Cas9 editing. The CRISPR/Cas9 system worked well."
        issues = check_term_consistency(text, text.split("\n"))
        assert len(issues) >= 1
        assert any("CRISPR/Cas9" in i.message for i in issues)

    def test_in_vitro_variants(self):
        """Check for in vitro variants."""
        text = "The in vitro experiments showed results. The invitro data confirmed."
        issues = check_term_consistency(text, text.split("\n"))
        assert len(issues) >= 1
        assert any("invitro" in i.message for i in issues)

    def test_correct_terms_no_issue(self):
        """Correct term usage should not produce issues."""
        text = "We performed RNA-seq, ChIP-seq, and ATAC-seq experiments in vitro."
        issues = check_term_consistency(text, text.split("\n"))
        assert len(issues) == 0


class TestPunctuationMixed:
    """Tests for mixed punctuation."""

    def test_chinese_punctuation_in_english(self):
        """Chinese punctuation in English context should be flagged."""
        text = "The experiment was successful，and the results were significant。"
        issues = check_punctuation_mixed(text, text.split("\n"))
        assert len(issues) >= 2
        assert any("，" in i.message for i in issues)
        assert any("。" in i.message for i in issues)

    def test_missing_space_after_punctuation(self):
        """Missing space after punctuation should be flagged."""
        text = "First,we did this. Then,we did that."
        issues = check_punctuation_mixed(text, text.split("\n"))
        assert len(issues) >= 2
        assert all("space" in i.message.lower() for i in issues)

    def test_correct_punctuation_no_issue(self):
        """Correct punctuation should not produce issues."""
        text = "First, we did this. Then, we did that."
        issues = check_punctuation_mixed(text, text.split("\n"))
        # Filter out only the missing space issues
        space_issues = [i for i in issues if "space" in i.message.lower()]
        assert len(space_issues) == 0


class TestCaseTitle:
    """Tests for heading case style."""

    def test_consistent_title_case(self):
        """Consistent title case should not produce issues."""
        text = """# Introduction and Background
## Materials and Methods
### Results and Discussion"""
        issues = check_case_title(text, text.split("\n"))
        assert len(issues) == 0

    def test_consistent_sentence_case(self):
        """Consistent sentence case should not produce issues."""
        text = """# Introduction and background
## Materials and methods
### Results and discussion"""
        issues = check_case_title(text, text.split("\n"))
        assert len(issues) == 0

    def test_mixed_case_styles(self):
        """Mixed case styles should produce warnings."""
        text = """# Introduction and Background
## Materials and methods
### Results and Discussion"""
        issues = check_case_title(text, text.split("\n"))
        # The sentence case heading (Materials and methods) should be flagged
        assert len(issues) >= 1
        assert any("sentence case" in i.message.lower() for i in issues)


class TestRunLint:
    """Tests for the main lint runner."""

    def test_run_all_rules(self):
        """Test running all rules."""
        text = "Figure 1 shows X. Fig. 2 shows Y."
        report = run_lint(text)
        assert isinstance(report, LintReport)
        assert len(report.rules_checked) == len(RULES)

    def test_run_specific_rules(self):
        """Test running specific rules only."""
        text = "Figure 1 shows X. Fig. 2 shows Y."
        report = run_lint(text, rules=["figure-table-ref"])
        assert report.rules_checked == ["figure-table-ref"]

    def test_exclude_rules(self):
        """Test excluding rules."""
        text = "Figure 1 shows X. Fig. 2 shows Y."
        report = run_lint(text, exclude_rules=["figure-table-ref"])
        assert "figure-table-ref" not in report.rules_checked

    def test_report_counts(self):
        """Test report issue counts."""
        # Text with known issues
        text = """Figure 1 shows X. Fig. 2 shows Y.
We used 10mL of buffer."""
        report = run_lint(text)
        assert report.total_lines == 2
        assert report.total_chars == len(text)


class TestFormatLintMarkdown:
    """Tests for markdown formatting."""

    def test_format_empty_report(self):
        """Test formatting an empty report."""
        report = LintReport(
            issues=[],
            rules_checked=["figure-table-ref"],
            total_lines=10,
            total_chars=100,
        )
        md = format_lint_markdown(report)
        assert "# Lint Report" in md
        assert "No issues found" in md

    def test_format_report_with_issues(self):
        """Test formatting a report with issues."""
        report = LintReport(
            issues=[
                LintIssue(
                    rule_id="figure-table-ref",
                    severity="warn",
                    message="Inconsistent figure reference",
                    line=5,
                    suggestion="Use Figure consistently",
                )
            ],
            rules_checked=["figure-table-ref"],
            total_lines=10,
            total_chars=100,
        )
        md = format_lint_markdown(report)
        assert "# Lint Report" in md
        assert "figure-table-ref" in md
        assert "Line 5" in md
        assert "Inconsistent figure reference" in md

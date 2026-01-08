"""Deduplication and coarse ranking for papers."""

import re
from collections import defaultdict

from papercli.models import Paper, QueryIntent


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """
    Remove duplicate papers using multiple strategies.

    Priority:
    1. Exact DOI match
    2. Source-specific ID match (for same source)
    3. Normalized title similarity

    When duplicates are found, prefer papers with more complete metadata.
    """
    if not papers:
        return []

    # Group by DOI
    by_doi: dict[str, list[Paper]] = defaultdict(list)
    no_doi: list[Paper] = []

    for paper in papers:
        if paper.doi:
            by_doi[paper.doi.lower()].append(paper)
        else:
            no_doi.append(paper)

    # Select best paper from each DOI group
    unique_papers: list[Paper] = []
    seen_normalized_titles: set[str] = set()

    for doi, group in by_doi.items():
        best = _select_best_paper(group)
        unique_papers.append(best)
        seen_normalized_titles.add(best.normalized_title)

    # Process papers without DOI
    for paper in no_doi:
        norm_title = paper.normalized_title

        # Check if similar title already exists
        if not _title_exists(norm_title, seen_normalized_titles):
            unique_papers.append(paper)
            seen_normalized_titles.add(norm_title)

    return unique_papers


def _select_best_paper(papers: list[Paper]) -> Paper:
    """Select the best paper from a group of duplicates."""
    if len(papers) == 1:
        return papers[0]

    # Score each paper by completeness
    def completeness_score(p: Paper) -> int:
        score = 0
        if p.abstract:
            score += 3  # Abstract is most valuable
        if p.authors:
            score += 1
        if p.year:
            score += 1
        if p.venue:
            score += 1
        if p.url:
            score += 1
        return score

    return max(papers, key=completeness_score)


def _title_exists(norm_title: str, seen: set[str], threshold: float = 0.9) -> bool:
    """Check if a similar title already exists in the seen set."""
    if norm_title in seen:
        return True

    # Check for high similarity
    for existing in seen:
        if _title_similarity(norm_title, existing) >= threshold:
            return True

    return False


def _title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two normalized titles using Jaccard index."""
    words1 = set(title1.split())
    words2 = set(title2.split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def coarse_rank(papers: list[Paper], intent: QueryIntent, k: int = 20) -> list[Paper]:
    """
    Coarse ranking using lexical matching.

    Scores papers based on:
    - Keyword matches in title (higher weight)
    - Keyword matches in abstract
    - Phrase matches (boost)
    - Synonym matches

    Returns top-k papers by score.
    """
    if not papers:
        return []

    if len(papers) <= k:
        return papers

    # Build search terms
    search_terms = set()

    # Add keywords from intent
    search_terms.update(word.lower() for word in intent.keywords)

    # Add words from query_en
    search_terms.update(word.lower() for word in intent.query_en.split())

    # Add synonyms
    for term, synonyms in intent.synonyms.items():
        search_terms.add(term.lower())
        search_terms.update(s.lower() for s in synonyms)

    # Required phrases (for boost)
    required_phrases = [p.lower() for p in intent.required_phrases]

    # Score each paper
    scored_papers = []
    for paper in papers:
        score = _score_paper(paper, search_terms, required_phrases)
        scored_papers.append((score, paper))

    # Sort by score descending
    scored_papers.sort(key=lambda x: x[0], reverse=True)

    return [paper for _, paper in scored_papers[:k]]


def _score_paper(
    paper: Paper,
    search_terms: set[str],
    required_phrases: list[str],
) -> float:
    """Score a paper based on lexical matching."""
    score = 0.0

    title_lower = paper.title.lower()
    title_words = set(re.findall(r"\w+", title_lower))

    abstract_lower = (paper.abstract or "").lower()
    abstract_words = set(re.findall(r"\w+", abstract_lower))

    # Title matches (weight: 3x)
    title_matches = len(title_words & search_terms)
    score += title_matches * 3.0

    # Abstract matches (weight: 1x)
    abstract_matches = len(abstract_words & search_terms)
    score += abstract_matches * 1.0

    # Phrase matches in title (boost: 5 per phrase)
    for phrase in required_phrases:
        if phrase in title_lower:
            score += 5.0

    # Phrase matches in abstract (boost: 2 per phrase)
    for phrase in required_phrases:
        if phrase in abstract_lower:
            score += 2.0

    # Bonus for having abstract (papers with abstract are more useful)
    if paper.abstract:
        score += 1.0

    return score


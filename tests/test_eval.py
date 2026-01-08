"""Tests for LLM-based reranking and evaluation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from papercli.eval import rerank_with_llm
from papercli.models import Paper, EvalResult


@pytest.mark.asyncio
async def test_exception_preserves_correct_paper_index():
    """Test that exceptions preserve the correct paper index, not completion order.
    
    When tasks complete out of order and one fails, the result should still
    correspond to the correct paper by index. With the bug fixed, exceptions
    are caught inside evaluate_with_semaphore which has access to the actual
    task index, not just the completion counter.
    """
    papers = [
        Paper(source_id=f"id_{i}", source="test", title=f"Title_{i}", abstract=f"Abstract_{i}")
        for i in range(5)
    ]
    
    mock_llm = MagicMock()
    
    # Make tasks complete in non-sequential order: 4, 3, 2, 1, 0
    # Task 2 will raise an exception
    completion_times = {0: 0.05, 1: 0.04, 2: 0.03, 3: 0.02, 4: 0.01}
    
    async def mock_eval_completion(prompt, response_model, system_prompt):
        from papercli.eval import PaperEvaluation
        
        for i, paper in enumerate(papers):
            if paper.title in prompt:
                await asyncio.sleep(completion_times[i])
                
                # Force paper 2 to fail
                if i == 2:
                    raise RuntimeError(f"Simulated failure for paper {i}")
                
                return PaperEvaluation(
                    score=float(i),  # Use valid scores 0-4 (within 0-10 range)
                    meets_need=True,
                    evidence_quote=paper.title,
                    evidence_field="title",
                    short_reason=f"Success for paper {i}"
                )
        raise ValueError("Unknown paper")
    
    mock_llm.eval_completion = AsyncMock(side_effect=mock_eval_completion)
    
    results = await rerank_with_llm(
        query="test",
        papers=papers,
        llm=mock_llm,
        cache=None,
    )
    
    # Verify all papers got results
    assert len(results) == len(papers)
    
    # KEY TEST: Each result must correspond to its correct paper by index
    # With the old bug, results[2] would have the wrong paper
    for i, result in enumerate(results):
        assert result.paper.source_id == papers[i].source_id, \
            f"Result {i} should be for paper with source_id 'id_{i}', got '{result.paper.source_id}'"
        assert result.paper.title == papers[i].title, \
            f"Result {i} should be for paper '{papers[i].title}', got '{result.paper.title}'"
    
    # Paper 2 should have fallback values from exception handling
    assert results[2].score == 1.0
    assert results[2].meets_need is False
    assert "Evaluation error" in results[2].short_reason or "Evaluation failed" in results[2].short_reason
    
    # Other papers should have their correct scores (0, 1, _, 3, 4)
    assert results[0].score == 0.0
    assert results[1].score == 1.0
    assert results[3].score == 3.0
    assert results[4].score == 4.0


@pytest.mark.asyncio
async def test_all_results_match_correct_papers():
    """Verify that results are correctly matched to their papers regardless of completion order."""
    papers = [
        Paper(source_id=f"paper_{i}", source="test", title=f"Title_{i}", abstract=f"Abstract_{i}")
        for i in range(10)
    ]
    
    mock_llm = MagicMock()
    
    async def mock_eval_completion(prompt, response_model, system_prompt):
        from papercli.eval import PaperEvaluation
        # Find which paper this is
        for i, paper in enumerate(papers):
            if paper.title in prompt:
                return PaperEvaluation(
                    score=float(i),  # Use unique score (0-9) within valid range (0-10)
                    meets_need=True,
                    evidence_quote=paper.title,
                    evidence_field="title",
                    short_reason=f"Paper {i}"
                )
        raise ValueError("Paper not found")
    
    mock_llm.eval_completion = AsyncMock(side_effect=mock_eval_completion)
    
    results = await rerank_with_llm(
        query="test",
        papers=papers,
        llm=mock_llm,
        cache=None,
    )
    
    # Each result should match its corresponding paper
    for i, result in enumerate(results):
        assert result.paper.source_id == papers[i].source_id
        assert result.score == float(i), \
            f"Paper {i} should have score {i}, got {result.score}"


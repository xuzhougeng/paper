"""Base class for search source adapters."""

from abc import ABC, abstractmethod

from papercli.models import Paper, QueryIntent


class BaseSource(ABC):
    """Abstract base class for paper search sources."""

    name: str = "base"

    @abstractmethod
    async def search(
        self,
        intent: QueryIntent,
        max_results: int = 20,
    ) -> list[Paper]:
        """
        Search for papers matching the query intent.

        Args:
            intent: Parsed query intent with search terms
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects
        """
        pass


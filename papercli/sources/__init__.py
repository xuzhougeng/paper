"""Search source adapters."""

from papercli.sources.arxiv import ArxivSource
from papercli.sources.base import BaseSource
from papercli.sources.openalex import OpenAlexSource
from papercli.sources.pubmed import PubMedSource
from papercli.sources.scholar import ScholarSource
from papercli.sources.zotero import ZoteroSource

__all__ = [
    "BaseSource",
    "PubMedSource",
    "OpenAlexSource",
    "ScholarSource",
    "ArxivSource",
    "ZoteroSource",
]

"""Shared interface for news providers.

Any future provider can be used by the collector as long as it has a
``fetch_news`` method that returns a list of news items.
"""

from typing import Protocol


class NewsProvider(Protocol):
    """Describe the method a news provider must make available."""

    def fetch_news(self, company_name: str, limit: int = 10) -> list[dict[str, str]]:
        """Return news items for a company."""


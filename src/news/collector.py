"""Provider-neutral entry point for collecting company news."""

from src.news.google_news import GoogleNewsRSSProvider
from src.news.mock_news import get_mock_news
from src.news.provider import NewsProvider


def collect_news(
    company_name: str,
    limit: int = 10,
    provider: NewsProvider | None = None,
) -> list[dict[str, str]]:
    """Collect news, using sample data if the real provider fails.

    Pass a different provider later to change where news comes from without
    changing the rest of the application.
    """
    cleaned_name = company_name.strip()
    if not cleaned_name:
        raise ValueError("Please provide a stock or company name.")
    if limit < 1:
        raise ValueError("The news limit must be at least 1.")

    selected_provider = provider or GoogleNewsRSSProvider()

    try:
        news_items = selected_provider.fetch_news(cleaned_name, limit=limit)
        if news_items:
            return news_items[:limit]
    except Exception:
        # A network, feed, or provider error should not stop a student demo.
        pass

    # This also handles a valid feed that contains no matching articles.
    return get_mock_news(cleaned_name)[:limit]


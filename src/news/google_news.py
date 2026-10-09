"""Free news collection through Google News RSS search results."""

from datetime import date, datetime, timedelta, timezone
import calendar
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import feedparser


class GoogleNewsRSSProvider:
    """Fetch news from the public Google News RSS search feed."""

    feed_url = "https://news.google.com/rss/search"

    def fetch_news(self, company_name: str, limit: int = 10) -> list[dict[str, str]]:
        """Search for recent financial news and return standard news fields."""
        query = f'"{company_name}" stock OR shares OR finance when:30d'
        query_string = urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        request = Request(
            f"{self.feed_url}?{query_string}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; StudentNewsProject/1.0)"},
        )

        # A timeout keeps the app from waiting too long if the feed is unavailable.
        with urlopen(request, timeout=10) as response:
            feed_content = response.read()

        feed = feedparser.parse(feed_content)
        news_items: list[dict[str, str]] = []

        for entry in feed.entries:
            source_details = entry.get("source", {})
            source_name = source_details.get("title", "Google News")
            published_date = _get_publication_date(entry)
            # RSS search may still return stale items; keep only dated items
            # from the last 30 days for the recent-news display.
            if published_date:
                try:
                    if date.fromisoformat(published_date) < date.today() - timedelta(days=30):
                        continue
                except ValueError:
                    continue

            news_items.append(
                {
                    "headline": entry.get("title", "Untitled news article"),
                    "publication_date": published_date,
                    "source": source_name,
                    "article_url": entry.get("link", ""),
                }
            )
            if len(news_items) >= limit:
                break

        return news_items


def _get_publication_date(entry: dict) -> str:
    """Convert an RSS publication date into YYYY-MM-DD, when available."""
    parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed_date:
        return ""

    # RSS dates are interpreted as UTC so dates are consistent across computers.
    timestamp = calendar.timegm(parsed_date)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


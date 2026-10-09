"""Example news items used when the external RSS feed cannot be reached."""

from datetime import date
from urllib.parse import quote


def get_mock_news(company_name: str) -> list[dict[str, str]]:
    """Return clearly labelled sample items for the requested company."""
    today = date.today().isoformat()
    safe_name = quote(company_name.strip().replace(" ", "-"))

    return [
        {
            "headline": f"Sample: {company_name} announces a business update",
            "publication_date": today,
            "source": "Sample Data",
            "article_url": f"https://example.com/mock-news/{safe_name}/business-update",
        },
        {
            "headline": f"Sample: Analysts discuss the outlook for {company_name}",
            "publication_date": today,
            "source": "Sample Data",
            "article_url": f"https://example.com/mock-news/{safe_name}/analyst-outlook",
        },
        {
            "headline": f"Sample: Investors follow recent developments at {company_name}",
            "publication_date": today,
            "source": "Sample Data",
            "article_url": f"https://example.com/mock-news/{safe_name}/investor-update",
        },
    ]


"""Small, testable helpers for the trading-terminal views."""

import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import pandas as pd


STOP_WORDS = {
    "about", "after", "against", "amid", "among", "and", "are", "bank", "company",
    "from", "have", "into", "its", "more", "over", "says", "share", "shares",
    "stock", "that", "their", "this", "through", "with", "will", "year",
}


def top_headline_keywords(
    articles: Sequence[Mapping[str, Any]], limit: int = 10
) -> pd.DataFrame:
    """Rank headline terms by frequency and average headline sentiment."""
    counts: Counter[str] = Counter()
    scores: dict[str, list[float]] = defaultdict(list)
    for article in articles:
        headline = str(article.get("headline", ""))
        try:
            score = float(article.get("sentiment_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        terms = {
            word.lower()
            for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", headline)
            if word.lower() not in STOP_WORDS
        }
        for term in terms:
            counts[term] += 1
            scores[term].append(score)
    rows = [
        {
            "Keyword": term,
            "Headline count": count,
            "Average sentiment": sum(scores[term]) / len(scores[term]),
        }
        for term, count in counts.most_common(limit)
    ]
    return pd.DataFrame(rows, columns=["Keyword", "Headline count", "Average sentiment"])


def sentiment_label_to_status(classification: str) -> str:
    """Map sentiment wording to a neutralized, clearly non-advisory label."""
    return {
        "Bullish": "Buy (sentiment)",
        "Bearish": "Sell (sentiment)",
        "Neutral": "Hold (sentiment)",
    }.get(classification, "N/A")


def daily_headline_sentiment(articles: Sequence[Mapping[str, Any]]) -> pd.Series:
    """Average real headline scores by publication date; exclude mock rows."""
    rows = []
    for article in articles:
        if article.get("source") == "Sample Data":
            continue
        published = pd.to_datetime(article.get("publication_date"), errors="coerce", utc=True)
        if pd.isna(published):
            continue
        try:
            score = float(article["sentiment_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if -1 <= score <= 1:
            rows.append({"date": published.tz_convert(None).normalize(), "score": score})
    if not rows:
        return pd.Series(dtype=float, name="Sentiment")
    result = pd.DataFrame(rows).groupby("date")["score"].mean()
    result.name = "Sentiment"
    return result


def local_deep_dive(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Create a transparent headline-only brief when no AI key is configured."""
    summary = analysis["summary"]
    articles = analysis["articles"]
    real_articles = [article for article in articles if article.get("source") != "Sample Data"]
    risks = [
        article for article in real_articles
        if article.get("sentiment") == "Negative"
    ][:5]
    catalysts = [
        article for article in real_articles
        if article.get("sentiment") == "Positive"
    ][:5]
    company = analysis["company_name"]
    count = summary["article_count"]
    executive = (
        f"{company} has {summary['classification'].lower()} headline sentiment "
        f"(average score {summary['average_sentiment_score']:+.2f}) across {count} "
        "analyzed headlines. This describes the sampled news tone; it is not a forecast."
    )
    return {
        "executive_summary": executive,
        "risks": risks,
        "catalysts": catalysts,
        "is_ai": False,
    }


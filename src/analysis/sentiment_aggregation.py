"""Transparent sentiment and return calculations for one selected stock."""

from math import isfinite
from typing import Any, Mapping, Sequence

import pandas as pd


def aggregate_stock_sentiment(
    articles: Sequence[Mapping[str, Any]],
    price_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize article sentiment and optionally include the latest daily return.

    Each article must have a ``sentiment`` label (Positive, Neutral, or
    Negative) and a ``sentiment_score`` between -1 and +1. The average score
    determines the stock-level classification using the thresholds requested
    for this project.

    If ``price_data`` is provided, it should be the DataFrame returned by
    ``fetch_stock_data``. The function reports its latest available daily
    return; the full daily return history remains in ``price_data``.
    """
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    scores: list[float] = []

    for article_number, article in enumerate(articles, start=1):
        label = str(article.get("sentiment", "")).strip().capitalize()
        if label not in counts:
            raise ValueError(
                f"Article {article_number} has an unknown sentiment label: {label!r}. "
                "Expected Positive, Neutral, or Negative."
            )

        try:
            score = float(article["sentiment_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Article {article_number} must have a numeric sentiment_score."
            ) from error

        if not isfinite(score) or not -1 <= score <= 1:
            raise ValueError(
                f"Article {article_number} sentiment_score must be between -1 and +1."
            )

        counts[label] += 1
        scores.append(score)

    article_count = len(articles)
    if article_count:
        percentages = {
            label: count / article_count * 100
            for label, count in counts.items()
        }
        average_score = sum(scores) / article_count
    else:
        # With no articles there is no measured sentiment, so report zero counts
        # and a neutral default while keeping article_count visible to the user.
        percentages = {label: 0.0 for label in counts}
        average_score = 0.0

    if average_score >= 0.25:
        classification = "Bullish"
    elif average_score <= -0.25:
        classification = "Bearish"
    else:
        classification = "Neutral"

    latest_daily_return, return_date = _get_latest_daily_return(price_data)

    return {
        "article_count": article_count,
        "positive_articles": counts["Positive"],
        "neutral_articles": counts["Neutral"],
        "negative_articles": counts["Negative"],
        "positive_percentage": round(percentages["Positive"], 2),
        "neutral_percentage": round(percentages["Neutral"], 2),
        "negative_percentage": round(percentages["Negative"], 2),
        "average_sentiment_score": average_score,
        "classification": classification,
        "latest_daily_return": latest_daily_return,
        "daily_return_date": return_date,
    }


def _get_latest_daily_return(
    price_data: pd.DataFrame | None,
) -> tuple[float | None, str | None]:
    """Find the latest valid daily return in a stock-price DataFrame."""
    if price_data is None or price_data.empty:
        return None, None

    if "Daily Return" in price_data.columns:
        daily_returns = price_data["Daily Return"]
    elif "Close" in price_data.columns:
        # Also support ordinary price DataFrames that do not have the return
        # column yet: daily return = today's close / yesterday's close - 1.
        daily_returns = price_data["Close"].pct_change()
    else:
        raise ValueError("price_data must contain a 'Daily Return' or 'Close' column.")

    valid_returns = daily_returns.dropna()
    if valid_returns.empty:
        return None, None

    latest_date = valid_returns.index[-1]
    return float(valid_returns.iloc[-1]), str(latest_date)[:10]


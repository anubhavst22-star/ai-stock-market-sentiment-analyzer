"""Align daily news sentiment with stock returns and calculate correlation."""

from math import isfinite
from typing import Any, Mapping, Sequence

import pandas as pd


def analyze_sentiment_vs_returns(
    articles: Sequence[Mapping[str, Any]],
    price_data: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build a trading-day table and calculate Pearson correlation.

    Headlines are grouped by their publication calendar date. They are matched
    to the stock return for the same date; weekend and holiday news is not moved
    to another trading day. Correlation uses only rows with both values.
    """
    price_days = _get_price_days(price_data)
    real_articles = [
        article for article in articles if article.get("source") != "Sample Data"
    ]
    sentiment_days = _get_sentiment_days(real_articles)

    if price_days.empty:
        daily_data = pd.DataFrame(
            columns=["trading_day", "average_news_sentiment", "daily_return"]
        )
    else:
        daily_data = price_days.merge(sentiment_days, on="trading_day", how="left")
        daily_data = daily_data[
            ["trading_day", "average_news_sentiment", "daily_return"]
        ].sort_values("trading_day")

    paired_days = daily_data.dropna(
        subset=["average_news_sentiment", "daily_return"]
    ).copy()
    correlation = None
    if len(paired_days) >= 2:
        value = paired_days["average_news_sentiment"].corr(paired_days["daily_return"])
        if pd.notna(value) and isfinite(float(value)):
            correlation = float(value)

    return {
        "daily_data": daily_data.reset_index(drop=True),
        "paired_data": paired_days.reset_index(drop=True),
        "correlation": correlation,
        "matched_days": len(paired_days),
        "trading_days": len(daily_data),
        "news_days": len(sentiment_days),
        "sample_articles_excluded": len(articles) - len(real_articles),
        "interpretation": explain_correlation(correlation),
    }


def explain_correlation(correlation: float | None) -> str:
    """Describe the direction and approximate strength of a correlation."""
    if correlation is None:
        return (
            "Correlation is unavailable. At least two matched trading days with "
            "both news sentiment and a stock return are needed."
        )

    magnitude = abs(correlation)
    if magnitude < 0.01:
        strength = "no linear"
    elif magnitude < 0.20:
        strength = "very weak"
    elif magnitude < 0.40:
        strength = "weak"
    elif magnitude < 0.60:
        strength = "moderate"
    elif magnitude < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    if magnitude < 0.01:
        return f"Correlation = {correlation:.2f} indicates {strength} relationship in this sample."

    direction = "positive" if correlation > 0 else "negative"
    return f"Correlation = {correlation:.2f} indicates a {strength} {direction} relationship in this sample."


def _get_sentiment_days(articles: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Average valid headline scores by their publication date."""
    rows = []
    for article_number, article in enumerate(articles, start=1):
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

        published_day = pd.to_datetime(
            article.get("publication_date"), errors="coerce"
        )
        if pd.isna(published_day):
            continue
        rows.append(
            {
                "trading_day": pd.Timestamp(published_day).normalize(),
                "sentiment_score": score,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["trading_day", "average_news_sentiment"])

    sentiment_days = pd.DataFrame(rows).groupby("trading_day", as_index=False)[
        "sentiment_score"
    ].mean()
    return sentiment_days.rename(columns={"sentiment_score": "average_news_sentiment"})


def _get_price_days(price_data: pd.DataFrame | None) -> pd.DataFrame:
    """Return one daily return per normalized calendar date."""
    if price_data is None or price_data.empty:
        return pd.DataFrame(columns=["trading_day", "daily_return"])

    if "Daily Return" in price_data.columns:
        daily_returns = price_data["Daily Return"]
    elif "Close" in price_data.columns:
        daily_returns = price_data["Close"].pct_change()
    else:
        raise ValueError("price_data must contain a 'Daily Return' or 'Close' column.")

    days = pd.DatetimeIndex(pd.to_datetime(price_data.index, errors="coerce"))
    # Keep the exchange's calendar date when removing a timezone from the index.
    if days.tz is not None:
        days = days.tz_localize(None)
    prices_by_day = pd.DataFrame(
        {"trading_day": days.normalize(), "daily_return": daily_returns.to_numpy()}
    )
    prices_by_day = prices_by_day.dropna(subset=["trading_day", "daily_return"])
    return prices_by_day.groupby("trading_day", as_index=False)["daily_return"].mean()


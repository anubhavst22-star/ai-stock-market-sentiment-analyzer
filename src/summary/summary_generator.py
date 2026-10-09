"""Create a short educational summary of collected news and analysis."""

import os
from typing import Any, Mapping, Sequence

DEFAULT_MODEL = "gpt-6-astra"


def generate_summary(
    stock_name: str,
    symbol: str,
    exchange: str,
    aggregate: Mapping[str, Any],
    articles: Sequence[Mapping[str, Any]],
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the OpenAI Responses API to summarize the supplied analysis.

    Set the ``OPENAI_API_KEY`` environment variable before running the app.
    Only the headlines and already-calculated figures are sent to the model.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Streamlit Community Cloud stores deployed secrets in st.secrets.
        try:
            import streamlit as st

            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    headline_lines = [
        f"- {article.get('headline', '')} "
        f"(sentiment: {article.get('sentiment', 'Unknown')}, "
        f"score: {float(article.get('sentiment_score', 0)):.2f})"
        for article in articles[:12]
    ]
    if not headline_lines:
        headline_lines = ["- No headlines were available."]

    daily_return = aggregate.get("latest_daily_return")
    daily_return_text = (
        "not available"
        if daily_return is None
        else f"{float(daily_return) * 100:.2f}%"
    )
    prompt = f"""Write a concise academic summary for a PGDM Finance student.

Company: {stock_name} ({symbol}, {exchange})
Overall sentiment: {aggregate.get('classification', 'Unknown')}
Average sentiment score: {float(aggregate.get('average_sentiment_score', 0)):.3f}
Positive / neutral / negative articles: {aggregate.get('positive_articles', 0)} / {aggregate.get('neutral_articles', 0)} / {aggregate.get('negative_articles', 0)}
Latest daily stock return: {daily_return_text}

Headlines and model labels:
{chr(10).join(headline_lines)}

Use two or three plain-English sentences. Explain the observed sentiment and
mention the daily return if available. Do not claim that news caused the return,
do not make a price prediction, and state that the result is based on a limited
sample of headlines."""

    if not api_key:
        return _build_local_summary(stock_name, aggregate, len(articles))

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=180,
        )
        return response.output_text.strip()
    except Exception:
        # A missing network connection or unavailable API should not break the dashboard.
        return _build_local_summary(stock_name, aggregate, len(articles))


def _build_local_summary(
    stock_name: str, aggregate: Mapping[str, Any], article_count: int
) -> str:
    """Build a transparent, rule-based summary when the optional AI API is unavailable."""
    classification = str(aggregate.get("classification", "Neutral")).lower()
    score = float(aggregate.get("average_sentiment_score", 0.0))
    daily_return = aggregate.get("latest_daily_return")
    return_text = (
        "The latest daily stock return was unavailable."
        if daily_return is None
        else f"The latest daily stock return was {float(daily_return) * 100:+.2f}%."
    )
    return (
        f"The headlines for {stock_name} show {classification} sentiment, with an average "
        f"score of {score:+.3f} across {article_count} article(s). {return_text} "
        "This short classroom summary is generated from the displayed sentiment and price "
        "calculations; it describes association and does not imply causation or predict returns."
    )


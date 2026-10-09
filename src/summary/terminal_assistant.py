"""Optional OpenAI-powered deep dive and chat grounded in current analysis data."""

from typing import Any, Mapping

DEFAULT_MODEL = "gpt-4.1-mini"


def _client(api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def generate_deep_dive(context: str, api_key: str, model: str = DEFAULT_MODEL) -> str:
    """Create an executive summary, evidence-linked risks, and catalyst timeline."""
    response = _client(api_key).responses.create(
        model=model,
        instructions=(
            "You are a careful financial research assistant. Use only supplied news and technical "
            "observations. Do not invent company facts, dates, guidance, filings, or causal claims. "
            "Attribute events to source and date. Say when evidence is insufficient. This is educational research, not investment advice.",
        ),
        input=(
            "Prepare three Markdown sections: Executive Summary; Risk Analysis (bullets); "
            "Catalyst Timeline (bullets with dates). Include actual headline evidence.\n\n" + context
        ),
        max_output_tokens=800,
    )
    return response.output_text.strip()


def answer_question(
    question: str, context: str, history: list[Mapping[str, str]],
    api_key: str, model: str = DEFAULT_MODEL,
) -> str:
    """Answer a follow-up using selected-company headlines and technical context."""
    conversation = [
        {"role": "system", "content": (
            "Answer as a cautious financial research assistant. Ground every factual statement "
            "in the supplied news or price metrics. Mention headline date/source when useful. "
            "Say when evidence is missing. Do not invent facts, predict returns, or claim causation. Not investment advice."
        )},
        {"role": "user", "content": f"Current research context:\n{context}"},
    ]
    conversation.extend(
        {"role": message["role"], "content": message["content"]}
        for message in history[-8:]
        if message.get("role") in {"user", "assistant"}
    )
    conversation.append({"role": "user", "content": question})
    response = _client(api_key).responses.create(
        model=model, input=conversation, max_output_tokens=500,
    )
    return response.output_text.strip()


def build_analysis_context(analysis: Mapping[str, Any]) -> str:
    """Serialize collected headlines and concise price indicators for model grounding."""
    prices = analysis.get("prices")
    if prices is None or prices.empty:
        price_context = "Historical OHLC/technical price data: unavailable."
    else:
        closes = prices["Close"].dropna()
        recent_returns = closes.pct_change().dropna().tail(20)
        latest_change = float(recent_returns.iloc[-1]) if not recent_returns.empty else None
        price_context = (
            f"Latest adjusted close: INR {float(closes.iloc[-1]):.2f}; latest daily return: {latest_change:+.2%}."
            if latest_change is not None
            else f"Latest adjusted close: INR {float(closes.iloc[-1]):.2f}; daily return unavailable."
        )
        if len(closes) >= 2:
            price_context += (
                f" Selected-period return: {float(closes.iloc[-1] / closes.iloc[0] - 1):+.2%};"
                f" observed high/low: {float(closes.max()):.2f}/{float(closes.min()):.2f} INR."
            )
        if not recent_returns.empty:
            price_context += f" 20-session annualized volatility: {float(recent_returns.std() * (252 ** 0.5)):.2%}."
        for window in (20, 50):
            if len(closes) >= window:
                price_context += f" SMA{window}: INR {float(closes.tail(window).mean()):.2f}."
        if len(closes) >= 15:
            moves = closes.diff()
            average_gain = float(moves.clip(lower=0).tail(14).mean())
            average_loss = float(-moves.clip(upper=0).tail(14).mean())
            if average_loss == 0:
                rsi = 100.0 if average_gain > 0 else 50.0
            else:
                relative_strength = average_gain / average_loss
                rsi = 100 - (100 / (1 + relative_strength))
            price_context += f" 14-session RSI: {rsi:.1f}."

    headlines = []
    for article in analysis.get("articles", []):
        headlines.append(
            f"- [{article.get('publication_date') or 'date unavailable'}; {article.get('source') or 'publisher unavailable'}] "
            f"{article.get('headline', '')} (sentiment={article.get('sentiment', 'unknown')}, "
            f"score={article.get('sentiment_score', 0):+.2f}; url={article.get('article_url') or 'unavailable'})"
        )
    return (
        f"Company: {analysis.get('company_name')} ({analysis.get('symbol')}, {analysis.get('exchange')})\n"
        f"Date range: {analysis.get('start_date')} to {analysis.get('end_date')}\n"
        f"{price_context}\n"
        f"Sentiment: {analysis['summary']['classification']}, average "
        f"{analysis['summary']['average_sentiment_score']:+.3f} across {analysis['summary']['article_count']} headlines.\n"
        "News items:\n" + ("\n".join(headlines) if headlines else "No headlines were available.")
    )

"""Streamlit dashboard for the AI Stock Market Sentiment Analyzer."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.aggregation import aggregate_sentiment
from src.analysis.backtest import run_educational_backtest
from src.analysis.sentiment import FinancialSentimentAnalyzer
from src.analysis.sentiment_return import (
    build_daily_sentiment_return_data,
    calculate_pearson_correlation,
)
from src.data import fetch_news
from src.data.mock_news import get_mock_news
from src.market import (
    INDIAN_COMPANY_NAMES,
    INDIAN_STOCKS,
    MarketDataError,
    get_demo_stock_data,
    get_stock_data,
)


st.set_page_config(
    page_title="AI Stock Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

# Keep the visual style restrained and readable for a classroom presentation.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.7rem; padding-bottom: 2rem; }
      h1, h2, h3 { color: #17324d; }
      div[data-testid="stMetric"] {
        background: #f5f8fb; border: 1px solid #e3eaf1;
        padding: 0.8rem 1rem; border-radius: 0.55rem;
      }
      .disclaimer { color: #536273; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

COMPANY_NAMES = INDIAN_COMPANY_NAMES
PERIODS = {"1 month": "1mo", "3 months": "3mo", "6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y"}
DEMO_MODE = os.getenv("AI_ANALYZER_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> FinancialSentimentAnalyzer:
    """Keep one model loaded while the Streamlit app is running."""
    use_finbert = not DEMO_MODE
    env_setting = os.getenv("AI_ANALYZER_USE_FINBERT")
    if env_setting is not None:
        use_finbert = env_setting.strip().lower() not in {"0", "false", "no", "off"}
    try:
        # Optional Streamlit Cloud setting for deployments with limited memory.
        if "use_finbert" in st.secrets:
            setting = str(st.secrets["use_finbert"]).strip().lower()
            use_finbert = setting not in {"0", "false", "no", "off"}
    except Exception:
        # A missing secrets.toml is normal; no secret is required by default.
        pass
    return FinancialSentimentAnalyzer(use_finbert=use_finbert)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(company_name: str) -> list[dict[str, Any]]:
    """Cache news results briefly so reruns do not repeatedly fetch the feed."""
    if DEMO_MODE:
        return get_mock_news(company_name, limit=30)
    return fetch_news(company_name, limit=30)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol: str, period: str) -> pd.DataFrame:
    """Cache price history briefly to reduce repeated source requests."""
    if DEMO_MODE:
        return get_demo_stock_data(symbol, period)
    return get_stock_data(symbol, period=period)


def make_ai_insight(
    symbol: str,
    metrics: dict[str, Any],
    articles: list[dict[str, Any]],
    correlation: float | None,
    correlation_explanation: str,
    comparison_days: int,
    model_name: str,
    price_error: str | None,
    price_is_demo: bool,
) -> dict[str, Any]:
    """Create a transparent template-based insight from computed evidence."""
    count = metrics["article_count"]
    label = metrics["classification"]
    score = metrics["average_sentiment_score"]
    distribution = (
        f"{metrics['positive_percentage']:.1f}% positive, "
        f"{metrics['neutral_percentage']:.1f}% neutral and "
        f"{metrics['negative_percentage']:.1f}% negative"
    )
    parts = [
        f"{symbol}: {count} headlines were classified. Overall sentiment is {label.lower()} "
        f"with an average score of {score:+.3f} ({distribution}).",
        f"Headline labels were produced using {model_name}.",
    ]
    if metrics["daily_return_pct"] is not None:
        parts.append(
            f"The latest available daily return was {metrics['daily_return_pct']:+.2f}% "
            f"on {metrics['daily_return_date']}."
        )
    elif price_error:
        parts.append("A daily return is unavailable because the price source could not provide data.")
    if correlation is None:
        parts.append(correlation_explanation)
    else:
        parts.append(
            f"Pearson correlation was {correlation:+.2f} across {comparison_days} matched dates. "
            "This describes association and does not show that news caused price changes."
        )

    positive_factors = _headline_examples(articles, "Positive", reverse=True)
    negative_factors = _headline_examples(articles, "Negative", reverse=False)
    limitations = [
        "News coverage is limited to the headlines returned by the free RSS provider; it may be incomplete or duplicated across publishers.",
        "Sentiment describes headline language, not the full article or the market's reaction.",
        "The news-return relationship is descriptive and does not establish causation.",
        "Historical prices and this academic model do not predict future returns or provide investment advice.",
    ]
    if any(item.get("source", "").startswith("DEMO DATA") for item in articles):
        limitations.append("DEMO DATA headlines are fictional examples, not reported company events.")
    if price_is_demo:
        limitations.append("DEMO DATA prices are synthetic examples and do not represent actual market prices.")
    return {
        "overview": " ".join(parts),
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "limitations": limitations,
    }


def _headline_examples(
    articles: list[dict[str, Any]], sentiment: str, reverse: bool
) -> list[str]:
    """Return up to two scored headlines, without treating them as causal drivers."""
    matches = [item for item in articles if item.get("sentiment") == sentiment]
    matches.sort(key=lambda item: float(item.get("sentiment_score", 0)), reverse=reverse)
    return [str(item.get("headline", "")).strip() for item in matches[:2] if item.get("headline")]


def compare_stocks(
    symbols: list[str], period: str, analyzer: FinancialSentimentAnalyzer
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect a comparable sentiment and recent-return row for each stock."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate in symbols:
        try:
            news = load_news(COMPANY_NAMES[candidate])
            analyzed = analyzer.analyze_headlines(news)
            try:
                prices = load_prices(candidate, period)
                price_source = "DEMO DATA" if DEMO_MODE else "Live Yahoo Finance"
            except (MarketDataError, ValueError) as error:
                prices = get_demo_stock_data(candidate, period)
                price_source = "DEMO DATA"
                errors.append(f"{candidate}: live price source unavailable; synthetic DEMO DATA is shown ({error})")
            metrics = aggregate_sentiment(candidate, analyzed, prices)
            rows.append(
                {
                    "Stock": candidate,
                    "Company": COMPANY_NAMES[candidate],
                    "Sentiment": metrics["classification"],
                    "Sentiment score": metrics["average_sentiment_score"],
                    "Positive %": metrics["positive_percentage"],
                    "Neutral %": metrics["neutral_percentage"],
                    "Negative %": metrics["negative_percentage"],
                    "Recent daily return %": metrics["daily_return_pct"],
                    "Return date": metrics["daily_return_date"],
                    "Price data": price_source,
                    "News data": "DEMO DATA" if analyzed and all(
                        item.get("source", "").startswith("DEMO DATA") for item in analyzed
                    ) else "Live RSS",
                }
            )
        except Exception as error:
            errors.append(f"{candidate}: analysis unavailable ({error})")
    return rows, errors


def render_dashboard() -> None:
    st.title("AI Stock Market Sentiment Analyzer")
    st.caption("Explore how financial-news headlines and recent stock prices move over time.")

    with st.sidebar:
        st.header("Analysis settings")
        with st.form("analysis_form"):
            symbol = st.selectbox(
                "Select an Indian stock",
                options=list(INDIAN_STOCKS),
                format_func=lambda item: f"{item} — {COMPANY_NAMES[item]}",
            )
            period_label = st.selectbox("Price history", options=list(PERIODS), index=2)
            submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)
        st.caption("News search and sentiment analysis use the selected company name.")

    if submitted:
        st.session_state.pop("analysis_output", None)
        st.session_state.pop("analysis_error", None)
        st.session_state.pop("comparison_output", None)
        st.session_state.pop("comparison_errors", None)
        company_name = COMPANY_NAMES[symbol]
        price_data = None
        price_error = None
        price_source = "DEMO DATA" if DEMO_MODE else "Live Yahoo Finance"
        try:
            with st.spinner("Collecting news, analyzing headlines and retrieving stock prices…"):
                news_articles = load_news(company_name)
                analyzer = get_sentiment_analyzer()
                analyzed_articles = analyzer.analyze_headlines(news_articles)
                model_name = analyzer.model_used
                try:
                    price_data = load_prices(symbol, PERIODS[period_label])
                except (MarketDataError, ValueError) as error:
                    price_error = str(error)
                    price_data = get_demo_stock_data(symbol, PERIODS[period_label])
                    price_source = "DEMO DATA"
                metrics = aggregate_sentiment(symbol, analyzed_articles, price_data)
                backtest = run_educational_backtest(analyzed_articles, price_data)

            st.session_state["analysis_output"] = {
                "symbol": symbol,
                "period_label": period_label,
                "period": PERIODS[period_label],
                "articles": analyzed_articles,
                "prices": price_data,
                "metrics": metrics,
                "backtest": backtest,
                "model_name": model_name,
                "price_error": price_error,
                "price_source": price_source,
            }
        except Exception as error:
            st.session_state["analysis_error"] = str(error)

    output = st.session_state.get("analysis_output")
    if not output:
        analysis_error = st.session_state.get("analysis_error")
        if analysis_error:
            st.error(f"The analysis could not be completed: {analysis_error}")
        else:
            st.info("Choose a stock and select **Analyze** to load its news and price history.")
        st.markdown(
            '<p class="disclaimer">This application is developed for academic and educational purposes only and does not constitute investment advice.</p>',
            unsafe_allow_html=True,
        )
        return

    symbol = output["symbol"]
    articles = output["articles"]
    prices = output["prices"]
    metrics = output["metrics"]
    price_error = output["price_error"]

    news_is_demo = bool(articles) and all(
        article.get("source", "").startswith("DEMO DATA") for article in articles
    )
    if news_is_demo or output["price_source"] == "DEMO DATA":
        demo_sources = []
        if news_is_demo:
            demo_sources.append("fictional sample headlines")
        if output["price_source"] == "DEMO DATA":
            demo_sources.append("synthetic sample prices")
        st.warning(
            "DEMO DATA is being shown for " + " and ".join(demo_sources) +
            ". It is not real news or market data."
        )
    if price_error:
        st.caption(f"Live price source message: {price_error}")

    st.subheader(f"{symbol} · {COMPANY_NAMES[symbol]}")
    latest_close = float(prices["Close"].dropna().iloc[-1]) if prices is not None and not prices.empty else None
    daily_return = metrics["daily_return_pct"]
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Latest adjusted close", f"₹{latest_close:,.2f}" if latest_close is not None else "Unavailable")
    kpi_columns[1].metric("Latest daily return", f"{daily_return:+.2f}%" if daily_return is not None else "Unavailable")
    kpi_columns[2].metric("Overall sentiment", metrics["classification"])
    kpi_columns[3].metric("Average sentiment score", f"{metrics['average_sentiment_score']:+.3f}", help="Mean of the individual headline scores; range is −1 to +1.")

    st.caption(
        f"Analyzed {metrics['article_count']} headlines · Price history: {output['period_label']} · "
        f"Sentiment model: {output['model_name']} · News: "
        f"{'DEMO DATA' if news_is_demo else 'Google News RSS'} · Price: {output['price_source']}"
    )

    st.subheader("News sentiment")
    sentiment_columns = st.columns(3)
    sentiment_columns[0].metric("Positive", f"{metrics['positive_percentage']:.1f}%", f"{metrics['positive_articles']} articles")
    sentiment_columns[1].metric("Neutral", f"{metrics['neutral_percentage']:.1f}%", f"{metrics['neutral_articles']} articles")
    sentiment_columns[2].metric("Negative", f"{metrics['negative_percentage']:.1f}%", f"{metrics['negative_articles']} articles")

    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.markdown("#### Sentiment distribution")
        distribution = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Articles": [metrics["positive_articles"], metrics["neutral_articles"], metrics["negative_articles"]],
                "Percentage": [metrics["positive_percentage"], metrics["neutral_percentage"], metrics["negative_percentage"]],
            }
        )
        chart = px.bar(
            distribution,
            x="Sentiment",
            y="Percentage",
            text=distribution["Percentage"].map(lambda value: f"{value:.1f}%"),
            color="Sentiment",
            color_discrete_map={"Positive": "#3c8d76", "Neutral": "#8b98a5", "Negative": "#bf5b55"},
            custom_data=["Articles"],
        )
        chart.update_traces(hovertemplate="%{x}<br>%{y:.1f}%<br>Articles: %{customdata[0]}<extra></extra>")
        chart.update_layout(showlegend=False, yaxis_title="Share of analyzed headlines", xaxis_title="", yaxis_range=[0, 100], margin=dict(t=10, b=10))
        st.plotly_chart(chart, use_container_width=True)
        st.caption("Percentages use all analyzed headlines as the denominator.")

    with right_chart:
        st.markdown("#### Sentiment trend")
        trend_data = pd.DataFrame(articles)
        if not trend_data.empty and "publication_date" in trend_data and "sentiment_score" in trend_data:
            trend_data["publication_date"] = pd.to_datetime(trend_data["publication_date"], errors="coerce")
            trend_data = trend_data.dropna(subset=["publication_date"])
            if not trend_data.empty:
                by_date = trend_data.groupby("publication_date", as_index=False)["sentiment_score"].mean()
                trend = px.line(by_date, x="publication_date", y="sentiment_score", markers=True)
                trend.add_hline(y=0, line_dash="dot", line_color="#8b98a5")
                trend.update_layout(xaxis_title="Publication date", yaxis_title="Average headline score", yaxis_range=[-1, 1], margin=dict(t=10, b=10))
                st.plotly_chart(trend, use_container_width=True)
                st.caption("Each point is the average headline score for that publication date.")
            else:
                st.info("No usable publication dates are available for a trend chart.")
        else:
            st.info("No sentiment records are available for a trend chart.")

    st.subheader("Stock price history")
    if prices is not None and not prices.empty:
        price_chart_data = prices.reset_index()
        price_chart = px.line(price_chart_data, x="Date", y="Close", labels={"Date": "Date", "Close": "Adjusted close (₹)"})
        price_chart.update_traces(line_color="#315f85")
        price_chart.update_layout(margin=dict(t=10, b=10), yaxis_title="Adjusted close (₹)")
        st.plotly_chart(price_chart, use_container_width=True)
        if output["price_source"] == "DEMO DATA":
            st.caption("Synthetic demonstration prices; these are not real market observations.")
        else:
            st.caption("Prices are adjusted for splits and dividends by the data provider. The first daily return is undefined because there is no prior close in the selected period.")
    else:
        st.info("No stock price history is available for this selection.")

    st.subheader("Sentiment vs Stock Price Analysis")
    paired_data = build_daily_sentiment_return_data(articles, prices)
    correlation, correlation_explanation = calculate_pearson_correlation(paired_data)
    correlation_columns = st.columns([1, 3])
    correlation_columns[0].metric(
        "Pearson correlation (r)",
        f"{correlation:+.2f}" if correlation is not None else "Unavailable",
        help="Pearson's r measures the direction and strength of a linear relationship; its range is −1 to +1.",
    )
    correlation_columns[1].write(correlation_explanation)

    if not paired_data.empty:
        comparison_chart = go.Figure()
        comparison_chart.add_trace(
            go.Scatter(
                x=paired_data["date"],
                y=paired_data["daily_sentiment_score"],
                name="Average news sentiment",
                mode="lines+markers",
                line={"color": "#315f85", "width": 2},
                hovertemplate="%{x|%d %b %Y}<br>Average sentiment: %{y:+.3f}<extra></extra>",
            )
        )
        comparison_chart.add_trace(
            go.Scatter(
                x=paired_data["date"],
                y=paired_data["daily_return_pct"],
                name="Stock daily return",
                mode="lines+markers",
                yaxis="y2",
                line={"color": "#c1843d", "width": 2, "dash": "dot"},
                hovertemplate="%{x|%d %b %Y}<br>Daily return: %{y:+.2f}%<extra></extra>",
            )
        )
        comparison_chart.update_layout(
            xaxis_title="Trading date",
            yaxis={"title": "Average news sentiment score", "range": [-1, 1], "zeroline": True},
            yaxis2={"title": "Stock daily return (%)", "overlaying": "y", "side": "right", "zeroline": False},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            hovermode="x unified",
            margin={"t": 55, "b": 10},
        )
        st.plotly_chart(comparison_chart, use_container_width=True)
        st.caption(
            f"Matched dates: {len(paired_data)}. Sentiment is the average score of headlines published on that date; "
            "returns use adjusted closing prices. The left and right axes use different units."
        )
        with st.expander("View daily paired values"):
            daily_table = paired_data.rename(
                columns={
                    "date": "Trading date",
                    "daily_sentiment_score": "Average news sentiment score",
                    "daily_return_pct": "Stock daily return (%)",
                }
            )
            st.dataframe(
                daily_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Trading date": st.column_config.DateColumn("Trading date"),
                    "Average news sentiment score": st.column_config.NumberColumn("Average news sentiment score", format="%+.3f"),
                    "Stock daily return (%)": st.column_config.NumberColumn("Stock daily return (%)", format="%+.2f%%"),
                },
            )
    else:
        st.info(
            "No dates have both a dated news headline and a valid daily stock return, so a comparison chart cannot be shown."
        )

    st.info("Correlation does not imply causation.")
    insight = make_ai_insight(
        symbol,
        metrics,
        articles,
        correlation,
        correlation_explanation,
        len(paired_data),
        output["model_name"],
        price_error,
        output["price_source"] == "DEMO DATA",
    )
    st.subheader("AI Insight")
    st.write(insight["overview"])
    positive_column, negative_column = st.columns(2)
    with positive_column:
        st.markdown("**Headline factors classified positive**")
        if insight["positive_factors"]:
            for headline in insight["positive_factors"]:
                st.write(f"• {headline}")
        else:
            st.write("No positive headlines were classified in this news sample.")
    with negative_column:
        st.markdown("**Headline factors classified negative**")
        if insight["negative_factors"]:
            for headline in insight["negative_factors"]:
                st.write(f"• {headline}")
        else:
            st.write("No negative headlines were classified in this news sample.")
    with st.expander("Important limitations"):
        for limitation in insight["limitations"]:
            st.write(f"• {limitation}")
    st.caption("This insight is assembled from model labels and calculations; no separate text-generation service is used.")

    st.subheader("Educational Historical Backtest")
    backtest = output["backtest"]
    st.caption("Long-only demonstration: BUY above +0.25, SELL below −0.25 (exit to cash), otherwise HOLD. A signal observed at a day's close applies to the next trading day's return.")
    if backtest["strategy_return_pct"] is None:
        st.info("Backtest results are unavailable because valid historical closing prices could not be retrieved.")
    else:
        backtest_columns = st.columns(5)
        backtest_columns[0].metric("Strategy return", f"{backtest['strategy_return_pct']:+.2f}%")
        buy_hold = backtest["buy_and_hold_return_pct"]
        backtest_columns[1].metric("Buy-and-hold return", f"{buy_hold:+.2f}%" if buy_hold is not None else "Unavailable")
        backtest_columns[2].metric("Trades", str(backtest["number_of_trades"]), help="A trade is one BUY entry, including an open position marked at the final close.")
        backtest_columns[3].metric("Winning trades", str(backtest["winning_trades"]))
        backtest_columns[4].metric("Losing trades", str(backtest["losing_trades"]))
        backtest_data = backtest["daily_results"]
        if not backtest_data.empty:
            backtest_chart = px.line(
                backtest_data,
                x="date",
                y=["strategy_cumulative_return_pct", "buy_and_hold_cumulative_return_pct"],
                labels={
                    "date": "Trading date",
                    "value": "Cumulative return (%)",
                    "variable": "Method",
                    "strategy_cumulative_return_pct": "Sentiment strategy",
                    "buy_and_hold_cumulative_return_pct": "Buy and hold",
                },
            )
            backtest_chart.update_layout(margin={"t": 10, "b": 10}, yaxis_title="Cumulative return (%)")
            st.plotly_chart(backtest_chart, use_container_width=True)
        st.caption("The strategy holds cash after SELL and carries its position during HOLD. An open position at the final date is marked to market and counted in trade outcomes. No fees, taxes, or slippage are included.")
    st.warning("EDUCATIONAL HISTORICAL BACKTEST only. Past performance does not guarantee future results.")

    st.subheader("Stock Comparison")
    with st.form("stock_comparison_form"):
        comparison_symbols = st.multiselect(
            "Select up to 5 stocks",
            options=list(INDIAN_STOCKS),
            default=[symbol],
            max_selections=5,
            format_func=lambda item: f"{item} — {COMPANY_NAMES[item]}",
            key="comparison_symbols",
        )
        comparison_submitted = st.form_submit_button("Compare selected stocks", type="secondary")
    if comparison_submitted:
        if not comparison_symbols:
            st.session_state.pop("comparison_output", None)
            st.session_state.pop("comparison_errors", None)
            st.session_state["comparison_error"] = "Select at least one stock to compare."
        else:
            with st.spinner("Preparing stock comparison…"):
                comparison_rows, comparison_errors = compare_stocks(
                    comparison_symbols,
                    output["period"],
                    get_sentiment_analyzer(),
                )
            st.session_state["comparison_output"] = comparison_rows
            st.session_state["comparison_errors"] = comparison_errors
            st.session_state.pop("comparison_error", None)

    comparison_error = st.session_state.get("comparison_error")
    if comparison_error:
        st.info(comparison_error)
    comparison_rows = st.session_state.get("comparison_output", [])
    if comparison_rows:
        comparison_table = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sentiment score": st.column_config.NumberColumn("Sentiment score", format="%+.3f"),
                "Positive %": st.column_config.NumberColumn("Positive %", format="%.1f%%"),
                "Neutral %": st.column_config.NumberColumn("Neutral %", format="%.1f%%"),
                "Negative %": st.column_config.NumberColumn("Negative %", format="%.1f%%"),
                "Recent daily return %": st.column_config.NumberColumn("Recent daily return", format="%+.2f%%"),
            },
        )
        comparison_chart_columns = st.columns(2)
        with comparison_chart_columns[0]:
            score_chart = px.bar(
                comparison_table,
                x="Stock",
                y="Sentiment score",
                color="Sentiment",
                color_discrete_map={"Bullish": "#3c8d76", "Neutral": "#8b98a5", "Bearish": "#bf5b55"},
                range_y=[-1, 1],
                title="Average sentiment score",
            )
            st.plotly_chart(score_chart, use_container_width=True)
        with comparison_chart_columns[1]:
            return_chart = px.bar(
                comparison_table.dropna(subset=["Recent daily return %"]),
                x="Stock",
                y="Recent daily return %",
                color_discrete_sequence=["#315f85"],
                title="Latest daily stock return",
            )
            st.plotly_chart(return_chart, use_container_width=True)
        if any(row["News data"] == "DEMO DATA" for row in comparison_rows):
            st.warning("One or more comparison rows use DEMO DATA headlines; check the News data column.")
        if any(row["Price data"] == "DEMO DATA" for row in comparison_rows):
            st.warning("One or more comparison rows use synthetic DEMO DATA prices; check the Price data column.")
    for message in st.session_state.get("comparison_errors", []):
        st.warning(message)

    st.subheader("Analyzed news")
    if articles:
        news_table = pd.DataFrame(articles)
        table_columns = [
            "headline", "publication_date", "source", "sentiment", "sentiment_score", "confidence", "article_url"
        ]
        news_table = news_table[[column for column in table_columns if column in news_table.columns]]
        st.dataframe(
            news_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "headline": st.column_config.TextColumn("Headline", width="large"),
                "publication_date": st.column_config.TextColumn("Publication date"),
                "sentiment_score": st.column_config.NumberColumn("Score", format="%+.3f"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.1%"),
                "article_url": st.column_config.LinkColumn("Article link", display_text="Open"),
            },
        )
    else:
        st.info("No articles were returned for this company.")

    with st.expander("Viva notes: how to explain the calculations"):
        st.markdown(
            """
            - FinBERT (when available) estimates positive, neutral, and negative probabilities for each headline.
            - A headline score is positive probability minus negative probability, so it ranges from −1 to +1. If FinBERT is unavailable, VADER supplies a compound score and confidence is not reported.
            - Overall score is the arithmetic mean of headline scores. Score ≥ 0.25 is Bullish; score ≤ −0.25 is Bearish; values in between are Neutral.
            - Daily return (%) = (latest adjusted close ÷ previous adjusted close − 1) × 100.
            - Pearson correlation is calculated from dates with both a news score and a daily return; it measures linear association, not causation.
            - Correlation does not imply causation.
            """
        )

    st.markdown(
        '<p class="disclaimer">This application is developed for academic and educational purposes only and does not constitute investment advice.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_dashboard()



"""Streamlit dashboard for the AI Stock Market Sentiment Analyzer."""

from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import aggregate_stock_sentiment, analyze_sentiment_vs_returns
from src.market_data import INDIAN_STOCKS, MarketDataError, fetch_stock_data
from src.news import collect_news
from src.sentiment import analyze_news
from src.summary import generate_summary


st.set_page_config(
    page_title="AI Stock Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e7ebf1;
        padding: 1rem 1.1rem;
        border-radius: 0.65rem;
    }
    [data-testid="stMetricLabel"] {color: #526174;}
    div[data-testid="stSidebar"] {border-right: 1px solid #e7ebf1;}
    </style>
    """,
    unsafe_allow_html=True,
)


def run_analysis(symbol: str, exchange: str, period: str, article_limit: int) -> dict[str, Any]:
    """Collect data and calculate all dashboard results for the selected stock."""
    company_name = INDIAN_STOCKS[symbol]
    raw_articles = collect_news(company_name, limit=article_limit)
    articles = analyze_news(raw_articles)

    prices = None
    price_error = None
    try:
        prices = fetch_stock_data(symbol, exchange=exchange, period=period)
    except MarketDataError as error:
        price_error = str(error)

    summary = aggregate_stock_sentiment(articles, prices)
    comparison = analyze_sentiment_vs_returns(articles, prices)
    ai_summary = None
    summary_error = None
    try:
        ai_summary = generate_summary(
            stock_name=company_name,
            symbol=symbol,
            exchange=exchange,
            aggregate=summary,
            articles=articles,
        )
    except Exception as error:
        summary_error = str(error)

    return {
        "symbol": symbol,
        "company_name": company_name,
        "exchange": exchange,
        "period": period,
        "articles": articles,
        "prices": prices,
        "summary": summary,
        "sentiment_price_analysis": comparison,
        "ai_summary": ai_summary,
        "price_error": price_error,
        "summary_error": summary_error,
        "analyzed_on": date.today().isoformat(),
    }


def show_price_information(result: dict[str, Any]) -> None:
    """Display the latest price and return information as metric cards."""
    prices = result["prices"]
    if prices is None or prices.empty:
        st.warning(result["price_error"] or "Stock price data is unavailable.")
        return

    latest = prices.iloc[-1]
    latest_close = float(latest["Close"])
    latest_return = result["summary"]["latest_daily_return"]
    first_close = float(prices["Close"].iloc[0])
    period_return = latest_close / first_close - 1 if first_close else 0.0
    price_date = str(prices.index[-1])[:10]

    st.caption(f"Adjusted historical prices · latest available date: {price_date} · currency: INR")
    price_col, daily_col, period_col = st.columns(3)
    price_col.metric("Latest closing price", f"₹{latest_close:,.2f}")
    daily_col.metric(
        "Latest daily return",
        "N/A" if latest_return is None else f"{latest_return * 100:+.2f}%",
    )
    period_col.metric(f"Return over selected period ({result['period']})", f"{period_return * 100:+.2f}%")


def show_sentiment_charts(result: dict[str, Any]) -> None:
    """Draw the sentiment distribution and dated headline sentiment trend."""
    summary = result["summary"]
    articles = result["articles"]
    distribution = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Articles": [
                summary["positive_articles"],
                summary["neutral_articles"],
                summary["negative_articles"],
            ],
        }
    )
    colors = {"Positive": "#2f7d62", "Neutral": "#8795a8", "Negative": "#b85c5c"}
    pie = px.pie(
        distribution,
        names="Sentiment",
        values="Articles",
        hole=0.58,
        color="Sentiment",
        color_discrete_map=colors,
    )
    pie.update_traces(textposition="inside", textinfo="percent+label")
    pie.update_layout(
        margin=dict(l=8, r=8, t=12, b=8),
        legend_title_text="",
        showlegend=False,
    )

    sentiment_rows = []
    for article in articles:
        try:
            published = pd.to_datetime(article.get("publication_date"), errors="coerce")
            score = float(article["sentiment_score"])
        except (TypeError, ValueError, KeyError):
            continue
        if not pd.isna(published):
            sentiment_rows.append({"Date": published, "Score": score})

    if sentiment_rows:
        trend = pd.DataFrame(sentiment_rows).groupby("Date", as_index=False)["Score"].mean()
        line = px.line(trend, x="Date", y="Score", markers=True)
        line.update_traces(line_color="#365d86", marker_color="#365d86")
        line.add_hline(y=0, line_dash="dot", line_color="#aab3bf")
        line.update_yaxes(range=[-1, 1], title="Average headline score")
        line.update_xaxes(title="Publication date")
        line.update_layout(margin=dict(l=8, r=8, t=12, b=8), showlegend=False)
    else:
        line = None

    chart_col, trend_col = st.columns(2)
    with chart_col:
        st.subheader("Sentiment distribution")
        st.plotly_chart(pie, use_container_width=True)
    with trend_col:
        st.subheader("Sentiment trend")
        if line is None:
            st.info("Publication dates are unavailable, so a sentiment trend cannot be plotted.")
        else:
            st.plotly_chart(line, use_container_width=True)


def show_stock_price_chart(result: dict[str, Any]) -> None:
    """Draw the selected stock's adjusted closing price over time."""
    prices = result["prices"]
    st.subheader("Stock price history")
    if prices is None or prices.empty:
        st.info("The price chart is unavailable because no historical prices were retrieved.")
        return

    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices["Close"],
            mode="lines",
            name="Adjusted close",
            line=dict(color="#365d86", width=2),
        )
    )
    chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        margin=dict(l=8, r=8, t=12, b=8),
        hovermode="x unified",
    )
    st.plotly_chart(chart, use_container_width=True)


def show_sentiment_price_analysis(result: dict[str, Any]) -> None:
    """Compare same-day average news sentiment with the daily stock return."""
    st.subheader("Sentiment vs Stock Price Analysis")
    comparison = result["sentiment_price_analysis"]
    correlation = comparison["correlation"]
    matched_data = comparison["paired_data"]

    correlation_col, matched_col = st.columns(2)
    correlation_col.metric(
        "Pearson correlation",
        "N/A" if correlation is None else f"{correlation:+.2f}",
    )
    matched_col.metric("Matched trading days", comparison["matched_days"])
    st.write(comparison["interpretation"])

    if comparison["sample_articles_excluded"]:
        st.info(
            f"{comparison['sample_articles_excluded']} mock news article(s) were excluded "
            "from this comparison because they are placeholders, not real market news."
        )

    st.caption(
        "Headlines are grouped by publication date and matched to returns from the same "
        "calendar date. Weekend or holiday news is not shifted to another session. "
        "Correlation uses only days with both a news score and a valid return."
    )

    if matched_data.empty:
        st.info(
            "There are no trading days with both real, dated news and a daily return. "
            "Try again when the news feed has matching articles."
        )
    else:
        chart = make_subplots(specs=[[{"secondary_y": True}]])
        chart.add_trace(
            go.Scatter(
                x=matched_data["trading_day"],
                y=matched_data["average_news_sentiment"],
                mode="lines+markers",
                name="Average news sentiment",
                line=dict(color="#365d86", width=2),
            ),
            secondary_y=False,
        )
        chart.add_trace(
            go.Scatter(
                x=matched_data["trading_day"],
                y=matched_data["daily_return"] * 100,
                mode="lines+markers",
                name="Daily stock return",
                line=dict(color="#2f7d62", width=2),
            ),
            secondary_y=True,
        )
        chart.update_yaxes(title_text="Average news sentiment (-1 to +1)", range=[-1, 1], secondary_y=False)
        chart.update_yaxes(title_text="Daily stock return (%)", secondary_y=True)
        chart.update_xaxes(title_text="Trading day")
        chart.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=8, r=8, t=28, b=8),
        )
        st.plotly_chart(chart, use_container_width=True)

    daily_table = comparison["daily_data"].copy()
    if not daily_table.empty:
        daily_table["daily_return_pct"] = daily_table["daily_return"] * 100
        daily_table = daily_table[
            ["trading_day", "average_news_sentiment", "daily_return_pct"]
        ].rename(
            columns={
                "trading_day": "Trading day",
                "average_news_sentiment": "Average news sentiment",
                "daily_return_pct": "Daily stock return (%)",
            }
        )
        with st.expander("View daily sentiment and return calculations"):
            st.dataframe(daily_table, use_container_width=True, hide_index=True)

    st.info("Correlation does not imply causation.")


def show_news_table(articles: list[dict[str, Any]]) -> None:
    """Show collected articles with sentiment and clickable article links."""
    st.subheader("Recent financial news")
    if not articles:
        st.info("No news articles were returned.")
        return

    table = pd.DataFrame(articles)
    display_columns = {
        "headline": "Headline",
        "publication_date": "Publication date",
        "source": "Source",
        "sentiment": "Sentiment",
        "sentiment_score": "Sentiment score",
        "article_url": "Article URL",
    }
    visible = table[[column for column in display_columns if column in table.columns]].rename(
        columns=display_columns
    )
    if "Sentiment score" in visible.columns:
        visible["Sentiment score"] = visible["Sentiment score"].map(lambda value: round(float(value), 3))
    st.dataframe(
        visible,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Article URL": st.column_config.LinkColumn("Article", display_text="Open article"),
            "Headline": st.column_config.TextColumn("Headline", width="large"),
        },
    )
    if any(article.get("source") == "Sample Data" for article in articles):
        st.caption("Sample Data rows are placeholders used when the news feed has no results or is unavailable.")


st.title("AI Stock Market Sentiment Analyzer")
st.caption("A student dashboard for exploring financial headlines, sentiment, and historical returns.")

with st.sidebar:
    st.header("Analysis settings")
    display_choices = [f"{symbol} — {company}" for symbol, company in INDIAN_STOCKS.items()]
    selected_choice = st.selectbox("Select an Indian stock", display_choices)
    selected_symbol = selected_choice.split(" — ", maxsplit=1)[0]
    selected_exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=0)
    selected_period = st.selectbox(
        "Price history",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3,
    )
    article_limit = st.slider("Number of recent headlines", min_value=5, max_value=20, value=10)
    analyze_clicked = st.button("Analyze stock", type="primary", use_container_width=True)
    st.divider()
    st.caption("A classroom summary is always available; an OpenAI key can add a generated summary.")

if analyze_clicked:
    with st.spinner("Collecting news, analyzing headlines, and loading prices…"):
        try:
            st.session_state["analysis_result"] = run_analysis(
                selected_symbol,
                selected_exchange,
                selected_period,
                article_limit,
            )
        except Exception as error:
            st.error(f"The analysis could not be completed: {error}")
            st.session_state.pop("analysis_result", None)

result = st.session_state.get("analysis_result")
if result is None:
    st.info("Choose a stock and select **Analyze stock** to load its news, sentiment, and price history.")
else:
    st.markdown(f"### {result['company_name']} · {result['symbol']} · {result['exchange']}")
    st.caption(f"Analysis run on {result['analyzed_on']} · prices shown in Indian rupees (INR)")

    st.subheader("Stock price information")
    show_price_information(result)

    summary = result["summary"]
    st.subheader("Overall sentiment")
    overall_col, score_col, positive_col, neutral_col, negative_col = st.columns(5)
    overall_col.metric("Classification", summary["classification"])
    score_col.metric("Average score", f"{summary['average_sentiment_score']:+.3f}")
    positive_col.metric("Positive", f"{summary['positive_percentage']:.2f}%")
    neutral_col.metric("Neutral", f"{summary['neutral_percentage']:.2f}%")
    negative_col.metric("Negative", f"{summary['negative_percentage']:.2f}%")
    st.caption(
        f"Based on {summary['article_count']} headlines: "
        f"{summary['positive_articles']} positive, {summary['neutral_articles']} neutral, "
        f"and {summary['negative_articles']} negative."
    )
    with st.expander("How the calculations work"):
        st.markdown(
            "- **Headline score:** FinBERT positive probability minus negative probability, from -1 to +1.\n"
            "- **Average sentiment:** the mean of the headline scores.\n"
            "- **Classification:** Bullish at 0.25 or above; Bearish at -0.25 or below; Neutral between those values.\n"
            "- **Daily return:** today’s adjusted closing price divided by the previous trading day’s adjusted close, minus 1."
        )

    show_sentiment_charts(result)
    show_stock_price_chart(result)
    show_sentiment_price_analysis(result)
    show_news_table(result["articles"])

    st.subheader("AI-generated summary")
    if result["ai_summary"]:
        st.write(result["ai_summary"])
    elif result["summary_error"] and "OPENAI_API_KEY" in result["summary_error"]:
        st.info(
            "AI summary is not configured yet. Set the OPENAI_API_KEY environment variable "
            "and run the analysis again. The news, sentiment, and price sections work without it."
        )
    else:
        st.warning(f"The AI summary could not be generated: {result['summary_error']}")

st.divider()
st.caption(
    "This application is developed for academic and educational purposes only "
    "and does not constitute investment advice."
)

"""AlphaPulse: an evidence-led Indian equities sentiment research terminal."""

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import aggregate_stock_sentiment, analyze_sentiment_vs_returns
from src.analysis.terminal_features import (
    daily_headline_sentiment, local_deep_dive,
    sentiment_label_to_status, top_headline_keywords,
)
from src.market_data import (
    INDIAN_STOCKS, CompanyUniverseError, MarketDataError,
    fetch_company_research, fetch_nse_equity_universe, fetch_stock_data,
)
from src.news import collect_news
from src.sentiment import analyze_news
from src.summary.terminal_assistant import (
    DEFAULT_MODEL, answer_question, build_analysis_context, generate_deep_dive,
)


st.set_page_config(layout="wide", page_title="AlphaPulse // AI Market Terminal")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #070b12; color: #e6edf7; }
    [data-testid="stHeader"] { background: rgba(7, 11, 18, .92); }
    [data-testid="stSidebar"] { background: #0b111b; border-right: 1px solid #202c3d; }
    [data-testid="stMetric"] { background: #0d1623; border: 1px solid #1d2b3e; border-radius: 12px; padding: 16px; }
    [data-testid="stMetricLabel"] { color: #98a8be; }
    [data-testid="stMetricValue"] { color: #f3f7fc; }
    [data-testid="stTabs"] button { color: #aab7c8; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #64d5c2; border-bottom-color: #64d5c2; }
    div[data-testid="stExpander"] { border: 1px solid #1d2b3e; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_universe() -> tuple[pd.DataFrame, str]:
    return fetch_nse_equity_universe()


@st.cache_data(ttl=20 * 60, show_spinner=False)
def load_analysis(
    symbol: str, company_name: str, exchange: str,
    start_iso: str, end_iso: str, article_limit: int,
) -> dict[str, Any]:
    """Fetch and calculate one analysis; cache market/news data briefly."""
    start_day, end_day = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
    articles = analyze_news(collect_news(company_name, limit=article_limit))
    prices, price_error = None, None
    try:
        # yfinance treats end as exclusive; add one day to include the selected end.
        prices = fetch_stock_data(
            symbol, exchange=exchange,
            start_date=start_day.isoformat(),
            end_date=(end_day + timedelta(days=1)).isoformat(),
        )
    except (MarketDataError, ValueError) as error:
        price_error = str(error)
    summary = aggregate_stock_sentiment(articles, prices)
    comparison = analyze_sentiment_vs_returns(articles, prices)
    return {
        "symbol": symbol, "company_name": company_name, "exchange": exchange,
        "start_date": start_iso, "end_date": end_iso, "articles": articles,
        "prices": prices, "summary": summary, "comparison": comparison,
        "price_error": price_error,
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="minutes"),
    }


def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _daily_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    return result[~result.index.duplicated(keep="last")].sort_index()


def _show_overview(data: dict[str, Any]) -> None:
    summary, prices = data["summary"], data["prices"]
    price, change = None, None
    if prices is not None and not prices.empty:
        closes = prices["Close"].dropna()
        if not closes.empty:
            price = float(closes.iloc[-1])
            if len(closes) > 1 and closes.iloc[-2] != 0:
                change = price / float(closes.iloc[-2]) - 1

    cards = st.columns(4)
    cards[0].metric("Current price", "—" if price is None else f"₹{price:,.2f}")
    cards[1].metric(
        "24h change", "—" if change is None else f"{change:+.2%}",
        delta=None if change is None else f"{change:+.2%}", delta_color="normal",
    )
    cards[2].metric("Overall sentiment", f"{summary['average_sentiment_score']:+.2f}")
    cards[3].metric("AI status · sentiment only", sentiment_label_to_status(summary["classification"]))
    st.caption("24h change means the latest available daily close-to-close move. The Buy/Sell/Hold label is mapped from headline sentiment; it is not an investment recommendation.")

    st.subheader("Price and daily news sentiment")
    if prices is None or prices.empty:
        st.warning(data["price_error"] or "OHLC price data is unavailable for this ticker/provider.")
        return
    days = _daily_price_frame(prices)
    sentiment = daily_headline_sentiment(data["articles"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(
        x=days.index, open=days["Open"], high=days["High"],
        low=days["Low"], close=days["Close"], name="OHLC price",
        increasing_line_color="#26c99a", decreasing_line_color="#ff647c",
    ), secondary_y=False)
    if not sentiment.empty:
        fig.add_trace(go.Scatter(
            x=sentiment.index, y=sentiment.values, name="Daily headline sentiment",
            mode="lines+markers", line={"color": "#59a9ff", "width": 2},
            marker={"size": 7}, connectgaps=False,
        ), secondary_y=True)
    fig.update_layout(
        template="plotly_dark", height=520, paper_bgcolor="#070b12",
        plot_bgcolor="#0a111b", margin={"l": 12, "r": 12, "t": 24, "b": 8},
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    fig.update_yaxes(title_text="Price (INR)", secondary_y=False, gridcolor="#1d2b3e")
    fig.update_yaxes(title_text="Sentiment (-1 to +1)", range=[-1, 1], secondary_y=True, showgrid=False)
    fig.update_xaxes(gridcolor="#1d2b3e")
    st.plotly_chart(fig, use_container_width=True)
    source_exchange = prices.attrs.get("exchange", data["exchange"])
    st.caption(f"Price provider: Yahoo Finance via yfinance · actual source exchange: {source_exchange} · retrieved {data['retrieved_at']}")
    correlation = data["comparison"]["correlation"]
    st.metric("Same-day sentiment / return correlation", "Unavailable" if correlation is None else f"{correlation:+.2f}")
    st.caption("Pearson correlation uses matching publication/trading dates. Correlation does not imply causation.")


def _show_sentiment(data: dict[str, Any]) -> None:
    summary = data["summary"]
    cards = st.columns(4)
    cards[0].metric("Positive headlines", f"{summary['positive_percentage']:.1f}%", f"{summary['positive_articles']} articles")
    cards[1].metric("Neutral headlines", f"{summary['neutral_percentage']:.1f}%", f"{summary['neutral_articles']} articles")
    cards[2].metric("Negative headlines", f"{summary['negative_percentage']:.1f}%", f"{summary['negative_articles']} articles")
    cards[3].metric("Mean sentiment score", f"{summary['average_sentiment_score']:+.3f}")
    if any(a.get("source") == "Sample Data" for a in data["articles"]):
        st.warning("Some headlines are sample placeholders because the news feed returned no usable articles. Treat their sentiment as demo data.")

    st.subheader("Most frequent headline keywords")
    keywords = top_headline_keywords(data["articles"], limit=10)
    if keywords.empty:
        st.info("There are not enough headlines to calculate keyword polarity.")
    else:
        colors = [
            "#19d18f" if score > 0.05 else "#ff526f" if score < -0.05 else "#718096"
            for score in keywords["Average sentiment"]
        ]
        bars = go.Figure(go.Bar(
            x=keywords["Headline count"], y=keywords["Keyword"], orientation="h",
            marker_color=colors, customdata=keywords[["Average sentiment"]],
            hovertemplate="%{y}<br>Headline count: %{x}<br>Mean polarity: %{customdata[0]:+.2f}<extra></extra>",
        ))
        bars.update_layout(
            template="plotly_dark", height=390, paper_bgcolor="#070b12",
            plot_bgcolor="#0a111b", margin={"l": 8, "r": 16, "t": 8, "b": 8},
            xaxis_title="Headline count", yaxis={"autorange": "reversed", "title": ""},
        )
        st.plotly_chart(bars, use_container_width=True)
        st.caption("Keyword color uses the mean score of headlines containing the term; it is not a separate keyword sentiment model.")

    st.subheader("News and model output")
    if not data["articles"]:
        st.info("No articles are available.")
        return
    table = pd.DataFrame(data["articles"])
    columns = [key for key in (
        "headline", "publication_date", "source", "sentiment",
        "sentiment_score", "confidence", "article_url",
    ) if key in table.columns]
    table = table[columns].rename(columns={
        "headline": "Headline", "publication_date": "Published",
        "source": "Publisher", "sentiment": "Sentiment",
        "sentiment_score": "Score", "confidence": "Confidence",
        "article_url": "Original article",
    })
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={"Original article": st.column_config.LinkColumn("Original article", display_text="Open")},
    )


def _show_whale_flow() -> None:
    st.warning("Illustrative placeholder only — this view has no live insider, institutional, or options-flow feed.")
    st.markdown("SEC Form 4 is a U.S. disclosure and is not the relevant filing feed for Indian-listed companies. No verified Indian insider or put/call data source is connected.")
    left, right = st.columns([1, 1.2])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=50,
            title={"text": "Demo positioning gauge"},
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8191a6"},
                "bar": {"color": "#77879b"}, "bgcolor": "#0a111b",
                "bordercolor": "#26354a",
                "steps": [
                    {"range": [0, 40], "color": "#291622"},
                    {"range": [40, 60], "color": "#202a38"},
                    {"range": [60, 100], "color": "#142a26"},
                ],
            },
        ))
        gauge.update_layout(template="plotly_dark", height=300, paper_bgcolor="#070b12", margin={"t": 45, "b": 8})
        st.plotly_chart(gauge, use_container_width=True)
    with right:
        st.subheader("Hooks for a future verified feed")
        st.markdown(
            "- Institutional ownership disclosures and quarter-end changes\n"
            "- Promoter/insider transactions from exchange filings\n"
            "- Options put/call ratio, if a licensed consistent source is added\n"
            "- Timestamp, original source link, and reported instrument per event"
        )
        st.caption("The neutral gauge value is a mockup, not an observed positioning estimate.")


def _show_deep_dive(data: dict[str, Any], api_key: str, model: str) -> None:
    context = build_analysis_context(data)
    local = local_deep_dive(data)
    st.subheader("Executive summary")
    st.write(local["executive_summary"])
    st.caption("The summary below is based only on the selected headlines; it is not a forecast.")
    left, right = st.columns(2)
    with left:
        st.subheader("Risk analysis")
        if local["risks"]:
            for item in local["risks"]:
                st.markdown(f"- **{item.get('publication_date') or 'Date unavailable'} · {item.get('source', 'Publisher unavailable')}** — {item['headline']}")
        else:
            st.info("No negative headlines appeared in this sample; that does not mean risks are absent.")
    with right:
        st.subheader("Catalyst timeline")
        if local["catalysts"]:
            for item in local["catalysts"]:
                st.markdown(f"- **{item.get('publication_date') or 'Date unavailable'}** — {item['headline']}")
        else:
            st.info("No positive headline catalysts were identified in this sample.")

    if st.button("Generate AI deep dive", type="primary", disabled=not api_key):
        try:
            with st.spinner("Preparing a sourced executive review…"):
                st.session_state["deep_dive_text"] = generate_deep_dive(context, api_key, model)
                st.session_state["deep_dive_symbol"] = data["symbol"]
        except Exception as error:
            st.error(f"AI review failed; the headline-based summary above remains available. Details: {error}")
    if not api_key:
        st.info("Add an API key in the sidebar to generate an AI-written synthesis.")
    elif st.session_state.get("deep_dive_symbol") == data["symbol"] and st.session_state.get("deep_dive_text"):
        st.markdown("#### AI deep dive")
        st.markdown(st.session_state["deep_dive_text"])

    with st.expander("Company profile and exchange filings"):
        try:
            profile = fetch_company_research(data["symbol"], data["exchange"])
            if profile.get("business_summary"):
                st.write(profile["business_summary"])
            else:
                st.info("Business overview is unavailable from the profile provider.")
            st.caption(f"Provider: {profile['provider']} · Retrieved {profile['retrieved_at']} · [Company profile]({profile['source_url']})")
            st.warning(profile["limitations"])
        except Exception as error:
            st.info(f"Company profile unavailable for this ticker/provider: {error}")
        st.markdown("[NSE corporate announcements](https://www.nseindia.com/companies-listing/corporate-filings) · [NSE financial results](https://www.nseindia.com/companies-listing/corporate-filings-financial-results)")

    st.divider()
    st.subheader("Ask the research assistant")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about risks, catalysts, headlines, or price data")
    if question:
        previous_history = st.session_state["chat_history"][:]
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if api_key:
                try:
                    answer = answer_question(question, context, previous_history, api_key, model)
                except Exception as error:
                    answer = f"AI service error: {error}. Check the key and try again."
            else:
                answer = "Configure an API key in the sidebar for contextual AI answers. The headline evidence and sentiment labels are available above."
            st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})


st.title("AlphaPulse // AI Market Terminal")
st.caption("Indian equities · headline sentiment · OHLC price action · research assistant")

with st.sidebar:
    st.header("Terminal controls")
    universe, universe_error = None, None
    try:
        universe, universe_updated = load_universe()
    except Exception as error:
        universe_error = str(error)
        universe_updated = None
    ticker_names = dict(INDIAN_STOCKS)
    if universe is not None and not universe.empty:
        ticker_names.update(dict(zip(universe["symbol"], universe["company_name"])))
        ticker_values = universe["symbol"].drop_duplicates().tolist()
    else:
        ticker_values = list(INDIAN_STOCKS)
    ticker_values = list(dict.fromkeys([s for s in INDIAN_STOCKS if s in ticker_values] + ticker_values))
    selected_symbol = st.selectbox(
        "Ticker", ticker_values, index=0,
        format_func=lambda symbol: f"{symbol} — {ticker_names.get(symbol, symbol)}",
    )
    manual_symbol = st.text_input("Or enter ticker", placeholder="e.g. RELIANCE").strip().upper()
    if manual_symbol:
        selected_symbol = manual_symbol
        ticker_names.setdefault(selected_symbol, selected_symbol)
    selected_exchange = st.selectbox("Exchange for price history", ["NSE", "BSE"], index=0)
    today = date.today()
    date_range = st.date_input(
        "Date range", value=(today - timedelta(days=180), today),
        min_value=today - timedelta(days=3650), max_value=today,
    )
    api_key_input = st.text_input(
        "OpenAI API key (optional)", type="password",
        help="Used for requests in this session only; this app does not save it.",
    )
    api_key = api_key_input or _secret("OPENAI_API_KEY")
    model = _secret("OPENAI_MODEL", DEFAULT_MODEL)
    analyze_clicked = st.button("Run analysis", type="primary", use_container_width=True)
    if universe_error:
        st.caption("Company list unavailable; showing the seven built-in example tickers.")
    elif universe is not None:
        st.caption(f"Official company list retrieved {universe_updated} · {len(universe):,} records")
    st.caption("AI status is sentiment-derived, not investment advice.")

valid_dates = isinstance(date_range, (tuple, list)) and len(date_range) == 2
if not valid_dates:
    st.warning("Select both a start and end date.")
else:
    start_date, end_date = date_range
    if start_date > end_date:
        st.warning("The start date must be on or before the end date.")
        valid_dates = False

if analyze_clicked and valid_dates:
    try:
        with st.spinner(f"Loading {selected_symbol}: headlines, sentiment, and prices…"):
            result = load_analysis(
                selected_symbol, ticker_names.get(selected_symbol, selected_symbol),
                selected_exchange, start_date.isoformat(), end_date.isoformat(), 20,
            )
        st.session_state["terminal_analysis"] = result
        st.session_state["chat_history"] = []
        st.session_state.pop("deep_dive_text", None)
        st.session_state.pop("deep_dive_symbol", None)
    except Exception as error:
        st.error(f"Analysis could not be completed: {error}")

analysis = st.session_state.get("terminal_analysis")
if analysis is None:
    st.info("Choose a ticker and date range in the sidebar, then run an analysis to open the terminal views.")
else:
    st.markdown(f"### {analysis['company_name']} · {analysis['symbol']} · {analysis['exchange']}")
    st.caption(f"Analysis window: {analysis['start_date']} to {analysis['end_date']} · retrieved {analysis['retrieved_at']}")
    overview_tab, sentiment_tab, whale_tab, ai_tab = st.tabs([
        "📊 Market Overview", "📰 Sentiment Engine",
        "🐳 Whale & Insider Flow", "🤖 AI Deep-Dive",
    ])
    with overview_tab:
        _show_overview(analysis)
    with sentiment_tab:
        _show_sentiment(analysis)
    with whale_tab:
        _show_whale_flow()
    with ai_tab:
        _show_deep_dive(analysis, api_key, model)
    st.divider()
    st.caption("Educational use only; not investment advice. Verify every headline and filing at its original source. Sentiment does not establish causation or predict returns.")


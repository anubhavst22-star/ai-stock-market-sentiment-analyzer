# AlphaPulse // AI Market Terminal

AlphaPulse is a Streamlit research dashboard for Indian listed equities. It brings price action, financial-news sentiment, headline themes, and an optional AI research assistant into one terminal-style interface.

## Dashboard sections

- **Market Overview:** price, latest daily change, sentiment-derived status, and an interactive OHLC candlestick chart overlaid with dated news sentiment.
- **Sentiment Engine:** positive/neutral/negative headline shares, top headline terms colored by average article polarity, and original article links.
- **Whale & Insider Flow:** clearly marked design placeholder. Its gauge is mock-only; the project does not yet load verified insider, institutional, or options-flow data.
- **AI Deep-Dive:** a headline-based executive summary, risks, catalyst timeline, company profile/filing links, optional AI synthesis, and contextual chat.

## Run on Windows

Use Python 3.12, open PowerShell in this folder, and run:

    py -3.12 -m venv .venv
    ./.venv/Scripts/python.exe -m pip install -r requirements.txt
    ./.venv/Scripts/python.exe -m streamlit run app.py

The app prints a local URL, normally http://localhost:8501. Keep PowerShell open while you use it.

## Data and keys

- Company choices come from NSE's daily CM MII security master for NSE-listed and BSE-exclusive securities. The app tries recent trading dates, caches the list for six hours, and falls back to NSE's official equity CSV if the combined file is unavailable. The fallback is NSE-only and is labeled in the directory.
- Historical OHLC data comes from Yahoo Finance through yfinance. The date range is user-selected. Some BSE-exclusive stocks may not have Yahoo price history.
- Headlines come from Google News RSS and are limited to the last 30 days where publication dates exist. Publisher attribution and original links are shown when present. Sample fallback headlines are labeled.
- FinBERT is attempted first; VADER is used if FinBERT cannot load. Scores are headline-text estimates, not verification of an event.
- Company profile and annual figures come from Yahoo Finance as secondary data. Missing values remain unavailable; verify material financial information against exchange/company filings.
- The AI assistant uses the OpenAI Responses API only after you enter a key or configure OPENAI_API_KEY in Streamlit secrets. A typed key is used in the current session and is not written to disk by the app. AI requests send the selected news headlines and summarized technical data to the API. Without a key, headline-based summary sections remain available, but AI generation and contextual chat are disabled.

For Streamlit Community Cloud, add OPENAI_API_KEY and optionally OPENAI_MODEL under the app's Settings → Secrets. Never commit API keys to GitHub.

## Tests

Offline unit tests cover keyword calculations, daily sentiment grouping, sentiment labels, and exchange-list parsing:

    py -3.12 -m unittest discover -s tests -v

This application is for academic and educational purposes only and does not constitute investment advice. A sentiment-derived Buy/Sell/Hold label is not a trade recommendation. Correlation does not imply causation.

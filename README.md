# AI Stock Market Sentiment Analyzer

A beginner-friendly Streamlit project for exploring Indian company news, headline sentiment, and historical prices. It includes a searchable NSE equity directory and an optional company research panel. Existing sentiment, stock return, chart, correlation, and summary features remain available.

## Run locally on Windows

Install Python 3.12, open PowerShell in this folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep PowerShell open while using the app. Streamlit prints a local address, usually `http://localhost:8501`.

## Data sources and coverage

- **Company directory:** the official NSE downloadable equity security list (`EQUITY_L.csv`). It is cached for six hours and the app displays its retrieval timestamp. Search supports company name, NSE symbol, and ISIN.
- **Coverage limitation:** the directory currently represents the NSE file only. BSE-exclusive companies are not included; a current stable, public BSE-wide machine-readable source could not be verified. NSE does not provide sector in this CSV, so sector is shown as unavailable. Duplicate NSE symbol/ISIN rows are removed.
- **Company profile and financial history:** Yahoo Finance through `yfinance` (a secondary aggregator). Company overview, ratios, and annual revenue/net income are shown only when supplied. Values can be missing or differ from audited exchange filings. The app links to NSE filing pages for verification.
- **Annual reports and announcements:** direct exchange filing pages are provided as starting points. The application does not yet download and summarize annual report PDFs, extract page references, or ingest the complete exchange announcement feed. Do not treat the research panel as a substitute for primary filings.
- **News:** Google News RSS search results, with publisher and original article links when supplied. The feed is an aggregator and can be incomplete. Sample rows are identified in the news table when the feed has no result or fails.
- **Prices:** Yahoo Finance through `yfinance`. The selected data source/exchange is shown in the price section.
- **Sentiment:** FinBERT is attempted first; VADER is the fallback. Sentiment describes headline text, not the truth or materiality of an event.
- **AI summary:** an OpenAI API key is optional. When configured, store it in `.streamlit/secrets.toml` for local use or Streamlit Community Cloud secrets. Never commit keys. The rest of the app works without one.

## Test the exchange-list module

Tests use a small synthetic CSV fixture and do not call an external service:

```powershell
py -3.12 -m unittest discover -s tests -v
```

## Deploy on Streamlit Community Cloud

The repository can be selected at [Streamlit Community Cloud](https://share.streamlit.io). Choose the repository, `main` branch, and `app.py`. A new commit to the deployed branch triggers a rebuild. A local code change does not update the public app until it has been pushed and the cloud build has succeeded.

> This application is developed for academic and educational purposes only and does not constitute investment advice. Correlation does not imply causation.

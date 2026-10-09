# AI Stock Market Sentiment Analyzer

A beginner-friendly Streamlit project for exploring Indian stock prices and financial-news headline sentiment. It is intended for classroom demonstrations and academic study.

## What the dashboard shows

- Recent headlines for a selected Indian company, with positive, neutral, or negative sentiment.
- An average sentiment score from -1 to +1 and a Bullish, Neutral, or Bearish label.
- Historical adjusted prices, daily returns, and sentiment charts.
- A same-day comparison of news sentiment and stock returns, including Pearson correlation.
- A short classroom summary. If the optional OpenAI service is unavailable, the dashboard creates a transparent summary from the displayed calculations instead.

## Run on Windows

Install Python 3.12, open PowerShell in this folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep PowerShell open while you use the dashboard. Streamlit prints a local address, usually `http://localhost:8501`.

## Deploy or update the public demo

This repository is connected to Streamlit Community Cloud. For a new deployment, sign in at [share.streamlit.io](https://share.streamlit.io), choose **Create app**, then select this repository, the `main` branch, and `app.py`. After deployment, commits to `main` are picked up by the app automatically.

Community Cloud may put an app to sleep after 12 hours without visits. Opening the public app wakes it again; its shared URL stays the same.

## Data and model notes

News comes from Google News RSS, with clearly marked sample headlines when the feed is unavailable. Historical prices come from Yahoo Finance through `yfinance`. FinBERT is attempted first; VADER is used if FinBERT cannot be loaded. The first FinBERT run may take longer because model files need to be downloaded. The AI-generated summary uses the OpenAI API when `OPENAI_API_KEY` is configured; otherwise a rule-based classroom summary is shown so the app remains usable without an API account.

For a local OpenAI summary, create `.streamlit/secrets.toml` and add:

```toml
OPENAI_API_KEY = "your-api-key"
```

Do not commit API keys to GitHub. No key is required for the rest of the dashboard.

> This application is developed for academic and educational purposes only and does not constitute investment advice. Correlation does not imply causation.

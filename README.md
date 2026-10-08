# AI Stock Market Sentiment Analyzer

A Streamlit academic project that studies the language of financial-news headlines alongside adjusted prices for selected NSE-listed companies. It is designed for a PGDM Finance classroom demonstration and uses free data sources with explicitly labeled demo fallbacks.

> This application is developed for academic and educational purposes only and does not constitute investment advice.

## Problem statement

Investors and students often see company news and price changes at the same time, but it can be difficult to summarize headline tone consistently. This project demonstrates a basic natural-language-processing workflow and compares daily headline sentiment with stock returns. The comparison is descriptive; it does not show that news caused a price movement.

## Objectives

- Retrieve recent company headlines from a free RSS source.
- Classify headlines as Positive, Neutral, or Negative and score them from −1 to +1.
- Retrieve adjusted historical prices and calculate daily returns.
- Summarize sentiment and compare daily sentiment with returns using Pearson correlation.
- Compare a selected group of up to five stocks.
- Demonstrate a transparent, long-only historical backtest for classroom discussion.

## Features

- Searchable selector for every symbol in NSE's downloadable NIFTY 500 list (501 symbols in the bundled snapshot), including RELIANCE, HDFCBANK, TCS, INFY, ICICIBANK, and SBIN. The original TATAMOTORS code remains as an alias for the renamed TMPV listing.
- Google News RSS collection with duplicate headline removal and fictional DEMO DATA if the feed is unavailable.
- FinBERT financial headline model, with VADER fallback if FinBERT cannot be loaded.
- Headline-level results, positive/neutral/negative counts and percentages, average score, and Bullish/Neutral/Bearish label.
- Adjusted price history, daily return, sentiment distribution and trend charts.
- Daily sentiment versus return comparison and Pearson correlation for dates with both measurements.
- AI Insight assembled from the model labels and calculations, with positive/negative headline examples and limitations. It does not call a separate text-generation API.
- Comparison table and charts for up to five selected companies.
- Educational historical backtest with strategy return, buy-and-hold return, trade count, winners, and losers.
- Offline synthetic prices and fictional headlines, both identified as DEMO DATA.

## Technology stack

Python, Streamlit, Pandas, Plotly, yfinance, Google News RSS, FinBERT through Transformers/PyTorch, and VADER as a lightweight fallback.

## AI and NLP methodology

The app sends each headline to ProsusAI/FinBERT when its model and dependencies are available. FinBERT estimates class probabilities for positive, neutral, and negative tone. If the model cannot be downloaded or loaded, the app applies VADER's general-purpose compound sentiment method instead. The dashboard identifies which model was used. The first FinBERT run downloads model files and may take longer; VADER needs no model download.

The model reads headlines, not full article text. News feeds can be incomplete, duplicate stories may remain if their headlines differ, and financial language can be ambiguous. A model label is a text classification, not a verified assessment of a company's fundamentals.

## Sentiment scoring methodology

For a FinBERT headline:

`headline score = probability(Positive) − probability(Negative)`

The score is bounded by −1 and +1. The predicted headline label is the class with the highest probability. For VADER fallback, the VADER compound score is used and its standard ±0.05 cutoffs assign Positive, Neutral, or Negative. VADER confidence is not reported.

For the stock-level sentiment score, the app takes the arithmetic mean of headline scores. It classifies the average as Bullish when it is at least +0.25, Bearish when it is at most −0.25, and Neutral otherwise. Label percentages use all analyzed headlines as the denominator.

## Stock return methodology

The selector uses NSE's published NIFTY 500 constituent CSV bundled with the project, so the app does not need to download the stock list at startup. This official snapshot was published April 22, 2026; replace the CSV when you want a newer constituent list. NSE describes NIFTY 500 as the top 500 eligible companies by full market capitalization and reports that it represented about 92% of NSE free-float market capitalization as of March 30, 2026 ([official index page](https://www.nseindia.com/static/products-services/indices-nifty500-index)). It is a broad index universe, not every security listed on NSE or BSE.

Prices are retrieved through yfinance using Yahoo Finance NSE symbols such as `TCS.NS`. The price module requests split/dividend-adjusted history. Daily return is:

`daily return (%) = (today's adjusted close / previous adjusted close − 1) × 100`

The first observation has no preceding close in the selected history and therefore has no daily return. If the live source fails, the dashboard uses deterministic synthetic DEMO DATA prices so charts can still be shown. Demo prices are not market observations and must not be used for conclusions.

## Correlation methodology

For each publication date, the app averages sentiment scores for the headlines dated that day. It pairs that daily average with the same date's stock daily return and calculates Pearson's correlation coefficient (`r`) on dates with both values. Dates without news and weekend news are not moved to another trading day. At least two matched dates and variation in both series are required. A small sample can make correlation unstable.

Pearson correlation describes the direction and strength of a linear relationship. **Correlation does not imply causation.**

## Educational backtesting methodology

The backtest uses daily average headline scores and adjusted daily returns for the selected period:

- Score above +0.25: BUY (enter or stay in a long position).
- Score below −0.25: SELL (exit to cash).
- Otherwise, including dates with no headlines: HOLD (keep the current position).

A signal observed at the day's close changes the position for the following close-to-close return. The strategy compounds the returns earned while long. Buy-and-hold compounds daily adjusted returns over the same price window. Each BUY entry is one trade; any open position at the final date is marked to market for its win/loss result. The model is long-only, assumes no transaction costs, taxes, or slippage, and does not model execution delays or liquidity. This is an educational illustration, not a trading recommendation.

> Past performance does not guarantee future results.

## Project structure

```text
ai_stock_market_sentiment_analyzer/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Runtime Python dependencies
├── README.md                      # Project, methodology, run, and deployment guide
├── setup_windows.bat              # Creates the local environment and installs packages
├── run_windows.bat                # Starts the dashboard
├── run_demo_windows.bat           # Starts offline DEMO DATA mode
├── .gitignore                     # Excludes secrets and generated/local files
├── .streamlit/
│   ├── config.toml                # Clean finance-style theme
│   └── secrets.toml.example       # Placeholder only; no key is currently required
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── news_fetcher.py        # RSS provider, normalization, deduplication, fallback
│   │   └── mock_news.py           # Fictional, dated demo headlines
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── sentiment.py           # FinBERT/VADER headline classification
│   │   ├── aggregation.py         # Counts, percentages, score, classification
│   │   ├── sentiment_return.py    # Daily alignment and Pearson correlation
│   │   └── backtest.py            # Educational historical strategy
│   └── market/
│       ├── __init__.py
│       ├── stock_data.py          # NIFTY 500 lookup and NSE adjusted prices/returns
│       ├── nifty500_constituents.csv # Official bundled NSE constituent snapshot
│       └── mock_stock_data.py     # Synthetic offline demo prices
└── tests/
    ├── test_analysis.py           # Calculation and fallback tests
    └── test_dashboard.py          # Streamlit demo-mode smoke test
```

## Run locally on Windows

Open Command Prompt in this project folder. Python 3.12 or newer is recommended.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `setup_windows.bat`, wait for installation to finish, then double-click `run_windows.bat`. To run the fully offline classroom demo, double-click `run_demo_windows.bat`; it uses VADER and clearly labeled DEMO DATA headlines and synthetic prices. To stop the server, press `Ctrl+C` in its terminal. To leave the virtual environment, type `deactivate`.

Run the automated checks from the activated environment with `python -m unittest discover -s tests -v`.

The first installation includes PyTorch and Transformers and may download several hundred megabytes. The first analysis may also download FinBERT. If the model cannot be loaded, VADER is used automatically. The app does not require API keys; the free news feed and price source may still be unavailable or rate-limited.

## Secrets and configuration

No API key, password, or token is required for the current sources. `.streamlit/secrets.toml.example` is a placeholder for future integrations. For a low-memory host, the optional setting `use_finbert = "false"` selects VADER instead of loading FinBERT; this is a configuration choice, not a secret. Alternatively, set `AI_ANALYZER_USE_FINBERT=false` in the local environment. If you later add a provider key, copy the example to `.streamlit/secrets.toml`, put the key in that local file, and keep it out of Git. `.gitignore` already excludes the real secrets file. For Community Cloud, paste `use_finbert = "false"` into the app's Secrets setting only if FinBERT exceeds the app's memory; otherwise leave Secrets empty.

## GitHub and Streamlit Community Cloud deployment

1. Create an empty GitHub repository for the project. In a terminal opened in this project folder, run:

   ```bash
   git init
   git branch -M main
   git add .
   git status --short
   git commit -m "Prepare AI stock sentiment analyzer"
   git remote add origin https://github.com/<your-username>/<your-repository>.git
   git push -u origin main
   ```

   Review `git status --short` before committing. Upload this folder's contents so `app.py`, `requirements.txt`, and `.streamlit/config.toml` are at the repository root. Do not upload `.venv`, `.streamlit/secrets.toml`, caches, or local data.
2. Sign in to Streamlit Community Cloud at [share.streamlit.io](https://share.streamlit.io/) using GitHub and choose **Create app**. See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) if the screens change.
3. Select your repository, branch (usually `main`), and entrypoint file `app.py`. Choose a subdomain if you want a memorable address.
4. In **Advanced settings**, select the same Python version you tested locally. Paste no secrets; this project currently requires none. If the first model load exceeds available memory, set `use_finbert = "false"` in the Secrets field to use VADER and redeploy.
5. After the build completes, open the generated URL and test a stock. The expected URL format is `https://<your-chosen-subdomain>.streamlit.app`.

The free RSS and Yahoo Finance/yfinance sources can change, throttle requests, or be unavailable. The app labels demo fallbacks so they are not confused with live observations. Streamlit Community Cloud builds from the GitHub repository and installs the packages listed in `requirements.txt`. The app entrypoint, requirements, and config are at the repository root as described in Streamlit's official [file organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Limitations and future scope

- Headlines are not full articles; RSS coverage, publication dates, and source metadata vary.
- Correlation is sensitive to sample size and does not prove a causal relationship.
- FinBERT/VADER can misread sarcasm, context, company-specific terms, or market jargon.
- The educational backtest excludes fees, taxes, slippage, and realistic order execution.
- Synthetic DEMO DATA is for interface demonstrations only.
- Future scope could include a licensed financial-news source, more historical coverage, event-time alignment, sector benchmarks, risk-adjusted backtest metrics, and a separately configured generative-summary provider.

## Educational disclaimer

This application is developed for academic and educational purposes only and does not constitute investment advice. Past performance does not guarantee future results.

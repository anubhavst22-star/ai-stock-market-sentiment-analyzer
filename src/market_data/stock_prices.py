"""Download historical Indian share prices and calculate daily returns.

This module uses Yahoo Finance through the free ``yfinance`` Python package.
It is kept separate from the news and sentiment modules.
"""

from typing import Final

import pandas as pd
import yfinance as yf


# Starter choices for the future dropdown. The fetch function also accepts
# other Yahoo Finance symbols, so this list does not limit the data provider.
INDIAN_STOCKS: Final[dict[str, str]] = {
    "RELIANCE": "Reliance Industries",
    "HDFCBANK": "HDFC Bank",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "TATAMOTORS": "Tata Motors",
}


class MarketDataError(Exception):
    """A clear error raised when stock price data cannot be retrieved."""


def fetch_stock_data(
    symbol: str,
    exchange: str = "NSE",
    period: str = "1y",
) -> pd.DataFrame:
    """Get historical daily prices and add a ``Daily Return`` column.

    ``symbol`` may be a bare Indian symbol such as ``RELIANCE`` or a Yahoo
    symbol with a suffix such as ``RELIANCE.NS`` / ``RELIANCE.BO``. For a bare
    symbol, ``exchange`` decides which suffix to use. Daily Return is a decimal
    fraction: for example, 0.02 means a 2% gain from the previous trading day.
    """
    cleaned_symbol = symbol.strip().upper()
    if not cleaned_symbol:
        raise ValueError("Please provide a stock symbol, such as RELIANCE.")

    cleaned_exchange = exchange.strip().upper()
    if cleaned_exchange not in {"NSE", "BSE"}:
        raise ValueError("Exchange must be either 'NSE' or 'BSE'.")

    if cleaned_symbol.endswith(".NS"):
        yahoo_symbol = cleaned_symbol
        cleaned_exchange = "NSE"
    elif cleaned_symbol.endswith(".BO"):
        yahoo_symbol = cleaned_symbol
        cleaned_exchange = "BSE"
    else:
        suffix = ".NS" if cleaned_exchange == "NSE" else ".BO"
        yahoo_symbol = f"{cleaned_symbol}{suffix}"

    try:
        # auto_adjust=True adjusts historical prices for corporate actions.
        prices = yf.Ticker(yahoo_symbol).history(period=period, auto_adjust=True)
    except Exception as error:
        raise MarketDataError(
            f"Could not retrieve price data for {yahoo_symbol}. "
            "Check your internet connection, stock symbol, and selected period."
        ) from error

    if prices is None or prices.empty:
        raise MarketDataError(
            f"No price history was returned for {yahoo_symbol}. "
            "Check that the symbol is listed and try another period."
        )

    if "Close" not in prices.columns:
        raise MarketDataError(f"Price data for {yahoo_symbol} does not include a closing price.")

    prices = prices.copy()
    prices["Daily Return"] = prices["Close"].pct_change()
    prices.attrs["symbol"] = yahoo_symbol
    prices.attrs["exchange"] = cleaned_exchange
    prices.attrs["currency"] = "INR"
    return prices


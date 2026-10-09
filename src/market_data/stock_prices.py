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

# Yahoo Finance identifies many BSE listings by their six-digit BSE scrip code.
# These codes are the BSE equivalents of the beginner-friendly symbols above.
BSE_SCRIP_CODES: Final[dict[str, str]] = {
    "RELIANCE": "500325",
    "HDFCBANK": "500180",
    "TCS": "532540",
    "INFY": "500209",
    "ICICIBANK": "532174",
    "SBIN": "500112",
    "TATAMOTORS": "500570",
}


class MarketDataError(Exception):
    """A clear error raised when stock price data cannot be retrieved."""


def fetch_stock_data(
    symbol: str,
    exchange: str = "NSE",
    period: str = "1y",
    start_date: str | None = None,
    end_date: str | None = None,
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

    requested_exchange = exchange.strip().upper()
    cleaned_exchange = requested_exchange
    if cleaned_exchange not in {"NSE", "BSE"}:
        raise ValueError("Exchange must be either 'NSE' or 'BSE'.")

    if cleaned_symbol.endswith(".NS"):
        yahoo_symbol = cleaned_symbol
        cleaned_exchange = "NSE"
    elif cleaned_symbol.endswith(".BO"):
        bse_symbol = cleaned_symbol[:-3]
        bse_symbol = BSE_SCRIP_CODES.get(bse_symbol, bse_symbol)
        yahoo_symbol = f"{bse_symbol}.BO"
        cleaned_exchange = "BSE"
    else:
        if cleaned_exchange == "BSE":
            cleaned_symbol = BSE_SCRIP_CODES.get(cleaned_symbol, cleaned_symbol)
        suffix = ".NS" if cleaned_exchange == "NSE" else ".BO"
        yahoo_symbol = f"{cleaned_symbol}{suffix}"

    symbols_to_try = [(yahoo_symbol, cleaned_exchange)]
    if requested_exchange == "BSE" and not symbol.strip().upper().endswith(".NS"):
        # Yahoo Finance sometimes has a live BSE quote but no BSE history. Since
        # these companies also trade on NSE, use NSE history as a disclosed backup.
        bse_to_nse_symbol = {code: name for name, code in BSE_SCRIP_CODES.items()}
        nse_symbol = cleaned_symbol.removesuffix(".BO")
        nse_symbol = bse_to_nse_symbol.get(nse_symbol, nse_symbol)
        symbols_to_try.append((f"{nse_symbol}.NS", "NSE"))

    prices = None
    used_symbol = yahoo_symbol
    actual_exchange = cleaned_exchange
    last_error = None
    for candidate_symbol, candidate_exchange in symbols_to_try:
        try:
            # auto_adjust=True adjusts historical prices for corporate actions.
            ticker = yf.Ticker(candidate_symbol)
            if start_date or end_date:
                candidate_prices = ticker.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                )
            else:
                candidate_prices = ticker.history(period=period, auto_adjust=True)
        except Exception as error:
            last_error = error
            continue

        # A single quote is not enough to calculate a daily return. If BSE
        # history is sparse, the loop can use the NSE history backup instead.
        if (
            candidate_prices is None
            or len(candidate_prices) < 2
            or "Close" not in candidate_prices.columns
        ):
            continue

        prices = candidate_prices
        used_symbol = candidate_symbol
        actual_exchange = candidate_exchange
        break

    if prices is None:
        if last_error is not None:
            raise MarketDataError(
                f"Could not retrieve enough price history for {yahoo_symbol}. "
                "Check your internet connection, stock symbol, and selected period."
            ) from last_error
        raise MarketDataError(
            f"No usable price history was returned for {yahoo_symbol}. "
            "At least two trading days with closing prices are needed."
        )

    prices = prices.copy()
    prices["Daily Return"] = prices["Close"].pct_change()
    prices.attrs["symbol"] = used_symbol
    prices.attrs["exchange"] = actual_exchange
    prices.attrs["requested_exchange"] = requested_exchange
    prices.attrs["currency"] = "INR"
    return prices


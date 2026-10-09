"""Optional company fundamentals via Yahoo Finance, kept separate from prices."""

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf


def fetch_company_research(symbol: str, exchange: str = "NSE") -> dict[str, Any]:
    """Return available profile and annual income-statement data.

    Yahoo Finance is a secondary aggregator, not an exchange filing source.
    Missing fields remain None so the UI can identify gaps instead of guessing.
    """
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("A company ticker is required.")
    suffix = ".BO" if exchange.upper() == "BSE" else ".NS"
    ticker = yf.Ticker(cleaned if cleaned.endswith((".NS", ".BO")) else cleaned + suffix)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    try:
        statements = ticker.income_stmt
    except Exception:
        statements = pd.DataFrame()

    def pick(*keys: str) -> Any:
        for key in keys:
            value = info.get(key)
            if value not in (None, ""):
                return value
        return None

    annual_rows: list[dict[str, Any]] = []
    if isinstance(statements, pd.DataFrame) and not statements.empty:
        revenue_row = next((key for key in ("Total Revenue", "Operating Revenue") if key in statements.index), None)
        profit_row = next((key for key in ("Net Income", "Net Income Common Stockholders") if key in statements.index), None)
        for period in list(statements.columns)[:5]:
            year = pd.Timestamp(period).strftime("%Y-%m-%d")
            annual_rows.append({
                "Period ended": year,
                "Revenue": _number(statements.loc[revenue_row, period]) if revenue_row else None,
                "Net income": _number(statements.loc[profit_row, period]) if profit_row else None,
            })

    return {
        "symbol": cleaned,
        "company_name": pick("longName", "shortName"),
        "business_summary": pick("longBusinessSummary"),
        "sector": pick("sector"),
        "industry": pick("industry"),
        "financial_currency": pick("financialCurrency", "currency"),
        "website": pick("website"),
        "market_cap": pick("marketCap"),
        "trailing_pe": pick("trailingPE"),
        "price_to_book": pick("priceToBook"),
        "return_on_equity": pick("returnOnEquity"),
        "profit_margin": pick("profitMargins"),
        "annual_performance": annual_rows,
        "provider": "Yahoo Finance via yfinance (secondary data aggregator)",
        "source_url": f"https://finance.yahoo.com/quote/{ticker.ticker}/",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limitations": "Financial fields can be missing, delayed, or differently classified from audited filings. Verify material figures against exchange/company filings.",
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None

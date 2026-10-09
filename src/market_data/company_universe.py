"""Load and search the exchange's official NSE equity security list.

The downloadable file is published by NSE and refreshed by the exchange. It
does not provide sector classifications or the BSE-exclusive universe.
"""

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import pandas as pd
import requests

NSE_EQUITY_CSV = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


class CompanyUniverseError(RuntimeError):
    """Raised when the official exchange security list cannot be loaded."""


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    normalized = {str(col).strip().upper(): col for col in frame.columns}
    for name in names:
        if name.upper() in normalized:
            return normalized[name.upper()]
    return None


def normalize_nse_equity_csv(content: bytes, updated_at: str | None = None) -> pd.DataFrame:
    """Parse NSE's equity CSV into a consistent company table."""
    try:
        frame = pd.read_csv(BytesIO(content), dtype=str, encoding="utf-8-sig")
    except Exception as error:
        raise CompanyUniverseError("The NSE security file could not be parsed.") from error
    symbol = _column(frame, "SYMBOL", "Symbol")
    name = _column(frame, "NAME OF COMPANY", "Company Name", "NAME")
    isin = _column(frame, "ISIN NUMBER", "ISIN", "ISIN CODE")
    series = _column(frame, "SERIES")
    listing_date = _column(frame, "DATE OF LISTING", "LISTING DATE")
    if not symbol or not name:
        raise CompanyUniverseError("The NSE security file has an unexpected column layout.")
    result = pd.DataFrame({
        "symbol": frame[symbol].fillna("").str.strip().str.upper(),
        "company_name": frame[name].fillna("").str.strip(),
        "isin": frame[isin].fillna("").str.strip().str.upper() if isin else "",
        "series": frame[series].fillna("").str.strip().str.upper() if series else "",
        "listing_date": frame[listing_date].fillna("").str.strip() if listing_date else "",
        "exchange": "NSE",
        "sector": "Not provided by exchange",
    })
    # EQ is the ordinary equity series; SME series can be equity too, so retain
    # them. Exclude blank/invalid rows but do not invent eligibility criteria.
    result = result[(result.symbol != "") & (result.company_name != "")]
    result = result.drop_duplicates(subset=["isin", "symbol"], keep="first")
    result.attrs["source_url"] = NSE_EQUITY_CSV
    result.attrs["updated_at"] = updated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result.reset_index(drop=True)


def fetch_nse_equity_universe() -> tuple[pd.DataFrame, str]:
    """Fetch and parse current NSE listed equity securities."""
    try:
        response = requests.get(
            NSE_EQUITY_CSV,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AcademicFinanceDashboard/1.0)"},
            timeout=20,
        )
        response.raise_for_status()
        updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        frame = normalize_nse_equity_csv(response.content, updated_at)
    except Exception as error:
        if isinstance(error, CompanyUniverseError):
            raise
        raise CompanyUniverseError(f"Could not retrieve the NSE company list: {error}") from error
    return frame, updated_at


def search_companies(
    companies: pd.DataFrame,
    query: str = "",
    exchange: str = "All",
    sector: str = "All",
) -> pd.DataFrame:
    """Filter companies by name, ticker, ISIN, exchange, and sector."""
    result = companies.copy()
    if exchange != "All" and "exchange" in result:
        result = result[result.exchange.str.casefold() == exchange.casefold()]
    if sector != "All" and "sector" in result:
        result = result[result.sector.fillna("Unknown").str.casefold() == sector.casefold()]
    term = query.strip().casefold()
    if term:
        mask = pd.Series(False, index=result.index)
        for column in ("company_name", "symbol", "isin"):
            if column in result:
                mask |= result[column].fillna("").astype(str).str.casefold().str.contains(term, regex=False)
        result = result[mask]
    return result.reset_index(drop=True)

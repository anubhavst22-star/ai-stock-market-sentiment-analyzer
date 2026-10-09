"""Load and search NSE's official combined listed-equity security universe.

NSE publishes a daily MII security master for NSE-listed and BSE-exclusive
securities. The older NSE equity CSV is used only as a disclosed fallback.
"""

from datetime import date, datetime, timedelta, timezone
import gzip
from io import BytesIO

import pandas as pd
import requests

NSE_REPORTS_PAGE = "https://www.nseindia.com/all-reports"
NSE_EQUITY_CSV = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_SECURITY_MASTER_PATTERN = (
    "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_{date}.csv.gz"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AcademicFinanceDashboard/1.0)",
    "Accept": "text/csv,application/gzip,application/octet-stream,*/*",
}
EQUITY_SERIES = {
    "EQ", "BE", "BZ", "SM", "ST", "SZ", "A", "B", "B1", "B2",
    "M", "MT", "S", "T", "X", "XT", "Z",
}


class CompanyUniverseError(RuntimeError):
    """Raised when the official exchange security list cannot be loaded."""


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    normalized = {str(col).strip().upper(): col for col in frame.columns}
    return next((normalized[name.upper()] for name in names if name.upper() in normalized), None)


def _clean(frame: pd.DataFrame, *, source_url: str, updated_at: str) -> pd.DataFrame:
    """Normalize known NSE and BSE security master headers."""
    symbol = _column(frame, "SYMBOL", "TCKRSYMB", "INSTRUMENT CODE")
    name = _column(frame, "NAME OF COMPANY", "COMPANY NAME", "NAME", "SCTYNM", "SCRIP NAME")
    isin = _column(frame, "ISIN NUMBER", "ISIN", "ISIN CODE")
    series = _column(frame, "SERIES", "SCTYSRS", "GROUP NAME")
    listing_date = _column(frame, "DATE OF LISTING", "LISTING DATE", "LISTGDT")
    permitted = _column(frame, "PRTDTOTRAD", "PERMITTED TO TRADE")
    if not symbol or not name:
        raise CompanyUniverseError("The exchange security file has an unexpected column layout.")

    symbols = frame[symbol].fillna("").astype(str).str.strip().str.upper()
    names = frame[name].fillna("").astype(str).str.strip()
    isin_values = frame[isin].fillna("").astype(str).str.strip().str.upper() if isin else pd.Series("", index=frame.index)
    series_values = frame[series].fillna("").astype(str).str.strip().str.upper() if series else pd.Series("", index=frame.index)
    # INE-prefixed ISINs identify Indian-issued equity shares. This avoids
    # presenting debt, derivatives, mutual funds, and most non-company products
    # as listed companies.
    mask = symbols.ne("") & names.ne("") & isin_values.str.match(r"^INE[A-Z0-9]{9}$")
    if series:
        mask &= series_values.isin(EQUITY_SERIES)

    exchange_values = pd.Series("NSE", index=frame.index)
    if permitted:
        is_bse_only = frame[permitted].fillna("").astype(str).str.strip().eq("2")
        exchange_values.loc[is_bse_only] = "BSE exclusive"
    # NSE marks BSE-exclusive symbols with a trailing dollar sign in this file.
    # Retain the marker in the displayed ticker so it cannot collide with an
    # NSE ticker of the same base name.
    is_bse_only = symbols.str.endswith("$")
    exchange_values.loc[is_bse_only] = "BSE exclusive"

    result = pd.DataFrame(
        {
            "symbol": symbols,
            "company_name": names,
            "isin": isin_values,
            "series": series_values,
            "listing_date": frame[listing_date].fillna("").astype(str).str.strip() if listing_date else "",
            "exchange": exchange_values,
            "sector": "Not provided by exchange",
        }
    )
    result = result.loc[mask].drop_duplicates(subset=["isin"], keep="first").reset_index(drop=True)
    result.attrs["source_url"] = source_url
    result.attrs["updated_at"] = updated_at
    return result


def normalize_nse_equity_csv(content: bytes, updated_at: str | None = None) -> pd.DataFrame:
    """Parse NSE's standard equity-list CSV (fallback and offline-test helper)."""
    try:
        frame = pd.read_csv(BytesIO(content), dtype=str, encoding="utf-8-sig")
    except Exception as error:
        raise CompanyUniverseError("The NSE security file could not be parsed.") from error
    symbol = _column(frame, "SYMBOL")
    name = _column(frame, "NAME OF COMPANY")
    isin = _column(frame, "ISIN NUMBER")
    series = _column(frame, "SERIES")
    listing_date = _column(frame, "DATE OF LISTING")
    if not symbol or not name:
        raise CompanyUniverseError("The NSE equity file has an unexpected column layout.")
    normalized = pd.DataFrame(
        {
            "SYMBOL": frame[symbol],
            "NAME OF COMPANY": frame[name],
            "ISIN NUMBER": frame[isin] if isin else "",
            "SERIES": frame[series] if series else "",
            "DATE OF LISTING": frame[listing_date] if listing_date else "",
        }
    )
    stamp = updated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return _clean(normalized, source_url=NSE_EQUITY_CSV, updated_at=stamp)


def normalize_nse_security_master(content: bytes, updated_at: str | None = None) -> pd.DataFrame:
    """Parse the daily NSE-listed + BSE-exclusive compressed security master."""
    try:
        raw = gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content
        frame = pd.read_csv(BytesIO(raw), dtype=str, encoding="utf-8-sig")
    except Exception as error:
        raise CompanyUniverseError("The daily combined security file could not be decompressed or parsed.") from error
    stamp = updated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return _clean(frame, source_url=NSE_REPORTS_PAGE, updated_at=stamp)


def _recent_security_master_urls(today: date | None = None) -> list[str]:
    """Return recent weekday archives in newest-first order (holidays are skipped by 404)."""
    current = today or date.today()
    return [
        NSE_SECURITY_MASTER_PATTERN.format(date=(current - timedelta(days=offset)).strftime("%d%m%Y"))
        for offset in range(8)
        if (current - timedelta(days=offset)).weekday() < 5
    ]


def fetch_nse_equity_universe() -> tuple[pd.DataFrame, str]:
    """Fetch current NSE and BSE-exclusive listed equity companies."""
    errors: list[str] = []
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for url in _recent_security_master_urls():
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            companies = normalize_nse_security_master(response.content, stamp)
            if not companies.empty:
                return companies, stamp
            errors.append(f"{url}: parsed file contained no eligible equity rows")
        except Exception as error:
            errors.append(f"{url}: {error}")

    # The NSE-listed-only CSV preserves useful coverage when the daily combined
    # archive is temporarily unavailable. The UI shows that this fallback ran.
    try:
        response = requests.get(NSE_EQUITY_CSV, headers=HEADERS, timeout=20)
        response.raise_for_status()
        companies = normalize_nse_equity_csv(response.content, stamp)
        companies.attrs["fallback"] = True
        return companies, stamp
    except Exception as error:
        errors.append(f"{NSE_EQUITY_CSV}: {error}")
    raise CompanyUniverseError("Could not retrieve the official company list. " + " | ".join(errors[-3:]))


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

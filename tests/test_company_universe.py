"""Offline tests for parsing and searching the exchange company list."""

import unittest

from src.market_data.company_universe import normalize_nse_equity_csv, search_companies


class CompanyUniverseTests(unittest.TestCase):
    def setUp(self):
        self.csv = (
            "SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,ISIN NUMBER\n"
            "ALPHA,Alpha Industries Limited,EQ,01-Jan-2000,INE000A01000\n"
            "BETA,Beta Finance Limited,SM,02-Feb-2001,INE000B01000\n"
            "ALPHA,Alpha Industries Limited,EQ,01-Jan-2000,INE000A01000\n"
        ).encode("utf-8")

    def test_parse_and_deduplicate_official_rows(self):
        companies = normalize_nse_equity_csv(self.csv, "2026-10-09T10:00:00+00:00")
        self.assertEqual(len(companies), 2)
        self.assertEqual(companies.iloc[0]["isin"], "INE000A01000")
        self.assertEqual(companies.attrs["updated_at"], "2026-10-09T10:00:00+00:00")

    def test_search_by_company_symbol_or_isin(self):
        companies = normalize_nse_equity_csv(self.csv)
        self.assertEqual(search_companies(companies, "alpha").iloc[0]["symbol"], "ALPHA")
        self.assertEqual(search_companies(companies, "BETA").iloc[0]["company_name"], "Beta Finance Limited")
        self.assertEqual(search_companies(companies, "INE000B01000").iloc[0]["symbol"], "BETA")

    def test_empty_query_and_exchange_filter(self):
        companies = normalize_nse_equity_csv(self.csv)
        self.assertEqual(len(search_companies(companies, exchange="NSE")), 2)
        self.assertEqual(len(search_companies(companies, exchange="BSE")), 0)


if __name__ == "__main__":
    unittest.main()

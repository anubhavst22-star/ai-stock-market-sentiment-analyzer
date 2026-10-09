"""Unit tests for the terminal's local chart and summary helpers."""

import unittest

from src.analysis.terminal_features import (
    daily_headline_sentiment,
    local_deep_dive,
    sentiment_label_to_status,
    top_headline_keywords,
)


class TerminalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.articles = [
            {
                "headline": "Company reports strong earnings growth",
                "publication_date": "2026-10-08",
                "source": "Example Publisher",
                "sentiment": "Positive",
                "sentiment_score": 0.8,
            },
            {
                "headline": "Company faces debt risk after weak results",
                "publication_date": "2026-10-08",
                "source": "Example Publisher",
                "sentiment": "Negative",
                "sentiment_score": -0.6,
            },
        ]

    def test_keyword_scores_and_counts(self):
        keywords = top_headline_keywords(self.articles, limit=10)
        company = keywords[keywords["Keyword"] == "company"].iloc[0]
        self.assertEqual(company["Headline count"], 2)
        self.assertAlmostEqual(company["Average sentiment"], 0.1)

    def test_daily_sentiment_averages_same_day_and_omits_mock_data(self):
        articles = self.articles + [
            {"headline": "Mock positive", "publication_date": "2026-10-08",
             "source": "Sample Data", "sentiment_score": 1.0}
        ]
        daily = daily_headline_sentiment(articles)
        self.assertEqual(len(daily), 1)
        self.assertAlmostEqual(float(daily.iloc[0]), 0.1)

    def test_non_advisory_status_labels_and_local_deep_dive(self):
        self.assertEqual(sentiment_label_to_status("Bullish"), "Buy (sentiment)")
        result = local_deep_dive({
            "company_name": "Example Co",
            "summary": {"classification": "Neutral", "average_sentiment_score": 0.1, "article_count": 2},
            "articles": self.articles,
        })
        self.assertIn("not a forecast", result["executive_summary"])
        self.assertEqual(len(result["risks"]), 1)
        self.assertEqual(len(result["catalysts"]), 1)


if __name__ == "__main__":
    unittest.main()

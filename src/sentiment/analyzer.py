"""Financial headline sentiment analysis using FinBERT when available.

FinBERT is trained for financial text. If its model files cannot be downloaded
or loaded, this module uses the VADER lexicon already included in requirements.
"""

from functools import lru_cache
from typing import Any


FINBERT_MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def _load_finbert() -> tuple[Any, Any] | None:
    """Load FinBERT once and reuse it for later headlines.

    The first use downloads the model from Hugging Face if it is not cached.
    Returning None lets the caller use VADER if loading fails.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception:
        # This can happen if the computer is offline or model setup fails.
        return None


def analyze_headline(headline: str) -> dict[str, str | float | None]:
    """Classify one headline and return its label, score, and confidence.

    The sentiment score is positive probability minus negative probability,
    which always produces a value from -1 (negative) to +1 (positive).
    """
    cleaned_headline = headline.strip()
    if not cleaned_headline:
        raise ValueError("Please provide a headline to analyze.")

    finbert = _load_finbert()
    if finbert is not None:
        try:
            return _analyze_with_finbert(cleaned_headline, *finbert)
        except Exception:
            # Keep the module usable if FinBERT inference fails after loading.
            pass

    return _analyze_with_vader(cleaned_headline)


def analyze_news(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add sentiment results to article dictionaries from the news collector.

    Original article fields such as publication date, source, and URL are kept.
    """
    analyzed_articles = []
    for article in articles:
        result = analyze_headline(str(article.get("headline", "")))
        analyzed_articles.append({**article, **result})
    return analyzed_articles


def _analyze_with_finbert(headline: str, tokenizer: Any, model: Any) -> dict[str, str | float | None]:
    """Run FinBERT and turn its three probabilities into the project output."""
    import torch

    encoded = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1).tolist()

    # Use the model's label map instead of assuming label numbers have an order.
    label_map = model.config.id2label
    scores = {
        str(label_map.get(index, label_map.get(str(index), f"label_{index}"))).lower(): float(probability)
        for index, probability in enumerate(probabilities)
    }
    positive_score = scores.get("positive", 0.0)
    negative_score = scores.get("negative", 0.0)
    neutral_score = scores.get("neutral", 0.0)
    sentiment = max(
        (("Positive", positive_score), ("Neutral", neutral_score), ("Negative", negative_score)),
        key=lambda item: item[1],
    )[0]

    return {
        "headline": headline,
        "sentiment": sentiment,
        "sentiment_score": positive_score - negative_score,
        "confidence": max(scores.values()),
        "model_used": "FinBERT",
    }


def _analyze_with_vader(headline: str) -> dict[str, str | float | None]:
    """Use VADER as a small fallback when FinBERT is unavailable."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    compound_score = SentimentIntensityAnalyzer().polarity_scores(headline)["compound"]
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "headline": headline,
        "sentiment": sentiment,
        "sentiment_score": float(compound_score),
        # VADER does not provide a model confidence probability.
        "confidence": None,
        "model_used": "VADER fallback",
    }


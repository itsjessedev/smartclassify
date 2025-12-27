import re
import time
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

from .models import (
    Category, SentimentLabel, ClassificationRequest, ClassificationResponse,
    CategoryPrediction, SentimentPrediction
)


# Training data for category classification
CATEGORY_TRAINING_DATA = [
    # Technology
    ("The new smartphone features a faster processor and improved camera", "technology"),
    ("Software developers are adopting artificial intelligence in their workflows", "technology"),
    ("Cloud computing services have revolutionized business infrastructure", "technology"),
    ("The latest programming language update includes new features", "technology"),
    ("Cybersecurity threats are increasing in the digital age", "technology"),
    ("Machine learning algorithms are becoming more sophisticated", "technology"),
    ("The tech startup raised millions in venture capital funding", "technology"),
    ("New gadget launches have excited consumers worldwide", "technology"),

    # Business
    ("The stock market showed gains after the quarterly report", "business"),
    ("Companies are investing heavily in digital transformation", "business"),
    ("The merger between two corporations was announced today", "business"),
    ("Retail sales exceeded expectations this quarter", "business"),
    ("Entrepreneurs are launching startups in emerging markets", "business"),
    ("The CEO announced a new strategic initiative for growth", "business"),
    ("Economic indicators suggest a recovery is underway", "business"),
    ("Corporate profits have risen despite challenging conditions", "business"),

    # Sports
    ("The team won the championship after an incredible season", "sports"),
    ("The athlete broke the world record in the competition", "sports"),
    ("Football fans celebrated their team's victory", "sports"),
    ("The basketball player scored the winning points", "sports"),
    ("Tennis tournament results surprised many spectators", "sports"),
    ("The Olympics attracted athletes from around the world", "sports"),
    ("Soccer match ended in a dramatic penalty shootout", "sports"),
    ("The coach announced changes to the team lineup", "sports"),

    # Entertainment
    ("The movie premiered to critical acclaim this weekend", "entertainment"),
    ("The singer released a new album that topped the charts", "entertainment"),
    ("The streaming service announced a new original series", "entertainment"),
    ("Award show honored the best in film and television", "entertainment"),
    ("The concert tour sold out in minutes", "entertainment"),
    ("Celebrity news dominated social media discussions", "entertainment"),
    ("The film festival featured independent productions", "entertainment"),
    ("Video game sales continue to break records", "entertainment"),

    # Health
    ("New research shows benefits of regular exercise", "health"),
    ("The vaccine received approval from health authorities", "health"),
    ("Doctors recommend preventive health screenings", "health"),
    ("Mental health awareness campaigns are gaining support", "health"),
    ("Nutrition studies reveal importance of balanced diet", "health"),
    ("Medical breakthrough offers hope for patients", "health"),
    ("Healthcare workers continue to serve communities", "health"),
    ("Wellness programs are becoming popular in workplaces", "health"),

    # Science
    ("Scientists discovered a new species in the rainforest", "science"),
    ("Space mission successfully landed on the asteroid", "science"),
    ("Climate research reveals alarming trends", "science"),
    ("Physics experiment confirmed theoretical predictions", "science"),
    ("Researchers published findings in scientific journal", "science"),
    ("DNA analysis provided insights into evolution", "science"),
    ("Astronomers detected signals from distant galaxy", "science"),
    ("Environmental study assessed ecosystem health", "science"),

    # Politics
    ("The president signed the new legislation into law", "politics"),
    ("Election results showed a shift in voter preferences", "politics"),
    ("Congress debated the proposed budget allocation", "politics"),
    ("Diplomatic negotiations continue between nations", "politics"),
    ("Political parties announced their policy platforms", "politics"),
    ("Government officials addressed public concerns", "politics"),
    ("International summit focused on global challenges", "politics"),
    ("Voting reforms were proposed by legislators", "politics"),
]

# Sentiment training data
SENTIMENT_TRAINING_DATA = [
    # Positive
    ("This is absolutely wonderful and amazing", "positive"),
    ("I love how great this product is", "positive"),
    ("Excellent service and fantastic experience", "positive"),
    ("The results exceeded all expectations", "positive"),
    ("I'm so happy with this purchase", "positive"),
    ("Outstanding quality and brilliant design", "positive"),
    ("This made my day so much better", "positive"),
    ("Incredible value and superb performance", "positive"),

    # Negative
    ("This is terrible and disappointing", "negative"),
    ("I hate how awful this experience was", "negative"),
    ("Poor quality and horrible service", "negative"),
    ("The results were completely unacceptable", "negative"),
    ("I'm very frustrated and angry", "negative"),
    ("Worst purchase I've ever made", "negative"),
    ("This ruined my entire day", "negative"),
    ("Dreadful experience and waste of money", "negative"),

    # Neutral
    ("The product arrived as described", "neutral"),
    ("It works as expected nothing special", "neutral"),
    ("Average quality meets basic needs", "neutral"),
    ("Standard service nothing remarkable", "neutral"),
    ("It does what it's supposed to do", "neutral"),
    ("Neither good nor bad just okay", "neutral"),
    ("Reasonable quality for the price", "neutral"),
    ("Typical experience nothing noteworthy", "neutral"),
]


class TextClassifier:
    """Multi-task text classifier for categories and sentiment."""

    def __init__(self):
        self.category_model = None
        self.sentiment_model = None
        self.category_classes = [c.value for c in Category]
        self.sentiment_classes = [s.value for s in SentimentLabel]
        self._trained = False

    def train(self):
        """Train both classification models."""
        # Train category classifier
        cat_texts = [t[0] for t in CATEGORY_TRAINING_DATA]
        cat_labels = [t[1] for t in CATEGORY_TRAINING_DATA]

        self.category_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=1000,
                stop_words='english'
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self.category_model.fit(cat_texts, cat_labels)

        # Train sentiment classifier
        sent_texts = [t[0] for t in SENTIMENT_TRAINING_DATA]
        sent_labels = [t[1] for t in SENTIMENT_TRAINING_DATA]

        self.sentiment_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=500,
                stop_words='english'
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self.sentiment_model.fit(sent_texts, sent_labels)

        self._trained = True

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction using word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'has', 'have', 'will', 'your', 'one', 'was', 'with',
            'this', 'that', 'from', 'they', 'been', 'would', 'their'
        }
        words = [w for w in words if w not in stopwords]
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_k)]

    def classify_category(self, text: str) -> CategoryPrediction:
        """Classify text into a category."""
        proba = self.category_model.predict_proba([text])[0]
        classes = self.category_model.classes_

        max_idx = np.argmax(proba)
        predicted_category = classes[max_idx]
        confidence = float(proba[max_idx])

        all_scores = {cat: float(prob) for cat, prob in zip(classes, proba)}

        return CategoryPrediction(
            category=Category(predicted_category),
            confidence=round(confidence, 4),
            all_scores={k: round(v, 4) for k, v in all_scores.items()}
        )

    def classify_sentiment(self, text: str) -> SentimentPrediction:
        """Classify sentiment of text."""
        proba = self.sentiment_model.predict_proba([text])[0]
        classes = self.sentiment_model.classes_

        max_idx = np.argmax(proba)
        predicted_sentiment = classes[max_idx]
        confidence = float(proba[max_idx])

        scores = {sent: float(prob) for sent, prob in zip(classes, proba)}

        return SentimentPrediction(
            sentiment=SentimentLabel(predicted_sentiment),
            confidence=round(confidence, 4),
            scores={k: round(v, 4) for k, v in scores.items()}
        )

    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """Full classification pipeline."""
        start_time = time.time()

        category = self.classify_category(request.text)
        sentiment = self.classify_sentiment(request.text)
        keywords = self.extract_keywords(request.text)

        if not request.include_confidence:
            category.all_scores = None

        processing_time = (time.time() - start_time) * 1000

        return ClassificationResponse(
            text=request.text[:200] + "..." if len(request.text) > 200 else request.text,
            category=category,
            sentiment=sentiment,
            keywords=keywords,
            processing_time_ms=round(processing_time, 2)
        )

    def classify_batch(self, texts: List[str]) -> Tuple[List[ClassificationResponse], float]:
        """Classify multiple texts."""
        start_time = time.time()

        results = []
        for text in texts:
            request = ClassificationRequest(text=text)
            result = self.classify(request)
            results.append(result)

        total_time = (time.time() - start_time) * 1000
        return results, round(total_time, 2)

    @property
    def is_trained(self) -> bool:
        return self._trained

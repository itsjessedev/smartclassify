from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class Category(str, Enum):
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    SCIENCE = "science"
    POLITICS = "politics"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ClassificationRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    include_confidence: bool = True


class CategoryPrediction(BaseModel):
    category: Category
    confidence: float
    all_scores: Optional[Dict[str, float]] = None


class SentimentPrediction(BaseModel):
    sentiment: SentimentLabel
    confidence: float
    scores: Dict[str, float]


class ClassificationResponse(BaseModel):
    text: str
    category: CategoryPrediction
    sentiment: SentimentPrediction
    keywords: List[str]
    processing_time_ms: float


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    results: List[ClassificationResponse]
    total_processing_time_ms: float


class ModelInfo(BaseModel):
    name: str
    version: str
    categories: List[str]
    accuracy: float
    last_trained: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str

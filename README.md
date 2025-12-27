# SmartClassify - ML Text Classification Pipeline

A production-ready machine learning pipeline for multi-task text classification including category detection and sentiment analysis.

## Features

- **Category Classification**: 7 categories (Technology, Business, Sports, Entertainment, Health, Science, Politics)
- **Sentiment Analysis**: Positive, Negative, Neutral detection with confidence scores
- **Keyword Extraction**: Automatic extraction of key terms
- **Batch Processing**: Classify multiple texts efficiently
- **REST API**: FastAPI-powered with automatic documentation
- **Demo UI**: Interactive web interface

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

Visit http://localhost:8001 for the demo UI.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Demo UI |
| `/health` | GET | Health check |
| `/model` | GET | Model information |
| `/classify` | POST | Classify single text |
| `/classify/batch` | POST | Classify multiple texts |

## Example Request

```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "The new smartphone features AI-powered camera", "include_confidence": true}'
```

## Response Format

```json
{
  "text": "The new smartphone features AI-powered camera",
  "category": {
    "category": "technology",
    "confidence": 0.89,
    "all_scores": {"technology": 0.89, "business": 0.05, ...}
  },
  "sentiment": {
    "sentiment": "neutral",
    "confidence": 0.72,
    "scores": {"positive": 0.15, "neutral": 0.72, "negative": 0.13}
  },
  "keywords": ["smartphone", "features", "camera", "powered"],
  "processing_time_ms": 2.34
}
```

## Architecture

- **TF-IDF Vectorization**: Text feature extraction with n-grams
- **Naive Bayes Classifier**: Fast, probabilistic classification
- **Pipeline Pattern**: Modular, extensible design
- **FastAPI**: Modern async web framework

## Tech Stack

- FastAPI - Web framework
- scikit-learn - Machine learning
- Pandas & NumPy - Data processing
- Pydantic - Data validation

## License

MIT

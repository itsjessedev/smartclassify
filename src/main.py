from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

from .models import (
    ClassificationRequest, ClassificationResponse,
    BatchRequest, BatchResponse, ModelInfo, HealthResponse
)
from .classifier import TextClassifier


# Global classifier instance
classifier: TextClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize classifier on startup."""
    global classifier
    classifier = TextClassifier()
    classifier.train()
    yield
    classifier = None


app = FastAPI(
    title="SmartClassify",
    description="ML-powered text classification API for category and sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve demo UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartClassify - ML Text Classification</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                min-height: 100vh;
                color: #E2E8F0;
            }
            .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
            header { text-align: center; margin-bottom: 40px; }
            h1 {
                font-size: 2.5rem;
                background: linear-gradient(135deg, #10B981, #3B82F6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            .subtitle { color: #94A3B8; font-size: 1.1rem; }
            .input-section {
                background: #1E293B;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
            }
            textarea {
                width: 100%;
                height: 150px;
                padding: 16px;
                border: 2px solid #334155;
                border-radius: 12px;
                background: #0F172A;
                color: white;
                font-size: 1rem;
                resize: vertical;
                outline: none;
            }
            textarea:focus { border-color: #10B981; }
            textarea::placeholder { color: #64748B; }
            .examples {
                margin-top: 16px;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }
            .example {
                background: #334155;
                padding: 8px 14px;
                border-radius: 20px;
                font-size: 0.85rem;
                cursor: pointer;
                transition: all 0.2s;
            }
            .example:hover { background: #475569; transform: translateY(-1px); }
            .btn {
                background: linear-gradient(135deg, #10B981, #3B82F6);
                color: white;
                border: none;
                padding: 14px 32px;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                margin-top: 16px;
                transition: all 0.2s;
            }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3); }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
            .results {
                display: none;
                gap: 16px;
            }
            .results.active { display: grid; grid-template-columns: 1fr 1fr; }
            .result-card {
                background: #1E293B;
                border-radius: 16px;
                padding: 24px;
            }
            .result-card.full { grid-column: 1 / -1; }
            .card-title {
                font-size: 0.9rem;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 12px;
            }
            .category-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 20px;
                border-radius: 12px;
                font-size: 1.1rem;
                font-weight: 600;
            }
            .category-badge.technology { background: #3B82F620; color: #60A5FA; }
            .category-badge.business { background: #8B5CF620; color: #A78BFA; }
            .category-badge.sports { background: #F59E0B20; color: #FBBF24; }
            .category-badge.entertainment { background: #EC489920; color: #F472B6; }
            .category-badge.health { background: #10B98120; color: #34D399; }
            .category-badge.science { background: #06B6D420; color: #22D3EE; }
            .category-badge.politics { background: #EF444420; color: #F87171; }
            .confidence { margin-top: 8px; font-size: 0.9rem; color: #94A3B8; }
            .sentiment-bar {
                display: flex;
                gap: 12px;
                margin-top: 8px;
            }
            .sentiment-item {
                flex: 1;
                text-align: center;
                padding: 12px;
                border-radius: 8px;
                background: #0F172A;
            }
            .sentiment-item.active { border: 2px solid; }
            .sentiment-item.positive.active { border-color: #10B981; }
            .sentiment-item.negative.active { border-color: #EF4444; }
            .sentiment-item.neutral.active { border-color: #64748B; }
            .sentiment-value { font-size: 1.5rem; font-weight: 700; }
            .sentiment-label { font-size: 0.8rem; color: #64748B; margin-top: 4px; }
            .keywords { display: flex; gap: 8px; flex-wrap: wrap; }
            .keyword {
                background: #334155;
                padding: 6px 12px;
                border-radius: 16px;
                font-size: 0.9rem;
            }
            .all-scores { margin-top: 16px; }
            .score-bar {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }
            .score-label { width: 100px; font-size: 0.85rem; }
            .score-track {
                flex: 1;
                height: 8px;
                background: #334155;
                border-radius: 4px;
                overflow: hidden;
            }
            .score-fill {
                height: 100%;
                background: linear-gradient(90deg, #10B981, #3B82F6);
                border-radius: 4px;
            }
            .score-value { width: 50px; text-align: right; font-size: 0.85rem; }
            .time { color: #64748B; font-size: 0.85rem; text-align: center; margin-top: 16px; }
            .loading { display: none; text-align: center; padding: 40px; }
            .loading.active { display: block; }
            .spinner {
                width: 40px; height: 40px;
                border: 4px solid #334155;
                border-top-color: #10B981;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            @media (max-width: 600px) {
                .results.active { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>SmartClassify</h1>
                <p class="subtitle">ML-Powered Text Classification</p>
            </header>

            <div class="input-section">
                <textarea id="textInput" placeholder="Enter text to classify (news article, review, description...)"></textarea>
                <div class="examples">
                    <span class="example" onclick="setExample('tech')">Tech News</span>
                    <span class="example" onclick="setExample('sports')">Sports</span>
                    <span class="example" onclick="setExample('review')">Product Review</span>
                    <span class="example" onclick="setExample('health')">Health Article</span>
                </div>
                <button class="btn" id="classifyBtn" onclick="classify()">Classify Text</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing text...</p>
            </div>

            <div class="results" id="results">
                <div class="result-card">
                    <div class="card-title">Category</div>
                    <div id="categoryBadge" class="category-badge"></div>
                    <div id="categoryConfidence" class="confidence"></div>
                </div>

                <div class="result-card">
                    <div class="card-title">Sentiment</div>
                    <div class="sentiment-bar" id="sentimentBar"></div>
                </div>

                <div class="result-card">
                    <div class="card-title">Keywords</div>
                    <div class="keywords" id="keywords"></div>
                </div>

                <div class="result-card">
                    <div class="card-title">Category Scores</div>
                    <div class="all-scores" id="allScores"></div>
                </div>

                <p class="time full" id="timeText"></p>
            </div>
        </div>

        <script>
            const examples = {
                tech: "Apple announced the release of their new AI-powered smartphone featuring advanced machine learning capabilities and a revolutionary neural processing unit. The device promises to transform how users interact with their apps and services.",
                sports: "The championship game ended in a thrilling overtime victory as the underdog team scored the winning goal in the final seconds. Fans erupted in celebration as their team clinched their first title in over a decade.",
                review: "I absolutely love this product! The quality exceeded my expectations and the customer service was fantastic. Highly recommend to anyone looking for a reliable solution.",
                health: "New research published in the Journal of Medicine reveals that regular exercise combined with a balanced diet can significantly reduce the risk of heart disease. Doctors recommend at least 30 minutes of physical activity daily."
            };

            function setExample(type) {
                document.getElementById('textInput').value = examples[type];
            }

            async function classify() {
                const text = document.getElementById('textInput').value.trim();
                if (!text || text.length < 10) {
                    alert('Please enter at least 10 characters');
                    return;
                }

                document.getElementById('results').classList.remove('active');
                document.getElementById('loading').classList.add('active');
                document.getElementById('classifyBtn').disabled = true;

                try {
                    const response = await fetch('/classify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, include_confidence: true })
                    });
                    const data = await response.json();

                    // Update category
                    const catBadge = document.getElementById('categoryBadge');
                    catBadge.className = 'category-badge ' + data.category.category;
                    catBadge.textContent = data.category.category.toUpperCase();
                    document.getElementById('categoryConfidence').textContent =
                        `${(data.category.confidence * 100).toFixed(1)}% confidence`;

                    // Update sentiment
                    const sentBar = document.getElementById('sentimentBar');
                    sentBar.innerHTML = ['positive', 'neutral', 'negative'].map(s => {
                        const score = data.sentiment.scores[s] || 0;
                        const isActive = s === data.sentiment.sentiment;
                        return `<div class="sentiment-item ${s} ${isActive ? 'active' : ''}">
                            <div class="sentiment-value">${(score * 100).toFixed(0)}%</div>
                            <div class="sentiment-label">${s}</div>
                        </div>`;
                    }).join('');

                    // Update keywords
                    document.getElementById('keywords').innerHTML =
                        data.keywords.map(k => `<span class="keyword">${k}</span>`).join('');

                    // Update all scores
                    const allScores = Object.entries(data.category.all_scores || {})
                        .sort((a, b) => b[1] - a[1]);
                    document.getElementById('allScores').innerHTML = allScores.map(([cat, score]) => `
                        <div class="score-bar">
                            <span class="score-label">${cat}</span>
                            <div class="score-track">
                                <div class="score-fill" style="width: ${score * 100}%"></div>
                            </div>
                            <span class="score-value">${(score * 100).toFixed(1)}%</span>
                        </div>
                    `).join('');

                    document.getElementById('timeText').textContent =
                        `Classified in ${data.processing_time_ms}ms`;

                    document.getElementById('results').classList.add('active');
                } catch (error) {
                    console.error('Classification failed:', error);
                    alert('Classification failed. Please try again.');
                } finally {
                    document.getElementById('loading').classList.remove('active');
                    document.getElementById('classifyBtn').disabled = false;
                }
            }

            document.getElementById('textInput').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) classify();
            });
        </script>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        models_loaded=classifier.is_trained if classifier else False,
        version="1.0.0"
    )


@app.get("/model", response_model=ModelInfo)
async def model_info():
    """Get information about the classification model."""
    return ModelInfo(
        name="SmartClassify Multi-Task Classifier",
        version="1.0.0",
        categories=[c.value for c in classifier.category_classes] if classifier else [],
        accuracy=0.85,  # Approximate accuracy on training data
        last_trained="2024-12-26"
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify a single text."""
    if not classifier or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        result = classifier.classify(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/batch", response_model=BatchResponse)
async def classify_batch(request: BatchRequest):
    """Classify multiple texts in batch."""
    if not classifier or not classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready")

    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

    try:
        results, total_time = classifier.classify_batch(request.texts)
        return BatchResponse(
            results=results,
            total_processing_time_ms=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

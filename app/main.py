from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.model import ModerationClassifier
from app.schemas import ModerationRequest, ModerationResponse, ClassScores

# 1. Manage the startup and shutdown lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {settings.model_id} into memory...")
    app.state.classifier = ModerationClassifier()
    print("Model loaded successfully.")
    yield
    app.state.classifier = None
    print("Model unloaded.")

# Toggle debug mode based on the Cloud Run environment variable
is_dev = settings.environment == "development"

app = FastAPI(
    title="BERTweet-Guard API", 
    lifespan=lifespan,
    debug=is_dev
)

# 2. Add CORS Middleware to allow external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains to test the frontend anywhere
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Root Endpoint (API Status)
@app.get("/")
async def root():
    return {
        "message": "BERTweet-Guard API is running",
        "docs_url": "/docs",
        "health_url": "/health"
    }

# 4. Health Check Endpoint (Crucial for Cloud Run Startup Probes)
@app.get("/health")
async def health_check():
    return {
        "status": "active", 
        "environment": settings.environment,
        "model": settings.model_id,
        "threshold": settings.decision_threshold
    }

# 5. The main inference endpoint
@app.post("/api/v1/predict", response_model=ModerationResponse)
async def predict(request: ModerationRequest):
    classifier = app.state.classifier
    
    prediction_label, scores = classifier.predict(request.text)
    
    # Use the threshold from your configured settings
    flagged_prob = scores.get("flagged", 0.0)
    
    is_flagged = flagged_prob >= settings.decision_threshold
    final_prediction = "flagged" if is_flagged else "normal"
    margin = flagged_prob - settings.decision_threshold
    
    return ModerationResponse(
        prediction=final_prediction,
        scores=ClassScores(**scores),
        flagged=is_flagged,
        threshold_info={"threshold": settings.decision_threshold, "margin": margin}
    )
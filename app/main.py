from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# 2. Add a Health Check Endpoint (Crucial for Cloud Run Startup Probes)
@app.get("/health")
async def health_check():
    return {
        "status": "active", 
        "environment": settings.environment,
        "model": settings.model_id,
        "threshold": settings.decision_threshold
    }

# 3. Mount the static directory to serve UI assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# 4. Serve the single-page frontend on the root URL
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

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
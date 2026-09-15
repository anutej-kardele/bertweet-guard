from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.model import ModerationClassifier
from app.schemas import ModerationRequest, ModerationResponse, ClassScores

# 1. Manage the startup and shutdown lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading BERTweet-Guard model into memory...")
    # Instantiate the PyTorch wrapper once
    app.state.classifier = ModerationClassifier()
    print("Model loaded successfully.")
    yield
    # Clean up when the server shuts down
    app.state.classifier = None
    print("Model unloaded.")

# Initialize the application with the lifespan manager
app = FastAPI(title="BERTweet-Guard API", lifespan=lifespan)

# 2. Mount the static directory to serve UI assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Serve the single-page frontend on the root URL
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# 4. The main inference endpoint
@app.post("/api/v1/predict", response_model=ModerationResponse)
async def predict(request: ModerationRequest):
    classifier = app.state.classifier
    
    # Run the text through the pipeline (assuming predict now returns 2 probabilities)
    prediction_label, scores = classifier.predict(request.text)
    
    # Apply your custom optimal threshold
    DECISION_THRESHOLD = 0.6760
    flagged_prob = scores.get("flagged", 0.0)
    
    is_flagged = flagged_prob >= DECISION_THRESHOLD
    final_prediction = "flagged" if is_flagged else "normal"
    margin = flagged_prob - DECISION_THRESHOLD
    
    return ModerationResponse(
        prediction=final_prediction,
        scores=ClassScores(**scores),
        flagged=is_flagged,
        threshold_info={"threshold": DECISION_THRESHOLD, "margin": margin}
    )
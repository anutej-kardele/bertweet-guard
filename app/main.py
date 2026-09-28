from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model import ModerationClassifier
from app.schemas import (
    ModerationRequest,
    ModerationResponse,
    ClassScores,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(
        f"Loading {settings.model_id} into memory..."
    )

    classifier = ModerationClassifier()

    app.state.classifier = classifier

    print(
        "Model loaded successfully. "
        f"revision={classifier.model_revision} "
        f"threshold="
        f"{classifier.decision_threshold:.4f}"
    )

    yield

    app.state.classifier = None

    print("Model unloaded.")


is_dev = (
    settings.environment
    == "development"
)

app = FastAPI(
    title="BERTweet-Guard API",
    lifespan=lifespan,
    debug=is_dev,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://bertweet.anutej.us",
        "https://openstream.anutej.us",
        "https://anutej.us",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message":
            "BERTweet-Guard API is running",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health")
async def health_check():
    classifier = (
        app.state.classifier
    )

    return {
        "status": "active",
        "environment":
            settings.environment,
        "model":
            classifier.model_id,
        "model_revision":
            classifier.model_revision,
        "threshold":
            classifier.decision_threshold,
    }


@app.post(
    "/api/v1/predict",
    response_model=ModerationResponse,
)
async def predict(
    request: ModerationRequest,
):
    classifier = (
        app.state.classifier
    )

    _, scores = classifier.predict(
        request.text
    )

    flagged_prob = scores.get(
        "flagged",
        0.0,
    )

    # Threshold now comes from threshold.json.
    threshold = (
        classifier.decision_threshold
    )

    is_flagged = (
        flagged_prob
        >= threshold
    )

    final_prediction = (
        "flagged"
        if is_flagged
        else "normal"
    )

    margin = (
        flagged_prob
        - threshold
    )

    return ModerationResponse(
        prediction=final_prediction,
        scores=ClassScores(
            **scores
        ),
        flagged=is_flagged,
        threshold_info={
            "threshold": threshold,
            "margin": margin,
        },
    )
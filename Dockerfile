# Use a slim Python image to keep the container small
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD THE MODEL (Crucial for Cloud Run)
# This bakes the 540MB weights into the image so it doesn't download on every cold start.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('anutej9/bertweet-guard'); \
    AutoModelForSequenceClassification.from_pretrained('anutej9/bertweet-guard')"

# Copy the rest of your application code
COPY . .

# Cloud Run injects the $PORT environment variable (default 8080)
ENV PORT=8080

# Launch the FastAPI app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
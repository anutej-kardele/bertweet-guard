# Use a slim Python image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Change this value whenever the Hugging Face model is updated.
# This invalidates the model-download Docker layer.
ARG MODEL_VERSION=1

# Pre-download the BERTweet Guard model and tokenizer.
# The files are baked into the image for fast Cloud Run cold starts.
RUN python -c "from transformers import BertweetTokenizer, AutoModelForSequenceClassification; \
    print('Downloading BERTweet Guard model version ${MODEL_VERSION}'); \
    BertweetTokenizer.from_pretrained( \
    'anutej9/bertweet-guard', \
    normalization=True \
    ); \
    AutoModelForSequenceClassification.from_pretrained( \
    'anutej9/bertweet-guard' \
    )"

# Copy application code
COPY . .

# Cloud Run injects PORT
ENV PORT=8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
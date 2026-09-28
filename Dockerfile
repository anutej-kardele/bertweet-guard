FROM python:3.10-slim

WORKDIR /app


# --------------------------------------------------
# Install dependencies
# --------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# --------------------------------------------------
# Model cache version
#
# Increment when the Hugging Face artifact changes.
# --------------------------------------------------

ARG MODEL_VERSION=2


# --------------------------------------------------
# Download one exact Hugging Face snapshot
# --------------------------------------------------

RUN python -c "from pathlib import Path; from huggingface_hub import HfApi, snapshot_download; repo_id='anutej9/bertweet-guard'; revision=HfApi().model_info(repo_id).sha; print('Downloading BERTweet Guard model version ${MODEL_VERSION}'); print('Revision:', revision); snapshot_download(repo_id=repo_id, revision=revision, local_dir='/opt/bertweet-guard-model'); Path('/opt/bertweet-guard-model/REVISION').write_text(revision, encoding='utf-8'); print('Model artifact downloaded successfully.')"


# --------------------------------------------------
# Application
# --------------------------------------------------

COPY . .


# Backend loads model + tokenizer + threshold.json
# from this exact baked snapshot.
ENV MODEL_DIR=/opt/bertweet-guard-model


# Cloud Run injects PORT.
ENV PORT=8080


CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
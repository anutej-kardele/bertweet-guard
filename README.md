# BERTweet-Guard

A standalone FastAPI microservice for real-time hate speech and offensive content detection. Designed as a protective layer for the OpenStream microblogging platform, flagging content before it hits the PostgreSQL database.

## Architecture & Model
- **Base Model:** `vinai/bertweet-base`
- **Adaptation:** LoRA (r=16) on query + value matrices
- **Training:** Fine-tuned on the THOS dataset with Class-Weighted Focal Loss. 
- **Backend:** FastAPI, PyTorch, Transformers

## Local Setup

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Create a virtual environment and install the requirements:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
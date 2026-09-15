# OpenStream Moderation — BERTweet + LoRA

A research-to-deployment text moderation project built around **BERTweet**, **LoRA**, **Focal Loss**, and **validation-tuned decision thresholds**.

The project was developed in two phases:

1. **Experiment broadly on a smaller dataset (THOS)** to compare architectures, LoRA configurations, loss functions, schedulers, and fine-tuning strategies.
2. **Scale the best-performing approach to the much larger Jigsaw Toxic Comment dataset**, then refine preprocessing, class weighting, checkpoint selection, and the deployment threshold.

The repository now includes both the **research notebooks** and a **production-style FastAPI inference service** with a small browser UI.

> **Current deployment task:** binary moderation — `normal` vs `flagged`  
> **Primary model-selection metric:** Macro F1  
> **Current backbone:** `vinai/bertweet-base`

---

## 1. Project Evolution

### Phase 1 — THOS experimentation

> **Public repository note:** The THOS work was completed as part of a university project. The original THOS notebook and university-project implementation are **not included in this repository**. Only a high-level summary of the experimentation and conclusions is documented here.

The first stage used the **THOS dataset** as a smaller, faster experimentation environment.

The THOS task was three-class classification:

- `normal`
- `offensive`
- `hate`

The goal was not to commit to one model immediately. Instead, many combinations were tested to determine which ideas consistently worked best before moving to a much larger dataset.

Experiments included:

- Zero-shot BERT
- Full BERT fine-tuning
- LoRA fine-tuning
- LoRA rank experiments (`r=8`, `r=16`, `r=32`)
- Focal Loss
- Focal Loss gamma experiments
- BERTweet
- Full BERTweet fine-tuning
- BERTweet + LoRA
- Broader LoRA target modules
- Cosine learning-rate scheduling
- Early stopping
- Longer training
- Attention visualization
- Fairness analysis

The strongest THOS result was:

**BERTweet + LoRA r=16 + Focal Loss (γ=1) + Cosine Scheduler**

| Metric | Score |
|---|---:|
| Test Macro F1 | **0.6848** |
| Best Validation F1 | 0.6741 |
| Normal F1 | 0.8125 |
| Offensive F1 | 0.5895 |
| Hate F1 | 0.6523 |

![THOS model comparison](docs/thos_model_comparison.png)

<details>
<summary><strong>THOS experiment comparison</strong></summary>

| Model | Test Macro F1 | Best Val F1 | Normal F1 | Offensive F1 | Hate F1 |
|---|---:|---:|---:|---:|---:|
| **BERTweet LoRA r=16 + Focal(γ=1) + Cosine** | **0.6848** | 0.6741 | 0.8125 | 0.5895 | 0.6523 |
| LoRA r=16 broad | 0.6810 | 0.6282 | 0.7951 | 0.5801 | 0.6680 |
| BERTweet Full | 0.6714 | 0.6565 | 0.7855 | 0.5508 | 0.6781 |
| BERT Extended (Cosine) | 0.6677 | 0.6304 | 0.7740 | 0.5591 | 0.6699 |
| BERTweet LoRA r=16 Q+K+V+Dense + Focal(γ=1) + Cosine | 0.6667 | 0.6665 | 0.7805 | 0.5580 | 0.6615 |
| LoRA r=32 | 0.6563 | 0.6371 | 0.7604 | 0.5588 | 0.6498 |
| LoRA r=16 | 0.6543 | 0.6402 | 0.7990 | 0.5514 | 0.6126 |
| BERT LoRA r=16 Q+K+V+Dense + Focal(γ=1) + Cosine | 0.6488 | 0.6381 | 0.7871 | 0.5509 | 0.6083 |
| Fine-Tuned BERT | 0.6486 | 0.6415 | 0.7990 | 0.5538 | 0.5931 |
| LoRA + Focal Loss | 0.6482 | 0.6234 | 0.7956 | 0.5633 | 0.5856 |
| BERTweet + LoRA | 0.6460 | 0.6408 | 0.7880 | 0.5680 | 0.5819 |
| BERT LoRA r=16 + Focal(γ=1) + Cosine | 0.6433 | 0.6344 | 0.7688 | 0.5422 | 0.6189 |
| LoRA BERT (r=8) | 0.6378 | 0.6296 | 0.8007 | 0.5271 | 0.5855 |
| LoRA + FL (γ=1.0) | 0.6365 | 0.6302 | 0.7966 | 0.5429 | 0.5701 |
| Zero-Shot BERT | 0.0879 | n/a | 0.0708 | 0.1931 | 0.0000 |

</details>

The THOS experiments showed that the best overall direction was:

```text
BERTweet
    +
LoRA
    +
Focal Loss
    +
Cosine learning-rate scheduling
```

That combination became the starting point for the larger-data phase.

---

## 2. Scaling to Jigsaw

After identifying the strongest strategy on THOS, the project moved to the much larger **Jigsaw Toxic Comment Classification** dataset.

The Jigsaw labels are collapsed into a binary moderation target:

```text
normal  = 0
flagged = 1
```

A comment is marked `flagged` when any of these source labels is positive:

- `toxic`
- `severe_toxic`
- `threat`
- `obscene`
- `insult`
- `identity_hate`

The larger dataset uses a **90 / 5 / 5 stratified train / validation / test split**.

The Jigsaw phase refined the THOS-winning approach rather than restarting from scratch.

### Current training configuration

| Component | Value |
|---|---|
| Backbone | `vinai/bertweet-base` |
| Task | Binary sequence classification |
| Labels | `normal`, `flagged` |
| Maximum sequence length | 128 |
| Batch size | 32 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.20 |
| LoRA targets | `query`, `key`, `value`, `dense` |
| Loss | Weighted Focal Loss |
| Focal gamma | 1.0 |
| Class weights | `[1.0, 1.25]` |
| Learning rate | `5e-5` |
| Weight decay | `0.05` |
| Maximum epochs | 5 |
| Early-stopping patience | 2 |
| Scheduler | Cosine with warmup |
| Warmup ratio | 0.06 |
| Random seed | 42 |
| Main metric | Macro F1 |

The current v4 notebook also aligns preprocessing more closely with BERTweet and selects checkpoints using **threshold-tuned validation Macro F1**.

---

## 3. Final Modeling Pipeline

```text
Jigsaw toxicity data
        ↓
Binary normal / flagged labels
        ↓
Minimal preprocessing
        ↓
BERTweet tokenizer
        ↓
BERTweet + LoRA fine-tuning
        ↓
Weighted Focal Loss
        ↓
Cosine LR + warmup
        ↓
Validation probability predictions
        ↓
Threshold optimization using Macro F1
        ↓
Best checkpoint selection
        ↓
Held-out test evaluation
        ↓
Merge LoRA into BERTweet
        ↓
Deploy model + optimized threshold
```

The important point is that this is not only a model-training pipeline. The **decision threshold is part of the trained artifact**.

---

## 4. Why the Decision Threshold Matters

The service does not simply use:

```python
prediction = logits.argmax()
```

Instead, it uses the probability of the `flagged` class:

```python
prediction = int(flagged_probability >= decision_threshold)
```

The threshold is selected using the validation set and then saved with the model.

For example:

```text
flagged probability = 0.5478
threshold           = 0.6200
prediction          = NORMAL
```

Even though the raw `flagged` probability is above 0.50, it is still below the operating threshold.

This is useful for reducing false positives without retraining the model every time the desired operating point changes.

---

## 5. Latest Reported Large-Dataset Result

A recent pre-v4 run produced:

| Metric | Result |
|---|---:|
| Threshold-tuned Validation Macro F1 | **0.9185** |
| Test Macro F1 | **0.9174** |
| Test Accuracy | **0.9710** |
| Selected threshold | **0.62** |
| Flagged precision | 0.89 |
| Flagged recall | 0.81 |
| Flagged F1 | 0.85 |

The v4 notebook modifies preprocessing alignment and checkpoint selection, so v4 results should be regenerated by running the notebook rather than copying the earlier metrics.

---

# Project Structure

```text
openstream-moderation/
├── app/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings and environment parsing
│   ├── main.py            # FastAPI application and lifespan model loading
│   ├── model.py           # Hugging Face inference + threshold logic
│   └── schemas.py         # Request / response validation models
│
├── static/
│   └── index.html         # Interactive browser dashboard
│
├── notebooks/
│   └── bertweet_moderation_model_v4_bertweet_aligned.ipynb
│
├── docs/
│   └── thos_model_comparison.png
│
├── .env.example           # Environment-variable template
├── .gitignore
├── requirements.txt
├── setup.sh               # Create environment and install dependencies
├── launch.sh              # Launch API or research notebooks
└── README.md
```

The structure deliberately separates:

- **research/training** under `notebooks/`
- **production inference** under `app/`
- **browser testing UI** under `static/`

The production model itself is hosted on Hugging Face and is downloaded automatically when the API starts.

This follows the same deployment pattern used in the companion BERTweet Guard service.

---

# Quickstart

## Prerequisites

- Python **3.10+**
- Git
- Internet access for first-time Hugging Face model download
- A GPU is strongly recommended for retraining; inference can run on CPU

---

## 1. Clone the repository

Replace the URL below with the final repository URL if this project is published separately.

```bash
git clone <repository-url>
cd openstream-moderation
```

---

## 2. Run setup

```bash
chmod +x setup.sh launch.sh
./setup.sh
```

`setup.sh` will:

1. create `.venv`,
2. install the dependencies in `requirements.txt`,
3. create `.env` from `.env.example` if it does not already exist,
4. run an import/environment check.

The generated `.env` contains only the Hugging Face model ID and environment name.

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env
```

---

## 3. Configure the environment

The service loads the production model directly from Hugging Face.

Create `.env` from the template if it was not already created by `setup.sh`:

```bash
cp .env.example .env
```

The environment only needs:

```ini
MODEL_ID="anutej9/bertweet-guard"
ENVIRONMENT="development"
```

### Model loading

On application startup:

```text
.env
 ↓
MODEL_ID = anutej9/bertweet-guard
 ↓
FastAPI starts
 ↓
Model + tokenizer are pulled from Hugging Face
 ↓
threshold.json is loaded from the same repository
 ↓
Inference service becomes ready
```

The model files are cached automatically by the Hugging Face / Transformers stack after the first download.

There is no local model directory or local artifact fallback in this repository.

The decision threshold should be shipped with the Hugging Face model repository in `threshold.json`, so the API uses the same validation-selected threshold that was produced during training.


# Launch

## Start the API and web dashboard

```bash
./launch.sh
```

or:

```bash
./launch.sh api
```

Equivalent direct command:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once started:

- **Web Dashboard:** `http://localhost:8000`
- **Swagger / OpenAPI Docs:** `http://localhost:8000/docs`
- **Health Endpoint:** `http://localhost:8000/health`

---

## Open the latest training notebook

```bash
./launch.sh notebook
```

## Open a specific notebook

```bash
./launch.sh path/to/notebook.ipynb
```

---

# API Reference

## Analyze text

```text
POST /api/v1/predict
```

### Request

```json
{
  "text": "I am going to attend the protest against the governor"
}
```

### Response

Example shape:

```json
{
  "prediction": "normal",
  "scores": {
    "normal": 0.4522,
    "flagged": 0.5478
  },
  "flagged": false,
  "threshold_info": {
    "threshold": 0.6200,
    "margin": -0.0722
  }
}
```

`margin` is:

```text
flagged_probability - threshold
```

Therefore:

- positive margin → above the moderation threshold
- negative margin → below the moderation threshold

The probability and Boolean decision are intentionally both returned so downstream systems can apply their own risk policy if needed.

---

# Web Dashboard

The root endpoint serves a lightweight browser interface.

It lets a user:

- enter text,
- call `/api/v1/predict`,
- view the normal probability,
- view the flagged probability,
- view the current decision threshold,
- view the threshold margin,
- open the Swagger API documentation.

The UI is intentionally separate from the inference logic so the FastAPI service can also be consumed directly by another backend or client application.

---

# Research Notebook

## Public repository boundary

The earlier THOS experimentation was completed as part of university work. Its source notebook is **intentionally not included in this public repository** in order to comply with university terms.

The README retains only a high-level description of the experimentation process and the resulting modeling direction. The underlying university notebook, assignment material, and implementation are not distributed here.

## `bertweet_moderation_model_v4_bertweet_aligned.ipynb`

The current larger-dataset training notebook.

It contains:

- Jigsaw loading
- binary-label construction
- BERTweet-aligned preprocessing
- stratified 90 / 5 / 5 splitting
- sequence-length diagnostics
- BERTweet + LoRA training
- Weighted Focal Loss
- cosine scheduling with warmup
- threshold-tuned checkpoint selection
- fine-grained threshold search
- held-out test evaluation
- LoRA merge
- production artifact export
- exported-model verification
- inference sanity checks

Use this notebook for the current training/retraining workflow.

---

# Model Export and Deployment

The training notebook produces a merged Transformers model and the validation-selected `threshold.json`.

For production, those files are uploaded to the Hugging Face repository configured by:

```ini
MODEL_ID="anutej9/bertweet-guard"
```

The FastAPI application then loads the deployment model directly from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    settings.model_id,
    use_fast=False,
    normalization=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    settings.model_id
)
```

The service also downloads:

```text
threshold.json
```

from the same Hugging Face repository and uses that value for the binary moderation decision.

This keeps the Git repository lightweight: trained model weights are not committed to GitHub.


# Production Deployment Considerations

## Memory

A standard float32 BERT/BERTweet-class model can require hundreds of megabytes of memory before runtime overhead.

For memory-constrained deployments, consider:

- FP16 inference where supported,
- ONNX export,
- INT8 quantization,
- a smaller CPU-optimized serving image,
- separating the web frontend from the inference process.

Do not assume a free-tier instance with roughly 512 MB RAM will comfortably host the unquantized model.

---

## Threshold decoupling

The API exposes both:

```text
flagged: true / false
```

and the underlying class probabilities.

This enables different upstream policies.

For example:

```text
flagged_probability >= 0.85
    → automatically reject

0.65 <= flagged_probability < 0.85
    → send to human review

flagged_probability < 0.65
    → allow
```

Those values are examples of an application policy, not replacements for the validation-selected model threshold.

---

## Human review

For high-impact moderation, use the classifier as one signal in a larger system.

Borderline cases, ambiguous context, political discussion, quoted abuse, satire, or reclaimed language can still require human review.

---

# Evaluation Strategy

The training workflow deliberately separates the three datasets:

```text
Training set
    → optimization

Validation set
    → checkpoint selection
    → decision-threshold selection

Test set
    → final evaluation only
```

The test set is not used to select:

- epochs,
- model configuration,
- class weights,
- LoRA parameters,
- decision threshold.

Macro F1 is the main metric because accuracy alone can hide weak minority-class performance.

---

# Reproducibility

The training notebooks use:

```python
RANDOM_SEED = 42
```

For fair experiment comparisons, keep these fixed unless they are the explicit subject of an ablation:

- dataset split
- random seed
- preprocessing
- evaluation metric
- threshold-selection procedure
- test-set isolation

---

# Limitations

This is a research and deployment prototype, not a complete moderation policy.

Known limitations include:

- current production output is binary rather than multi-policy,
- Jigsaw annotations can contain dataset bias,
- BERTweet has a relatively short sequence budget,
- long Jigsaw comments may be truncated,
- sarcasm and context remain difficult,
- political or protest-related language can be close to the decision boundary,
- one global threshold may not fit every application,
- the current model is primarily intended for English-language text.

---

# Future Work

Potential improvements include:

- chunking long comments rather than truncating them,
- probability calibration,
- hard-negative mining,
- broader fairness analysis on the final binary model,
- multi-label policy categories,
- ONNX export,
- INT8 quantization,
- FP16 deployment,
- Docker packaging,
- automated API tests,
- production monitoring and threshold-drift analysis.

---

# Summary

The project follows a deliberate **research-first, scale-second, deploy-third** strategy:

```text
THOS
    ↓
Compare many modeling approaches
    ↓
Identify BERTweet + LoRA + Focal + Cosine as strongest direction
    ↓
Move to large Jigsaw dataset
    ↓
Refine weighting, warmup, preprocessing, and LoRA setup
    ↓
Tune checkpoint selection using validation Macro F1
    ↓
Tune deployment threshold on validation data
    ↓
Evaluate once on the held-out test set
    ↓
Merge/export the model
    ↓
Serve through FastAPI + browser dashboard
```

The final system therefore combines both sides of the project:

**controlled model experimentation** and **practical deployment-oriented inference**.

---

## Public Release Scope

This public repository contains the current Jigsaw/BERTweet moderation implementation and deployment service.
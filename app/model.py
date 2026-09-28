import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    BertweetTokenizer,
    AutoModelForSequenceClassification,
)

from app.config import settings


class ModerationClassifier:
    def __init__(
        self,
        model_id: str = settings.model_id,
        model_dir: str | None = settings.model_dir,
    ):
        self.model_id = model_id

        # ---------------------------------------------------------
        # Resolve one complete model artifact.
        #
        # Cloud Run:
        #   Uses MODEL_DIR baked into the Docker image.
        #
        # Local development:
        #   Resolves one exact Hugging Face commit and downloads
        #   that snapshot.
        # ---------------------------------------------------------
        if model_dir:
            artifact_dir = Path(model_dir)

            if not artifact_dir.exists():
                raise RuntimeError(
                    f"MODEL_DIR does not exist: {artifact_dir}"
                )

            revision_file = artifact_dir / "REVISION"

            if revision_file.exists():
                self.model_revision = (
                    revision_file
                    .read_text(encoding="utf-8")
                    .strip()
                )
            else:
                self.model_revision = "local"

        else:
            api = HfApi()

            self.model_revision = api.model_info(
                model_id
            ).sha

            snapshot_path = snapshot_download(
                repo_id=model_id,
                revision=self.model_revision,
            )

            artifact_dir = Path(snapshot_path)

        self.artifact_dir = artifact_dir

        print(
            f"Loading model artifact from: {artifact_dir}"
        )

        print(
            f"Model revision: {self.model_revision}"
        )

        # ---------------------------------------------------------
        # Tokenizer
        # ---------------------------------------------------------
        self.tokenizer = BertweetTokenizer.from_pretrained(
            str(artifact_dir),
            normalization=True,
        )

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                str(artifact_dir)
            )
        )

        self.model.eval()

        # ---------------------------------------------------------
        # Threshold
        #
        # This now comes from the SAME artifact as the model.
        # ---------------------------------------------------------
        threshold_path = (
            artifact_dir / "threshold.json"
        )

        if not threshold_path.exists():
            raise RuntimeError(
                "threshold.json is missing from the "
                f"model artifact: {threshold_path}"
            )

        with threshold_path.open(
            "r",
            encoding="utf-8",
        ) as threshold_file:
            threshold_data = json.load(
                threshold_file
            )

        try:
            self.decision_threshold = float(
                threshold_data["threshold"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "threshold.json must contain a "
                "numeric 'threshold' value."
            ) from exc

        if not (
            0.0
            <= self.decision_threshold
            <= 1.0
        ):
            raise RuntimeError(
                "Invalid decision threshold: "
                f"{self.decision_threshold}"
            )

        print(
            "Decision threshold loaded from artifact: "
            f"{self.decision_threshold:.4f}"
        )

        # Current model is binary.
        self.id2label = {
            0: "normal",
            1: "flagged",
        }

    def predict(
        self,
        text: str,
    ):
        # Keep this identical to the verified notebook
        # inference path.
        #
        # DO NOT lowercase.
        # DO NOT manually rewrite URLs/mentions/hashtags.
        clean_text = text.strip()

        inputs = self.tokenizer(
            clean_text,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs[
                    "input_ids"
                ],
                attention_mask=inputs[
                    "attention_mask"
                ],
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        ).squeeze(0).tolist()

        scores = {
            "normal": float(
                probabilities[0]
            ),
            "flagged": float(
                probabilities[1]
            ),
        }

        top_class_idx = int(
            torch.argmax(
                outputs.logits,
                dim=-1,
            ).item()
        )

        prediction = self.id2label[
            top_class_idx
        ]

        return prediction, scores
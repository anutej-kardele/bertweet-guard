import torch
from transformers import BertweetTokenizer, AutoModelForSequenceClassification
from app.config import settings


class ModerationClassifier:
    def __init__(self, model_id: str = settings.model_id):
        # Use the exact tokenizer class used by the verified Hugging Face artifact.
        self.tokenizer = BertweetTokenizer.from_pretrained(
            model_id,
            normalization=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

        # Current production model is binary.
        self.id2label = {
            0: "normal",
            1: "flagged",
        }

    def predict(self, text: str):
        # IMPORTANT:
        # Do not lowercase or manually rewrite mentions/URLs/hashtags here.
        # The verified Hugging Face artifact was tested on the original text.
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
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        ).squeeze(0).tolist()

        scores = {
            "normal": float(probabilities[0]),
            "flagged": float(probabilities[1]),
        }

        top_class_idx = int(torch.argmax(outputs.logits, dim=-1).item())
        prediction = self.id2label[top_class_idx]

        return prediction, scores
import re
import html
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import settings

class ModerationClassifier:
    def __init__(self, model_id: str = settings.model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            use_fast=False, 
            normalization=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        # Set the model to evaluation mode (disables dropout layers)
        self.model.eval()

        # Map indices to your ablation study class labels
        self.id2label = {0: "normal", 1: "offensive", 2: "hate"}

    def preprocess(self, text: str) -> str:
        # Convert to lowercase and unescape HTML entities
        text = text.lower()
        text = html.unescape(text)
        
        # Apply specific replacement rules to match the training pipeline
        text = re.sub(r'@\w+', '[USER]', text)
        text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
        
        # Strip the '#' character from hashtags but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Collapse multiple whitespaces into a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def predict(self, text: str):
        clean_text = self.preprocess(text)

        # Tokenize with the exact constraints from the training configuration
        inputs = self.tokenizer(
            clean_text,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Run the forward pass without tracking gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply softmax to the raw logits to extract human-readable probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()

        scores = {
            "normal": float(probabilities[0]),
            "flagged": float(probabilities[1])
        }

        # Identify the class with the highest confidence score
        top_class_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
        prediction = self.id2label[top_class_idx]

        return prediction, scores
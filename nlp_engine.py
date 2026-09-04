import torch
from transformers import pipeline

# Automatically utilize Apple Silicon (MPS) if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Component B: Multi-Dimensional Sentiment
# Using a model fine-tuned on GoEmotions (28 emotion categories)
emotion_classifier = pipeline(
    "text-classification", 
    model="SamLowe/roberta-base-go_emotions", 
    top_k=3, 
    device=device
)

def infer_sentiment(text: str) -> list:
    """Returns top 3 nuanced emotions (e.g., approval, annoyance, excitement)."""
    results = emotion_classifier(text)
    return results[0]

# Component C: Demographic Profiling (Zero-Shot)
zero_shot_classifier = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli", 
    device=device
)

def infer_profession(bio_text: str) -> str:
    """Infers professional interest from public bio text."""
    candidate_labels = ["technology", "finance", "healthcare", "education", "arts", "politics"]
    result = zero_shot_classifier(bio_text, candidate_labels)
    return result['labels'][0] # Return the highest confidence label
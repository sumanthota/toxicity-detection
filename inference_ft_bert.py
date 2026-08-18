"""Label arbitrary text with the fine-tuned bert-base-uncased checkpoint
produced by fine_tune_bert.py.

Usage:
    python inference_ft_bert.py "You are an idiot" "Have a nice day!"

    # or read text interactively when no arguments are given
    python inference_ft_bert.py
"""

import argparse

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "fine_tuned_bert"
MAX_LENGTH = 128
CLASSIFICATION_THRESHOLD = 0.5

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label text for toxicity using the fine-tuned BERT model.")
    parser.add_argument("texts", nargs="*", help="One or more comments to label. Reads a single line from stdin if omitted.")
    return parser.parse_args()


def load_model(model_path: str) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return model, tokenizer


def predict_probs(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, texts: list[str]) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**encoded).logits
    return torch.sigmoid(logits).cpu().numpy()


def format_result(text: str, probs: np.ndarray) -> str:
    predicted = [label for label, prob in zip(LABEL_COLUMNS, probs) if prob >= CLASSIFICATION_THRESHOLD]
    lines = [f'Text: "{text}"', f"Labels: {', '.join(predicted) if predicted else 'none'}"]
    lines += [f"  {label:<15} {prob:.4f}" for label, prob in zip(LABEL_COLUMNS, probs)]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    texts = args.texts or [input("Enter text to label: ")]

    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}\n")

    model, tokenizer = load_model(MODEL_PATH)
    all_probs = predict_probs(model, tokenizer, texts)

    for text, probs in zip(texts, all_probs):
        print(format_result(text, probs))
        print()


if __name__ == "__main__":
    main()

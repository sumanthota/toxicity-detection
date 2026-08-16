"""Evaluate raw bert-base-uncased (no fine-tuning) as a toxicity classifier,
using a held-out slice of train.csv as validation data.

bert-base-uncased has no toxicity classification head, so a fresh,
randomly-initialized head is attached here. Predictions are expected to be
close to random -- this is the "before fine-tuning" baseline to compare a
fine-tuned model against.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "bert-base-uncased"
TRAIN_CSV_PATH = "data/train.csv"
VALIDATION_SIZE = 2000
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_LENGTH = 128
CLASSIFICATION_THRESHOLD = 0.5

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_validation_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _, val_df = train_test_split(df, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
    return val_df.reset_index(drop=True)


def predict_in_batches(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, texts: list[str]) -> np.ndarray:
    all_probs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
            batch = texts[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main() -> None:
    val_df = load_validation_split(TRAIN_CSV_PATH)
    print(f"Validating on {len(val_df)} held-out comments from {TRAIN_CSV_PATH}")
    print(f"Device: {DEVICE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
    ).to(DEVICE)
    model.eval()

    probs = predict_in_batches(model, tokenizer, val_df["comment_text"].tolist())
    preds = (probs >= CLASSIFICATION_THRESHOLD).astype(int)

    print()
    auc_scores = []
    accuracy_scores = []
    for i, label in enumerate(LABEL_COLUMNS):
        auc = roc_auc_score(val_df[label], probs[:, i])
        accuracy = accuracy_score(val_df[label], preds[:, i])
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        print(f"{label:<15} AUC: {auc:.4f}   Accuracy: {accuracy:.4f}")

    print(f"\nMean column-wise AUC:      {np.mean(auc_scores):.4f}")
    print(f"Mean column-wise Accuracy: {np.mean(accuracy_scores):.4f}")


if __name__ == "__main__":
    main()

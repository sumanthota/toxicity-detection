"""Evaluate the pretrained Detoxify ("original") model with no additional
fine-tuning, using the same held-out slice of train.csv as evaluate_base.py.

Note: this checkpoint (unitary/toxic-bert) was already trained on this exact
Jigsaw dataset, so these numbers measure fit/memorization, not
generalization -- see README.md for context.
"""

import numpy as np
import pandas as pd
from detoxify import Detoxify
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TRAIN_CSV_PATH = "data/train.csv"
VALIDATION_SIZE = 2000
RANDOM_SEED = 42
BATCH_SIZE = 32
CLASSIFICATION_THRESHOLD = 0.5

# Detoxify's "original" checkpoint was trained on this exact dataset, so it
# needs no fine-tuning, but it renamed two of the six label columns.
LABEL_TO_DETOXIFY_KEY = {
    "toxic": "toxicity",
    "severe_toxic": "severe_toxicity",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_hate": "identity_attack",
}


def load_validation_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _, val_df = train_test_split(df, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
    return val_df.reset_index(drop=True)


def predict_in_batches(model: Detoxify, texts: list[str]) -> dict[str, np.ndarray]:
    predictions: dict[str, list[float]] = {key: [] for key in LABEL_TO_DETOXIFY_KEY.values()}
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
        batch = texts[start : start + BATCH_SIZE]
        batch_scores = model.predict(batch)
        for key in predictions:
            predictions[key].extend(batch_scores[key])
    return {key: np.array(values) for key, values in predictions.items()}


def main() -> None:
    val_df = load_validation_split(TRAIN_CSV_PATH)
    print(f"Validating on {len(val_df)} held-out comments from {TRAIN_CSV_PATH}\n")

    model = Detoxify("original")
    predictions = predict_in_batches(model, val_df["comment_text"].tolist())

    print()
    auc_scores = []
    accuracy_scores = []
    for label, detoxify_key in LABEL_TO_DETOXIFY_KEY.items():
        probs = predictions[detoxify_key]
        preds = (probs >= CLASSIFICATION_THRESHOLD).astype(int)
        auc = roc_auc_score(val_df[label], probs)
        accuracy = accuracy_score(val_df[label], preds)
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        print(f"{label:<15} AUC: {auc:.4f}   Accuracy: {accuracy:.4f}")

    print(f"\nMean column-wise AUC:      {np.mean(auc_scores):.4f}")
    print(f"Mean column-wise Accuracy: {np.mean(accuracy_scores):.4f}")


if __name__ == "__main__":
    main()

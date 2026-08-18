"""Evaluate the LoRA adapter produced by peft_base_bert.py, using the same
held-out slice of train.csv as evaluate_base.py, evaluate_detoxify.py, and
evaluate_ft_bert.py so the AUC/accuracy numbers are directly comparable
across all evaluation scripts.
"""

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL_NAME = "bert-base-uncased"
ADAPTER_PATH = "peft_bert"
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


def load_peft_model() -> AutoModelForSequenceClassification:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    return model


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
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Device: {DEVICE}\n")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    model = load_peft_model().to(DEVICE)
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

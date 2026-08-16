"""Fine-tune bert-base-uncased on train.csv using LoRA (via peft) instead of
full fine-tuning: the pretrained encoder stays frozen, and small trainable
low-rank matrices are injected into the attention query/value projections.

Uses the same VALIDATION_SIZE/RANDOM_SEED/TRAIN_SAMPLE_SIZE as
fine_tune_bert.py, so the resulting AUC is directly comparable to full
fine-tuning -- for a fraction of the trainable parameters.
"""

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "bert-base-uncased"
TRAIN_CSV_PATH = "data/train.csv"
OUTPUT_DIR = "peft_bert"

VALIDATION_SIZE = 2000
RANDOM_SEED = 42
MAX_LENGTH = 128

# Same cap as fine_tune_bert.py, so the two are trained on identical data.
TRAIN_SAMPLE_SIZE = 20000

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 2
# LoRA trains far fewer parameters than full fine-tuning, so it tolerates
# (and generally needs) a higher learning rate to converge in few epochs.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
CLASSIFICATION_THRESHOLD = 0.5

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class ToxicCommentsDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        return self.texts[idx], self.labels[idx]


def make_collate_fn(tokenizer: AutoTokenizer):
    def collate(batch: list[tuple[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(np.array(labels), dtype=torch.float)
        return encoded

    return collate


def load_train_val_split(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
    if TRAIN_SAMPLE_SIZE is not None:
        train_df = train_df.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_SEED)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_lora_model() -> AutoModelForSequenceClassification:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict_probs(model, tokenizer: AutoTokenizer, texts: list[str]) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), EVAL_BATCH_SIZE), desc="Validating"):
            batch = texts[start : start + EVAL_BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**encoded).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def evaluate(val_df: pd.DataFrame, probs: np.ndarray) -> float:
    preds = (probs >= CLASSIFICATION_THRESHOLD).astype(int)
    auc_scores = []
    for i, label in enumerate(LABEL_COLUMNS):
        auc = roc_auc_score(val_df[label], probs[:, i])
        accuracy = accuracy_score(val_df[label], preds[:, i])
        auc_scores.append(auc)
        print(f"{label:<15} AUC: {auc:.4f}   Accuracy: {accuracy:.4f}")
    mean_auc = float(np.mean(auc_scores))
    print(f"Mean column-wise AUC: {mean_auc:.4f}\n")
    return mean_auc


def main() -> None:
    train_df, val_df = load_train_val_split(TRAIN_CSV_PATH)
    print(f"Training on {len(train_df)} rows, validating on {len(val_df)} held-out rows from {TRAIN_CSV_PATH}")
    print(f"Device: {DEVICE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_lora_model().to(DEVICE)

    train_dataset = ToxicCommentsDataset(train_df["comment_text"].tolist(), train_df[LABEL_COLUMNS].to_numpy())
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    val_texts = val_df["comment_text"].tolist()
    best_mean_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train loss: {train_loss:.4f}\n")

        probs = predict_probs(model, tokenizer, val_texts)
        mean_auc = evaluate(val_df, probs)

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"New best mean AUC ({mean_auc:.4f}) -- saved to {OUTPUT_DIR}\n")

    print(f"Best mean column-wise AUC: {best_mean_auc:.4f}")


if __name__ == "__main__":
    main()




https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=test.csv.zip

```python
import kagglehub

# Download latest version
path = kagglehub.competition_download('jigsaw-toxic-comment-classification-challenge')

print("Path to competition files:", path)
```

## Kagglehub 
The API key is saved to the environment and Late submission is accepted. 

## File descriptions
- train.csv - the training set, contains comments with their binary labels
- test.csv - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
- sample_submission.csv - a sample submission file in the correct format
- test_labels.csv - labels for the test data; value of -1 indicates it was not used for scoring; (Note: file added after competition close!)


## Dataset Description
You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. 
The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment.

## Setup Learnings

- `kagglehub.competition_download` needs authentication even for public competitions. It reads the `KAGGLE_API_TOKEN` env var natively (via `kagglesdk`), so storing the token in `.env` and calling `load_dotenv()` before the `kagglehub` import in [test.py](test.py) is enough — no `~/.kaggle/kaggle.json` needed.
- Authentication alone isn't sufficient: Kaggle also returns `403 Forbidden` until you've accepted the competition rules on kaggle.com in-browser. That step can't be scripted.

## Label Distribution (train.csv)

This is a **multi-label** problem (each comment can have zero, one, or several labels), and the labels are heavily imbalanced:

| Label | Positive rate | Positive count |
|---|---|---|
| toxic | 9.58% | 15,294 |
| severe_toxic | 1.00% | 1,595 |
| obscene | 5.29% | 8,449 |
| threat | 0.30% | 478 |
| insult | 4.94% | 7,877 |
| identity_hate | 0.88% | 1,405 |

Out of 159,571 total comments. This imbalance matters for both metric choice and model evaluation (see below).

## Base Model Evaluation (no fine-tuning)

Both scripts score the same held-out slice of `train.csv` (2,000 rows, seed 42) so the numbers are directly comparable:

| Script | Model | Mean AUC | Mean Accuracy |
|---|---|---|---|
| [evaluate_detoxify.py](evaluate_detoxify.py) | Detoxify `original` (`unitary/toxic-bert`) | 0.9966 | 0.9908 |
| [evaluate_base.py](evaluate_base.py) | raw `bert-base-uncased` + random head | ~0.55–0.58 | ~0.56–0.62 |

**Detoxify's numbers are misleading, not "good":** `unitary/toxic-bert` was already trained on this exact Jigsaw dataset, so validating it against a slice of `train.csv` measures memorization, not generalization. It's not a fair no-fine-tuning baseline — it's closer to grading a model on its own training data.

**`bert-base-uncased` is the genuine "before fine-tuning" baseline:** this checkpoint has no toxicity classifier at all, so `AutoModelForSequenceClassification` attaches a freshly random-initialized linear head on load. Running inference (forward pass only, `torch.no_grad()`, no weight updates) gives mean AUC around 0.55–0.58, barely above the 0.5 random-guess floor, and scores shift noticeably between runs since the head reinitializes randomly each time — expected, since the encoder has language understanding but the classifier has never seen a labeled example.

**Metric pitfall:** accuracy is misleading on these imbalanced labels, in both directions. The untrained `bert-base-uncased` baseline showed ~99% accuracy on `severe_toxic` in one run purely because only ~1% of comments are `severe_toxic` — predicting "no" almost always scores well by chance, regardless of skill. **AUC (threshold-independent) is the metric that reflects real discriminative ability**, and it's also the original competition metric (mean column-wise ROC-AUC). Detoxify's high accuracy, by contrast, is trustworthy — its AUC is high too, and both are inflated for the same reason (it already learned this dataset), not a metric artifact.

## Fine-Tuning Results

[fine_tune_bert.py](fine_tune_bert.py) fine-tunes `bert-base-uncased` on 20,000 rows of `train.csv` with a multi-label head (`BCEWithLogitsLoss`, sigmoid output per label, `LEARNING_RATE=2e-5`, 2 epochs), training all ~109.5M parameters. [evaluate_ft_bert.py](evaluate_ft_bert.py) scores the resulting checkpoint (`fine_tuned_bert/`) on the same 2,000-row held-out slice used above:

| Label | AUC | Accuracy |
|---|---|---|
| toxic | 0.9861 | 0.9630 |
| severe_toxic | 0.9943 | 0.9930 |
| obscene | 0.9939 | 0.9830 |
| threat | 0.9634 | 0.9970 |
| insult | 0.9823 | 0.9775 |
| identity_hate | 0.9649 | 0.9905 |
| **Mean** | **0.9808** | **0.9840** |

A jump from ~0.55–0.58 mean AUC (untrained baseline) to 0.9808 — full fine-tuning gives the encoder's language understanding a real, learned mapping onto the six toxicity labels.

## Parameter-Efficient Fine-Tuning (LoRA)

[peft_base_bert.py](peft_base_bert.py) fine-tunes the same base model on the same data (identical split, sample size, epochs) but freezes the entire pretrained encoder and trains only LoRA adapters injected into the attention `query` and `value` projections (`r=8`, `lora_alpha=16`, `lora_dropout=0.1`, all 12 layers), plus the classification head. Q and V were chosen because that combination gave the best accuracy-per-trainable-parameter tradeoff in the original LoRA paper's ablations — Q reshapes *what's attended to*, V reshapes *what content flows through*, while K and the feed-forward layers stay frozen.

**Trainable parameters:** 299,526 out of 109,786,380 total (**0.27%** — roughly 365x fewer than full fine-tuning). Breakdown: 294,912 in the 24 LoRA-adapted Q/V matrices (12 layers × 2 modules × 12,288 params each) + 4,614 in the classification head.

[evaluate_peft_base_bert.py](evaluate_peft_base_bert.py) scores the resulting adapter (`peft_bert/`, merged onto the frozen base) on the same held-out slice:

| Label | AUC | Accuracy |
|---|---|---|
| toxic | 0.9800 | 0.9590 |
| severe_toxic | 0.9933 | 0.9925 |
| obscene | 0.9908 | 0.9785 |
| threat | 0.9357 | 0.9970 |
| insult | 0.9827 | 0.9710 |
| identity_hate | 0.9511 | 0.9905 |
| **Mean** | **0.9723** | **0.9814** |

## Full Fine-Tuning vs. LoRA

| | Trainable params | Checkpoint size | Mean AUC |
|---|---|---|---|
| Full fine-tuning ([fine_tune_bert.py](fine_tune_bert.py)) | 109,486,854 (100%) | ~420MB | 0.9808 |
| LoRA ([peft_base_bert.py](peft_base_bert.py)) | 299,526 (0.27%) | ~1.2MB (adapter only) | 0.9723 |

LoRA gives up only ~0.0085 mean AUC (0.87 percentage points) versus full fine-tuning, while training ~365x fewer parameters and producing a checkpoint ~350x smaller. The gap is largest on the rarest labels (`threat`: 0.9634 → 0.9357, `identity_hate`: 0.9649 → 0.9511) — with only 478 and 1,405 positive examples respectively in the full dataset, these labels benefit most from the extra capacity of full fine-tuning.

## Running the Scripts

### Setup

```bash
# Create the virtual environment (already done if .venv exists)
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Deactivate anytime with `deactivate`. All commands below assume the venv is active and the working directory is the project root.

### Data download

```bash
python test.py
```
Downloads the Jigsaw competition data via `kagglehub` (reads `KAGGLE_API_TOKEN` from `.env`). See [Setup Learnings](#setup-learnings) above — you must accept the competition rules on kaggle.com first.

### Baseline evaluation (no fine-tuning)

```bash
python evaluate_base.py       # raw bert-base-uncased + random head
python evaluate_detoxify.py   # Detoxify's pretrained unitary/toxic-bert
```

### Full fine-tuning

```bash
python fine_tune_bert.py       # trains bert-base-uncased, saves checkpoint to fine_tuned_bert/
python evaluate_ft_bert.py     # scores fine_tuned_bert/ on the held-out slice
```

### LoRA (parameter-efficient) fine-tuning

```bash
python peft_base_bert.py            # trains LoRA adapters, saves to peft_bert/
python evaluate_peft_base_bert.py   # scores peft_bert/ on the held-out slice
```

### Inference on new text

```bash
python inference_ft_bert.py "You are an idiot" "Have a wonderful day!"
```
Omit arguments to be prompted for a single line of text interactively. See [Labeling New Text](#labeling-new-text) below for details.

## Labeling New Text

[inference_ft_bert.py](inference_ft_bert.py) loads the fine-tuned checkpoint from `fine_tuned_bert/` (produced by [fine_tune_bert.py](fine_tune_bert.py)) and scores arbitrary text instead of a held-out CSV slice — useful for spot-checking the model on hand-written examples.

```bash
python inference_ft_bert.py "You are an idiot" "Have a wonderful day!"
```

Prints, per input, the sigmoid probability for each of the six labels and which ones cross the 0.5 threshold. Omit arguments to be prompted for a single line of text interactively.


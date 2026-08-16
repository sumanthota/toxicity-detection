


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

## Next Step

Fine-tune `bert-base-uncased` on `train.csv` with a multi-label head (`BCEWithLogitsLoss`, sigmoid output per label) and compare its AUC against this baseline to measure the real effect of fine-tuning.


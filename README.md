
## The Problem

Most sentiment/intent classifiers are trained on clean, formal English. Real support tickets in India look like:

> *"bhai mera order abhi tak nahi aaya, refund chahiye"*

Generic models misclassify this kind of code-mixed text far more often than they should. This project fine-tunes a model specifically suited to it, and rigorously benchmarks whether that effort was actually worth it compared to simpler alternatives.

## Categories

- `billing_issue`
- `technical_problem`
- `refund_request`
- `complaint`
- `general_query`

## Results

Three approaches were built and benchmarked head-to-head on the same held-out test set:

| Approach | Accuracy | Weighted F1 | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 79.0% | 0.80 | Fast, simple, weak on ambiguous categories |
| **Fine-tuned MuRIL** | **89.0%** | **0.89** | Best accuracy, fast local inference |
| Zero-shot LLM (Groq/Llama 3.1 8B) | 81.6% | -- | No training needed, but ~0.6s/prediction |

**Fine-tuning MuRIL outperformed both a classical ML baseline and zero-shot LLM prompting**, while also being faster and cheaper per-prediction than repeated API calls — a genuine, measured answer to "is fine-tuning worth it here?" rather than an assumed one.

### Confusion Matrix (fine-tuned model, test set)

Only 3 misclassifications out of 38 test examples. The errors that did occur were semantically reasonable — e.g. a `billing_issue` message classified as `refund_request`, categories that naturally overlap in real usage.

## Architecture

```
Raw messages (self-collected)
      │
      ▼
Data cleaning + stratified train/val/test split
      │
      ├──────────────┐
      ▼               ▼
Baseline model    Fine-tuned MuRIL
(TF-IDF + LogReg) (HuggingFace Trainer)
      │               │
      └──────┬────────┘
             ▼
   Evaluation + zero-shot LLM comparison
             │
             ▼
      FastAPI /predict endpoint
```

## Tech Stack

`Python` `HuggingFace Transformers` `PyTorch` `MuRIL` `scikit-learn` `FastAPI` `Groq API` `Google Colab (GPU training)`

## Project Structure

```
hinglish-ticket-classifier/
├── data/
│   ├── raw_messages.csv     # self-collected seed dataset
│   ├── train.csv / val.csv / test.csv
├── src/
│   ├── data_prep.py          # cleaning + stratified split
│   ├── baseline_model.py     # TF-IDF + Logistic Regression
│   ├── train_finetune.py     # MuRIL fine-tuning (run on Colab w/ GPU)
│   ├── evaluate.py           # test-set evaluation + confusion matrix
│   └── compare_zeroshot.py   # zero-shot LLM benchmark via Groq
├── app/
│   └── main.py                # FastAPI inference endpoint
├── models/finetuned/final/    # saved fine-tuned model + tokenizer
└── requirements.txt
```

## How It Works

1. **Data**: ~250 labeled Hinglish/Hindi/English support messages, self-written and expanded with LLM-assisted variation generation, then manually reviewed for quality — no pre-cleaned Kaggle dataset used.
2. **Baseline first**: TF-IDF + Logistic Regression established a benchmark before any deep learning was attempted.
3. **Fine-tuning**: MuRIL fine-tuned with the HuggingFace `Trainer` API — small learning rate (3e-5) and warmup to avoid catastrophic forgetting, with the best checkpoint (by F1) kept automatically across 15 epochs.
4. **Honest comparison**: Rather than assuming fine-tuning is "better," it was directly benchmarked against zero-shot prompting a general-purpose LLM on accuracy *and* latency.
5. **Deployment**: Wrapped in a FastAPI endpoint (`/predict`) returning category + confidence score.

## Running It

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/baseline_model.py
# fine-tuning: run src/train_finetune.py in Google Colab with GPU enabled
python src/evaluate.py
python src/compare_zeroshot.py
uvicorn app.main:app --reload
```

## Key Takeaway

With a relatively small (~250 example) self-collected dataset, fine-tuning a language-appropriate pretrained model still meaningfully outperformed both a classical baseline and a much larger general-purpose LLM used zero-shot — demonstrating that **for narrow, high-volume, repeated classification tasks, fine-tuning a small specialized model can beat prompting a bigger one**, both on accuracy and inference cost.

#fine-tuning the model ---- fine tuning ususally refers to taking a pre-trained model and training it on a new dataset for a specific task. In this case, we are fine-tuning a pre-trained transformer model for text classification.
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

train_df=pd.read_csv('data/train.csv')
test_df=pd.read_csv('data/val.csv')

num_labels = train_df['label_id'].nunique()

train_dataset = Dataset.from_pandas(train_df[["message", "label_id"]])
val_dataset = Dataset.from_pandas(val_df[["message", "label_id"]])

# MuRIL is Google's model pretrained specifically on 17 Indian languages
MODEL_NAME = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    # Converts raw text into token IDs the model understands
    # truncation=True cuts off text longer than max_length
    # padding="max_length" makes every example the same length (required for batching)
    return tokenizer(
        batch["message"], truncation=True, padding="max_length", max_length=64
    )

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)


def compute_metrics(eval_pred):
    # Called automatically after each evaluation step during training
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)  # pick the class with highest score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./models/finetuned",
    eval_strategy="epoch",       # evaluate after every epoch
    save_strategy="epoch",
    learning_rate=2e-5,          # small LR — standard for fine-tuning, avoids catastrophic forgetting
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,           # regularization to reduce overfitting
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the fine-tuned model + tokenizer for later use in FastAPI
trainer.save_model("./models/finetuned/final")
tokenizer.save_pretrained("./models/finetuned/final")
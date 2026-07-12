import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_PATH = "./models/finetuned/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # switches off dropout etc., needed for consistent inference

test_df = pd.read_csv("data/test.csv")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
    with torch.no_grad():  # disables gradient tracking — faster, less memory, we're not training
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()
    return pred_id, confidence

preds = []
for msg in test_df["message"]:
    pred_id, _ = predict(msg)
    preds.append(pred_id)

print("=== Fine-tuned Model Performance ===")
print(classification_report(test_df["label_id"], preds))
print("Confusion Matrix:\n", confusion_matrix(test_df["label_id"], preds))
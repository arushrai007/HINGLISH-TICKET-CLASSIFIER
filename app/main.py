from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Hinglish Support Ticket Classifier")

MODEL_PATH = "./models/finetuned/final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = {
    0: "billing_issue",
    1: "complaint",
    2: "general_query",
    3: "refund_request",
    4: "technical_problem",
}  # adjust order to match your actual label2id mapping from data_prep.py

class TicketRequest(BaseModel):
    message: str

@app.post("/predict")
def predict(request: TicketRequest):
    inputs = tokenizer(
        request.message, return_tensors="pt", truncation=True,
        max_length=64, padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()

    return {
        "category": id2label[pred_id],
        "confidence": round(confidence, 3)
    }
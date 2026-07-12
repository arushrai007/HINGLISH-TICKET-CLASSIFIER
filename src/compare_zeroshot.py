import pandas as pd
import time
from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")

test_df = pd.read_csv("data/test.csv")
categories = ["billing_issue", "technical_problem", "refund_request", "complaint", "general_query"]

def classify_zeroshot(message):
    prompt = f"""Classify this customer support message into exactly one category: {categories}.
Message: "{message}"
Respond with ONLY the category name, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

correct = 0
start_time = time.time()

for _, row in test_df.iterrows():
    pred = classify_zeroshot(row["message"])
    if pred == row["label"]:
        correct += 1

elapsed = time.time() - start_time
accuracy = correct / len(test_df)

print(f"Zero-shot LLM Accuracy: {accuracy:.2%}")
print(f"Total time for {len(test_df)} predictions: {elapsed:.2f}s")
print(f"Avg latency per prediction: {elapsed/len(test_df):.3f}s")
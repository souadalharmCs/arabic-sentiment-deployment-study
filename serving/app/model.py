from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import time

MODEL_PATH = "./model/checkpoint-680"

print("MODEL_PATH =", MODEL_PATH)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

print("Model loaded successfully!")


def predict(text):

    start = time.perf_counter()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    inference_time = time.perf_counter() - start

    return prediction, inference_time
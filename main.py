from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json
import numpy as np
import os

app = FastAPI()

# 🚀 Model yükleme
MODEL_DIR = "model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 🏷️ Label Encoder yükle
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# 📚 Açıklama verisi (isteğe bağlı)
DESCRIPTION_PATH = os.path.join("data", "labels_with_description_v2.json")
label_descriptions = {}
if os.path.exists(DESCRIPTION_PATH):
    with open(DESCRIPTION_PATH, "r", encoding="utf-8") as f:
        label_descriptions = json.load(f)

class InputText(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "🍅 Domates Hastalık API'sine hoş geldiniz!"}

@app.post("/predict")
def predict(input: InputText):
    text = input.text

    # 🔢 Tokenizer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]

    top_indices = np.argsort(probs)[::-1][:3]
    top_3 = [
        {
            "label": id2label.get(idx, f"Unknown-{idx}"),
            "confidence": round(float(probs[idx]), 3)
        }
        for idx in top_indices
    ]

    predicted_idx = int(np.argmax(probs))
    predicted_label = id2label.get(predicted_idx, f"Unknown-{predicted_idx}")
    confidence = float(probs[predicted_idx])
    description = label_descriptions.get(predicted_label, "Henüz açıklama eklenmedi.")
    warning = None

    if confidence < 0.6:
        warning = "⚠️ Bu tahmin düşük güven içeriyor. Daha fazla belirti girmeniz önerilir."

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 3),
        "top_3": top_3,
        "description": description,
        "warning": warning
    }

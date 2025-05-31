from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
from model.decode_model import decode_model

app = FastAPI()

# Eğer model dosyası yoksa, parçalardan oluştur
model_path = "model/model.safetensors"
if not os.path.exists(model_path):
    decode_model()

# Tokenizer ve model yükle
tokenizer = BertTokenizer.from_pretrained("model")
model = BertForSequenceClassification.from_pretrained(
    "model",
    num_labels=10,
    torch_dtype=torch.float32
)

# Tahmin isteği için veri modeli
class PredictionRequest(BaseModel):
    text: str

# Tahmin endpoint'i
@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4)
    }

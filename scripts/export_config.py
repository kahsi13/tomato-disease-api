# 🔧 label_encoder.pkl'den config.json üret

import pickle
import json
import os

# Label encoder'ı yükle
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Haritaları oluştur
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

# Model konfigürasyonu (HuggingFace uyumlu)
config = {
    "architectures": ["BertForSequenceClassification"],
    "num_labels": len(label_encoder.classes_),
    "id2label": id2label,
    "label2id": label2id,
    "model_type": "bert"
}

# Kaydet
config_path = os.path.join("model", "config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("✅ config.json başarıyla oluşturuldu.")

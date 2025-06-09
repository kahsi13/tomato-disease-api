import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
import pickle

# 📂 Yollar
DATA_PATH = "data/bert_dataset_final_enhanced_lowF1_boosted.csv"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
OUTPUT_DIR = "model"

# 📦 Veriyi yükle
df = pd.read_csv(DATA_PATH)
with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
df["label"] = label_encoder.transform(df["label"])

# 📊 Dataset hazırlığı
dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 🔧 Tokenizer ve model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_),
    id2label={i: c for i, c in enumerate(label_encoder.classes_)},
    label2id={c: i for i, c in enumerate(label_encoder.classes_)}
)

# 🧼 Tokenize et
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# 🎯 Değerlendirme metrikleri
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# 🧠 Eğitim ayarları
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,  # 🔼 100 epoch
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    save_total_limit=2,
)

# 🏋️ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 🔼 patience = 3
)

# 🚀 Eğitimi başlat
trainer.train()

# 💾 Kaydet
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

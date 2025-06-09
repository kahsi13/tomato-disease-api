# 🔧 Label Encoder oluşturma scripti

import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder

# CSV yolu (gerekirse değiştir)
csv_path = os.path.join("data", "bert_dataset_final_enhanced_lowF1_boosted.csv")
df = pd.read_csv(csv_path)

# Etiketleri dönüştür
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Encoder'ı kaydet
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Sınıfları yazdır
print(f"✅ {len(label_encoder.classes_)} sınıf bulundu.")
print("📄 Etiketler:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i}: {cls}")

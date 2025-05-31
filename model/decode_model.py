import base64
import requests
from pathlib import Path

def decode_model():
    model_path = Path("model/model.safetensors")
    if model_path.exists():
        print("✅ Model zaten mevcut.")
        return

    print("🌐 Google Drive'dan model indiriliyor...")

    # Doğrudan dosyayı indirip metin olarak alma
    file_id = "1UuiTooFuNNxxXQ6hLotSUfYdfEk5NZOG"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception("❌ Model indirme başarısız oldu.")

    print("📦 Base64 çözümleniyor...")
    model_bytes = base64.b64decode(response.text)

    Path("model").mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(model_bytes)

    print("✅ Model başarıyla decode edildi ve kaydedildi.")

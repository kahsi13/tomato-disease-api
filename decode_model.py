import base64
from pathlib import Path

def decode_model():
    model_path = Path("model/model.safetensors")
    if model_path.exists():
        print("✅ model.safetensors zaten var.")
        return

    print("🔧 Parçalar birleştiriliyor...")
    chunk_dir = Path("model")
    chunk_files = sorted(chunk_dir.glob("safe_chunk_*.part"))

    if not chunk_files:
        raise FileNotFoundError("❌ Herhangi bir safe_chunk_*.part dosyası bulunamadı.")

    # Tüm parçaları birleştir
    full_base64 = ""
    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as f:
            full_base64 += f.read().decode("utf-8")

    print("🔄 Base64 decode işlemi yapılıyor...")
    model_bytes = base64.b64decode(full_base64)

    print("💾 model.safetensors yazılıyor...")
    with open(model_path, "wb") as f:
        f.write(model_bytes)

    print("✅ model.safetensors başarıyla oluşturuldu.")

# ✅ 1. Temel Python image
FROM python:3.10-slim

# ✅ 2. Çalışma dizini
WORKDIR /app

# ✅ 3. Gerekli dosyaları kopyala
COPY . /app

# ✅ 4. Paketleri yükle
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ 5. Sunucuyu başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

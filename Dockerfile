
# Menggunakan base image Python
FROM python:3.14-slim

# Menentukan working directory container
WORKDIR /app

# Mencegah pembuatan Python cache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Menginstal dependencies untuk inference
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Menyalin source code API
COPY api.py .

# Menyiapkan folder model dan database
RUN mkdir -p models runtime

# Mendokumentasikan port API
EXPOSE 8000

# Menjalankan FastAPI Server
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

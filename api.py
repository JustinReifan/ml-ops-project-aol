
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

import sqlite3
import json
import hashlib

from pathlib import Path
from fastapi import HTTPException


# STEP 1: Membuat aplikasi REST API

app = FastAPI(title="Online Shopping Prediction API")

# Membaca model sekali ketika aplikasi dimulai
model = joblib.load("models/best_model.joblib")

# Membuat folder penyimpanan database
Path("runtime").mkdir(exist_ok=True)

DB_PATH = "runtime/monitoring.db"

# Membuat identitas versi berdasarkan isi file model
model_version = hashlib.sha256(
    Path("models/best_model.joblib").read_bytes()
).hexdigest()

# Membuat tabel untuk menyimpan riwayat prediksi
with sqlite3.connect(DB_PATH) as conn:

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT,
            features TEXT,
            prediction INTEGER,
            probability REAL,
            actual INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


# STEP 2: Menentukan struktur dan validasi input

class VisitorData(BaseModel):

    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0)

    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)

    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)

    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    SpecialDay: float = Field(ge=0, le=1)

    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

# Struktur input untuk hasil pembelian aktual
class FeedbackData(BaseModel):
    actual: int = Field(ge=0, le=1)


# STEP 3: Endpoint untuk memeriksa status API

@app.get("/health")
def health():
    return {"status": "healthy"}


# STEP 4: Endpoint untuk melakukan prediksi

@app.post("/predict")
def predict(data: VisitorData):

    # Mengubah request menjadi DataFrame
    X = pd.DataFrame([data.model_dump()])

    # Melakukan prediksi pembelian
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    # Menyimpan input dan hasil prediksi ke database
    with sqlite3.connect(DB_PATH) as conn:

        cursor = conn.execute("""
            INSERT INTO predictions
            (model_version, features, prediction, probability)
            VALUES (?, ?, ?, ?)
        """, (
            model_version,
            json.dumps(data.model_dump()),
            prediction,
            probability
        ))

        prediction_id = cursor.lastrowid

    # Mengembalikan hasil prediksi kepada aplikasi
    return {
        "prediction_id": prediction_id,
        "prediction": prediction,
        "purchase_probability": round(probability, 4)
    }


# Endpoint untuk menerima ground truth, hasil feedback dari prediction
@app.post("/feedback/{prediction_id}")
def feedback(prediction_id: int, data: FeedbackData):

    # Memperbarui hasil pembelian aktual
    with sqlite3.connect(DB_PATH) as conn:

        cursor = conn.execute("""
            UPDATE predictions
            SET actual = ?
            WHERE id = ?
        """, (data.actual, prediction_id))

        # Memastikan prediction ID tersedia
        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found"
            )

    return {"message": "Feedback saved successfully"}
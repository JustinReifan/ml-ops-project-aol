
import sqlite3
import json
import joblib
import pandas as pd

from pathlib import Path
import hashlib

from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, average_precision_score

from preprocess import X_train, X_val, y_val


# STEP 1: Membaca model dan menentukan threshold
model_path = Path("models/best_model.joblib")
model = joblib.load(model_path)

model_version = hashlib.sha256(
    model_path.read_bytes()
).hexdigest()

# Membaca reference data terbaru jika model sudah pernah diperbarui
reference_path = Path(f"models/reference_{model_version}.csv")

if reference_path.exists():
    reference_data = pd.read_csv(
        reference_path, float_precision="round_trip"
    )

    X_train = reference_data.drop(columns=["Revenue"])

DRIFT_THRESHOLD = 0.20
PERFORMANCE_THRESHOLD = 0.10


# STEP 2: Membaca 200 prediksi terakhir dari database

with sqlite3.connect("runtime/monitoring.db") as conn:

    logs = pd.read_sql_query("""
        SELECT * FROM predictions
        WHERE model_version = ?
        ORDER BY id DESC
        LIMIT 200
    """, conn, params=(model_version,))


# Memastikan jumlah data cukup untuk monitoring
if len(logs) < 100:
    raise ValueError("Monitoring requires at least 100 predictions")

# Mengubah JSON menjadi DataFrame
current_data = pd.DataFrame(
    logs["features"].apply(json.loads).tolist()
)


# STEP 3: DATA DRIFT

# Menentukan fitur kategorikal
categorical_features = [
    "Month", "OperatingSystems", "Browser",
    "Region", "TrafficType", "VisitorType", "Weekend"
]

drifted_features = []

print("\n=== DATA DRIFT ===")

for column in X_train.columns:

    if column in categorical_features:

        # Menghitung distribusi setiap kategori
        reference = X_train[column].value_counts(normalize=True)
        current = current_data[column].value_counts(normalize=True)

        # Menghitung Total Variation Distance
        score = reference.sub(
            current, fill_value=0
        ).abs().sum() / 2

    else:

        # Menghitung KS Statistic untuk fitur numerik
        score = ks_2samp(
            X_train[column],
            current_data[column]
        ).statistic

    # Menentukan apakah terjadi drift
    if score >= DRIFT_THRESHOLD:
        drifted_features.append(column)
        status = "DRIFT"
    else:
        status = "NORMAL"

    print(column, round(score, 4), status)


# STEP 4: PREDICTION DRIFT

# Menghasilkan probabilitas referensi dari validation data
reference_proba = model.predict_proba(X_val)[:, 1]

# Membandingkan distribusi probabilitas prediksi
prediction_drift = float(ks_2samp(
    reference_proba,
    logs["probability"]
).statistic)

print("\n=== PREDICTION DRIFT ===")
print("Drift Score:", round(prediction_drift, 4))
print("Drift Detected:", prediction_drift >= DRIFT_THRESHOLD)


# STEP 5: MODEL PERFORMANCE MONITORING

# Mengambil prediksi yang sudah memiliki ground truth
labeled = logs.dropna(subset=["actual"])

performance_degraded = None

print("\n=== MODEL PERFORMANCE ===")

# Evaluasi hanya jika jumlah label cukup dan kedua kelas tersedia
if len(labeled) >= 100 and labeled["actual"].nunique() == 2:

    # Menghitung performa referensi
    reference_ap = average_precision_score(
        y_val, reference_proba
    )

    # Menghitung performa berdasarkan traffic terbaru
    current_ap = average_precision_score(
        labeled["actual"], labeled["probability"]
    )

    current_f1 = f1_score(
        labeled["actual"], labeled["prediction"],
        zero_division=0
    )

    # Menghitung penurunan Average Precision
    performance_drop = reference_ap - current_ap

    performance_degraded = bool(
        performance_drop >= PERFORMANCE_THRESHOLD
    )

    print("Reference AP:", round(reference_ap, 4))
    print("Current AP:", round(current_ap, 4))
    print("Current F1:", round(current_f1, 4))
    print("Performance Drop:", round(performance_drop, 4))
    print("Degradation Detected:", performance_degraded)

else:
    print("Not enough labeled data for evaluation")


# STEP 6: MENYIMPAN MONITORING REPORT

report = {
    "model_version": model_version,
    "last_prediction_id": int(logs["id"].max()),
    "total_predictions": len(logs),
    "labeled_predictions": len(labeled),
    "drifted_features": drifted_features,
    "prediction_drift_score": prediction_drift,
    "prediction_drift_detected": prediction_drift >= DRIFT_THRESHOLD,
    "performance_degraded": performance_degraded
}

# Menyimpan hasil monitoring untuk proses retraining
Path("outputs").mkdir(exist_ok=True)

with open("outputs/monitoring_report.json", "w") as file:
    json.dump(report, file, indent=4)

print("\nMonitoring completed!")

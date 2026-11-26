import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# Mengambil dataset dan preprocessor dari preprocess.py
from preprocess import (
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    preprocessor
)


# STEP 0: KONFIGURASI MLFLOW

# Menghubungkan Python dengan MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Membuat atau menggunakan experiment yang sudah ada
mlflow.set_experiment("Online Shopping Prediction")

# True untuk final testing dan menyimpan model
# False jika masih melakukan eksperimen hyperparameter
FINALIZE_MODEL = True

# Mengatur apakah final testing perlu dijalankan
RUN_FINAL_TEST = os.getenv("RUN_FINAL_TEST", "true") == "true"


# STEP 1: MENENTUKAN MODEL

# Menentukan tiga model yang akan dibandingkan
models = {
    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
}

# Menyimpan hasil evaluasi seluruh model
results = []

# Menyimpan model dengan performa terbaik
best_model = None
best_name = ""
best_score = -1


# STEP 2: TRAINING SETIAP MODEL

for name, model in models.items():

    # Membuat satu MLflow run untuk setiap model
    with mlflow.start_run(run_name=name):

        print("\n================================")
        print("Training:", name)
        print("================================")

        # Menggabungkan preprocessing dan model
        # Clone membuat preprocessor terpisah untuk setiap model
        pipeline = Pipeline([
            ("preprocessing", clone(preprocessor)),
            ("model", model)
        ])

        # Melatih preprocessing dan model
        pipeline.fit(X_train, y_train)


        # STEP 3: MELAKUKAN PREDIKSI

        # Menghasilkan prediksi kelas 0 atau 1
        y_pred = pipeline.predict(X_val)

        # Mengambil estimasi probabilitas pembelian
        y_proba = pipeline.predict_proba(X_val)[:, 1]


        # STEP 4: MENGHITUNG METRIK EVALUASI

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(
                y_val, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, y_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_val, y_pred, zero_division=0
            ),
            "average_precision": average_precision_score(
                y_val, y_proba
            )
        }


        # STEP 5: EXPERIMENT TRACKING

        # Mencatat seluruh hyperparameter model
        mlflow.log_params(model.get_params())

        # Mencatat hasil evaluasi validation dataset
        mlflow.log_metrics(metrics)

        # Menyimpan model beserta preprocessing ke MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )


        # STEP 6: MENAMPILKAN HASIL EVALUASI

        # Menampilkan setiap metrik evaluasi
        for metric, value in metrics.items():
            print(metric, ":", round(value, 4))

        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

        # Menyimpan hasil untuk tabel perbandingan
        results.append({
            "Model": name,
            **metrics
        })


        # STEP 7: MODEL SELECTION

        # Mengambil skor Average Precision model saat ini
        ap = metrics["average_precision"]

        # Memperbarui model terbaik jika skornya lebih tinggi
        if ap > best_score:
            best_score = ap
            best_model = pipeline
            best_name = name


# STEP 8: MODEL COMPARISON

# Mengubah hasil evaluasi menjadi DataFrame
results_df = pd.DataFrame(results)

# Mengurutkan model berdasarkan Average Precision tertinggi
results_df = results_df.sort_values(
    by="average_precision",
    ascending=False
)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))


# STEP 9: MENAMPILKAN MODEL TERBAIK

print("\n=== BEST MODEL ===")
print("Model:", best_name)
print("Average Precision:", round(best_score, 4))


# Final testing hanya dilakukan jika eksperimen sudah selesai

# STEP 10: FINAL TESTING

if FINALIZE_MODEL and RUN_FINAL_TEST:

    # Menguji model terbaik menggunakan testing dataset
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n=== FINAL TESTING ===")
    print("Model:", best_name)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
    print("Average Precision:", average_precision_score(y_test, y_proba))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# STEP 11: MODEL SAVING

if FINALIZE_MODEL:

    # Membuat folder models jika belum tersedia
    Path("models").mkdir(exist_ok=True)

    # Menyimpan model terbaik beserta preprocessing
    joblib.dump(best_model, "models/best_model.joblib")

    print("\n=== MODEL SAVING ===")
    print("Best model saved successfully!")





print("\nAll models trained successfully!")

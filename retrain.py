
import json
import sqlite3
import hashlib
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from preprocess import X_train, y_train, X_val, y_val


def retrain():

    # STEP 1: Membaca hasil monitoring
    with open("outputs/monitoring_report.json") as file:
        report = json.load(file)

    # Memeriksa apakah ada indikasi masalah
    need_retraining = (
        report["drifted_features"]
        or report["prediction_drift_detected"]
        or report["performance_degraded"]
    )

    if not need_retraining:
        print("Model normal. Retraining tidak diperlukan.")
        return


    # STEP 2: Membaca model yang sedang digunakan
    model_path = Path("models/best_model.joblib")
    champion = joblib.load(model_path)

    model_version = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    # Memastikan laporan monitoring berasal dari model aktif
    if report["model_version"] != model_version:
        raise ValueError("Monitoring report is outdated")


    # STEP 3: Membaca data training sebelumnya
    reference_path = Path(
        f"models/reference_{model_version}.csv"
    )

    if reference_path.exists():
        old_data = pd.read_csv(
            reference_path, float_precision="round_trip"
        )
    else:
        old_data = X_train.copy()
        old_data["Revenue"] = y_train

    features = champion.feature_names_in_.tolist()


    # STEP 4: Mengambil data pengunjung terbaru
    with sqlite3.connect("monitoring.db") as conn:

        logs = pd.read_sql_query("""
            SELECT * FROM predictions
            WHERE model_version = ? AND id <= ?
            ORDER BY id DESC
            LIMIT 200
        """, conn, params=(
            model_version,
            report["last_prediction_id"]
        ))

    # Hanya menggunakan data yang memiliki ground truth
    logs = logs.dropna(subset=["actual"])

    # Mengubah JSON menjadi DataFrame
    new_data = pd.DataFrame(
        logs["features"].apply(json.loads).tolist(),
        columns=features
    )

    new_data["Revenue"] = logs["actual"].astype(int).values


    # STEP 5: Memeriksa kualitas data baru

    # Menghindari penggunaan pola fitur identik berulang kali
    new_data = new_data.drop_duplicates(subset=features)

    known_data = pd.concat([
        old_data[features],
        X_val[features]
    ])

    # Membaca data yang pernah digunakan dalam retraining
    seen_path = Path("outputs/retraining_seen.csv")

    if seen_path.exists():
        seen_data = pd.read_csv(
            seen_path, float_precision="round_trip"
        )
        known_data = pd.concat([known_data, seen_data])

    # Menghapus pola data yang sudah pernah digunakan
    known_hash = pd.util.hash_pandas_object(
        known_data, index=False
    )

    new_hash = pd.util.hash_pandas_object(
        new_data[features], index=False
    )

    new_data = new_data[~new_hash.isin(known_hash)]

    print("\n=== RETRAINING DATA CHECK ===")
    print("Total logged samples:", len(logs))
    print("Eligible new samples:", len(new_data))
    print("Target distribution:")
    print(new_data["Revenue"].value_counts())

    # Memastikan data cukup dan kedua kelas tersedia
    if (
        len(new_data) < 100
        or new_data["Revenue"].nunique() != 2
        or new_data["Revenue"].value_counts().min() < 10
    ):
        print("Data baru belum cukup untuk retraining.")
        return


    # STEP 6: Membagi data terbaru
    new_train, new_val = train_test_split(
        new_data,
        test_size=0.3,
        random_state=42,
        stratify=new_data["Revenue"]
    )

    # Menggabungkan data training lama dan baru
    updated_train = pd.concat(
        [old_data, new_train],
        ignore_index=True
    )

    X = updated_train[features]
    y = updated_train["Revenue"]


    # STEP 7: Melatih model kandidat

    # Clone membuat model baru dengan konfigurasi yang sama
    challenger = clone(champion)

    # Melatih ulang menggunakan data lama dan data baru
    challenger.fit(X, y)


    # STEP 8: Membandingkan performa kedua model
    X_eval = new_val[features]
    y_eval = new_val["Revenue"]

    champion_ap = average_precision_score(
        y_eval, champion.predict_proba(X_eval)[:, 1]
    )

    challenger_ap = average_precision_score(
        y_eval, challenger.predict_proba(X_eval)[:, 1]
    )

    # Memeriksa agar performa pada validation lama tidak turun jauh
    old_reference_ap = average_precision_score(
        y_val, champion.predict_proba(X_val)[:, 1]
    )

    new_reference_ap = average_precision_score(
        y_val, challenger.predict_proba(X_val)[:, 1]
    )

    print("\n=== RETRAINING RESULT ===")
    print("New training data:", len(new_train))
    print("New evaluation data:", len(new_val))
    print("Champion AP:", round(champion_ap, 4))
    print("Challenger AP:", round(challenger_ap, 4))

    # Model baru harus meningkat minimal 0.01 AP
    # dan tidak turun lebih dari 0.02 pada validation lama
    approved = (
        challenger_ap >= champion_ap + 0.01
        and new_reference_ap >= old_reference_ap - 0.02
    )


    # STEP 9: Mencatat hasil retraining ke MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Online Shopping Retraining")

    with mlflow.start_run(run_name="Challenger Model"):

        mlflow.log_param("champion_version", model_version)
        mlflow.log_param("new_training_samples", len(new_train))
        mlflow.log_param("total_training_samples", len(X))

        mlflow.log_metrics({
            "champion_ap": champion_ap,
            "challenger_ap": challenger_ap,
            "old_reference_ap": old_reference_ap,
            "new_reference_ap": new_reference_ap
        })

        mlflow.set_tag(
            "decision", "approved" if approved else "rejected"
        )

        mlflow.sklearn.log_model(
            sk_model=challenger,
            name="model",
            input_example=X.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )


    # Menyimpan pola data yang sudah digunakan
    # Agar tidak digunakan berulang pada eksperimen berikutnya
    new_data[features].to_csv(
        seen_path,
        mode="a",
        header=not seen_path.exists(),
        index=False
    )


    # STEP 10: Memperbarui model jika lolos evaluasi
    if approved:

        # Menyimpan model kandidat terlebih dahulu
        candidate_path = Path("models/candidate.joblib")
        joblib.dump(challenger, candidate_path)

        # Membuat identitas versi model baru
        new_version = hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest()

        # Menyimpan reference data untuk monitoring berikutnya
        updated_train.to_csv(
            f"models/reference_{new_version}.csv",
            index=False
        )

        # Mengganti model lama dengan model baru
        candidate_path.replace(model_path)

        print("Challenger approved. Model updated!")
        print("Restart API untuk menggunakan model baru.")

    else:
        print("Challenger rejected. Champion tetap digunakan.")


if __name__ == "__main__":
    retrain()

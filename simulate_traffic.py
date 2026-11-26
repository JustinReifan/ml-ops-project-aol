
import requests

from preprocess import X_test, y_test

import pandas as pd
from pathlib import Path


# Alamat API yang menerima prediksi
API_URL = "http://127.0.0.1:8000"

# True untuk simulasi perubahan distribusi data
SIMULATE_DRIFT = True


# Memilih data untuk simulasi traffic
if SIMULATE_DRIFT:

    # Simulasi pengunjung yang didominasi bulan November
    visitors = X_test[X_test["Month"] == "Nov"]

else:

    # Menggunakan distribusi data secara umum
    visitors = X_test


# Membaca riwayat data yang sudah digunakan
seen_path = Path("outputs/retraining_seen.csv")

if seen_path.exists():

    seen_data = pd.read_csv(
        seen_path, float_precision="round_trip"
    )

    # Membuat identitas setiap pola data
    seen_hash = pd.util.hash_pandas_object(
        seen_data[visitors.columns], index=False
    )

    visitor_hash = pd.util.hash_pandas_object(
        visitors, index=False
    )

    # Hanya mengambil pengunjung yang belum pernah digunakan
    visitors = visitors[~visitor_hash.isin(seen_hash)]

print("Available new visitors:", len(visitors))

# Mengambil maksimal 200 sampel pengunjung
sample = visitors.sample(
    n=min(200, len(visitors)),
    random_state=42
)


# Mengirim data pengunjung satu per satu
for data, actual in zip(
    sample.to_dict("records"),
    y_test.loc[sample.index]
):

    # Mengirim request prediksi
    response = requests.post(
        f"{API_URL}/predict",
        json=data,
        timeout=10
    )

    response.raise_for_status()

    prediction_id = response.json()["prediction_id"]

    # Mengirim hasil pembelian yang sebenarnya
    response = requests.post(
        f"{API_URL}/feedback/{prediction_id}",
        json={"actual": int(actual)},
        timeout=10
    )

    response.raise_for_status()


print("Traffic simulation completed!")
print("Total visitors:", len(sample))

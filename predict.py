
import pandas as pd
import joblib


# STEP 1: Membaca model yang sudah dilatih
model = joblib.load("models/best_model.joblib")


# STEP 2: Membaca dataset
df = pd.read_csv("datasets/online_shoppers_intention.csv")

# Menghapus kolom yang tidak digunakan sebagai input
X = df.drop(columns=["Revenue", "PageValues"])

# Mengambil 5 baris sebagai contoh input prediksi
sample = X.head(5).copy()


# STEP 3: Melakukan prediksi

# Prediksi kelas membeli atau tidak membeli
predictions = model.predict(sample)

# Estimasi probabilitas pembelian
probabilities = model.predict_proba(sample)[:, 1]


# STEP 4: Menampilkan hasil prediksi

sample["Prediction"] = predictions
sample["PurchaseProbability"] = probabilities

print("\n=== PREDICTION RESULT ===")
print(sample[["Prediction", "PurchaseProbability"]])

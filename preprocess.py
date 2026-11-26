
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# STEP 1: DATA INGESTION

# Membaca dataset dari file CSV
df = pd.read_csv("datasets/online_shoppers_intention.csv")

# Mengubah target True/False menjadi 1/0
df["Revenue"] = df["Revenue"].astype(int)


# STEP 2: MEMISAHKAN FITUR DAN TARGET

# X adalah fitur yang digunakan untuk membuat prediksi
# PageValues dihapus untuk mengurangi risiko data leakage
X = df.drop(columns=["Revenue", "PageValues"])

# y adalah target yang ingin diprediksi
y = df["Revenue"]


# STEP 3: DATA SPLITTING

# Membagi dataset menjadi 70% training dan 30% temporary
# Stratify digunakan untuk menjaga proporsi kelas target
# Random state digunakan agar hasil pembagian konsisten

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Membagi temporary menjadi validation dan testing
# Masing-masing mendapatkan 15% dari keseluruhan dataset

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# STEP 4: MENENTUKAN JENIS FITUR

# Kolom kategorikal berisi kategori, bukan besaran angka
categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

# Kolom numerik berisi angka yang memiliki makna kuantitatif
numerical_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay"
]


# STEP 5: MEMBUAT PREPROCESSOR

# StandardScaler digunakan untuk menyesuaikan skala numerik
# OneHotEncoder digunakan untuk mengubah kategori menjadi angka
# handle_unknown mencegah error jika muncul kategori baru

preprocessor = ColumnTransformer([
    ("numerical", StandardScaler(), numerical_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])


# STEP 6: MENJALANKAN PREPROCESSING

# Fit hanya dilakukan pada training data
# Agar preprocessor tidak mempelajari data validation dan testing
X_train_processed = preprocessor.fit_transform(X_train)

# Menggunakan aturan preprocessing dari training data
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)


# STEP 7: MEMERIKSA OVERLAP DATA

# Membuat identitas berdasarkan isi setiap baris fitur
train_groups = set(pd.util.hash_pandas_object(X_train, index=False))
val_groups = set(pd.util.hash_pandas_object(X_val, index=False))
test_groups = set(pd.util.hash_pandas_object(X_test, index=False))

# Menghitung jumlah pola fitur identik antar dataset
print("\n=== DATA OVERLAP ===")
print("Train vs Validation:", len(train_groups & val_groups))
print("Train vs Test:", len(train_groups & test_groups))
print("Validation vs Test:", len(val_groups & test_groups))


# STEP 8: MENAMPILKAN HASIL

print("\n=== DATA SPLIT ===")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\n=== PURCHASE RATE (%) ===")
print("Training:", y_train.mean() * 100)
print("Validation:", y_val.mean() * 100)
print("Testing:", y_test.mean() * 100)

print("\n=== FEATURE TYPES ===")
print("Numerical:", numerical_features)
print("Categorical:", categorical_features)

print("\n=== BEFORE PREPROCESSING ===")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\n=== AFTER PREPROCESSING ===")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Testing:", X_test_processed.shape)

print("\n=== TRANSFORMED FEATURE NAMES ===")
print(preprocessor.get_feature_names_out())

print("\nPreprocessing completed!")

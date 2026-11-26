
import pandas as pd

# STEP 1: Data ingestion
df = pd.read_csv("datasets/online_shoppers_intention.csv")

# Mengubah target True/False menjadi 1/0
df["Revenue"] = df["Revenue"].astype(int)

# STEP 2: Melihat struktur dataset
print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== SAMPLE DATA ===")
print(df.head().to_string())

print("\n=== COLUMN TYPES ===")
print(df.dtypes.to_string())

# STEP 3: Memeriksa data kosong
print("\n=== MISSING VALUES ===")
print(df.isnull().sum().to_string())

# STEP 4: Distribusi label
print("\n=== PURCHASE DISTRIBUTION ===")
print(df["Revenue"].value_counts())

print("\n=== PURCHASE RATE (%) ===")
print(df["Revenue"].value_counts(normalize=True) * 100)

# STEP 5: Analisis jenis pengunjung
print("\n=== CONVERSION BY VISITOR TYPE ===")
print(
    df.groupby("VisitorType")["Revenue"]
    .agg(["count", "mean"])
    .rename(columns={"mean": "purchase_rate"})
)

# STEP 6: Analisis berdasarkan bulan
print("\n=== CONVERSION BY MONTH ===")
print(
    df.groupby("Month")["Revenue"]
    .agg(["count", "mean"])
    .rename(columns={"mean": "purchase_rate"})
)

# STEP 7: Periksa duplikat
print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())


import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression


# TEST 1: Memastikan dataset valid
def test_dataset():

    df = pd.read_csv("datasets/online_shoppers_intention.csv")

    # Dataset tidak boleh kosong
    assert len(df) > 0

    # Dataset harus memiliki target Revenue
    assert "Revenue" in df.columns

    # Dataset tidak boleh memiliki missing values
    assert df.isnull().sum().sum() == 0

    # Target hanya boleh berisi dua kelas
    assert set(df["Revenue"].unique()) == {0, 1}


# TEST 2: Memastikan fitur tidak mengandung data leakage
def test_features():

    from preprocess import X_train

    # Memastikan jumlah fitur sesuai
    assert X_train.shape[1] == 16

    # Target dan fitur yang berisiko leakage tidak boleh masuk
    assert "Revenue" not in X_train.columns
    assert "PageValues" not in X_train.columns


# TEST 3: Memastikan data splitting benar
def test_data_split():

    from preprocess import X_train, X_val, X_test

    # Memastikan setiap kelompok memiliki data
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Memastikan indeks baris tidak digunakan di dua kelompok
    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert len(set(X_train.index) & set(X_val.index)) == 0
    assert len(set(X_val.index) & set(X_test.index)) == 0


# TEST 4: Memastikan preprocessing berjalan
def test_preprocessing():

    from preprocess import X_train, X_val, preprocessor

    # Membuat salinan preprocessor untuk pengujian
    processor = clone(preprocessor)

    train_processed = processor.fit_transform(X_train)
    val_processed = processor.transform(X_val)

    # Jumlah baris tidak boleh berubah
    assert train_processed.shape[0] == len(X_train)

    # Jumlah fitur hasil transformasi harus konsisten
    assert train_processed.shape[1] == val_processed.shape[1]


# TEST 5: Memastikan model dapat melakukan prediksi
def test_model():

    from preprocess import X_train, X_val, y_train, preprocessor

    # Membuat pipeline model sederhana untuk pengujian
    model = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Melatih model
    model.fit(X_train, y_train)

    # Melakukan prediksi terhadap 10 sampel
    predictions = model.predict(X_val.head(10))
    probabilities = model.predict_proba(X_val.head(10))[:, 1]

    # Memastikan jumlah prediksi sesuai jumlah input
    assert len(predictions) == 10

    # Prediksi hanya boleh berupa 0 atau 1
    assert set(predictions).issubset({0, 1})

    # Probabilitas harus berada di antara 0 dan 1
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

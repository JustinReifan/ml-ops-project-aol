
import subprocess
import sys

from prefect import flow, task


# TASK 1: Exploratory Data Analysis
@task
def explore_data():

    # Menjalankan file explore_data.py
    subprocess.run(
        [sys.executable, "explore_data.py"],
        check=True
    )


# TASK 2: Preprocessing Check
@task
def preprocess_data():

    # Menjalankan file preprocess.py
    subprocess.run(
        [sys.executable, "preprocess.py"],
        check=True
    )


# TASK 3: Model Training dan Experiment Tracking
@task
def train_model():

    # Menjalankan training dan pencatatan MLflow
    subprocess.run(
        [sys.executable, "train.py"],
        check=True
    )


# TASK 4: Model Prediction
# @task
# def predict():

#     # Melakukan prediksi menggunakan model tersimpan
#     subprocess.run(
#         [sys.executable, "predict.py"],
#         check=True
#     )

# TASK: Automated Testing
@task
def run_tests():

    # Menjalankan seluruh pengujian menggunakan Pytest
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        check=True
    )


# FLOW: Mengatur urutan seluruh task
@flow(name="Online Shopping MLOps Pipeline")
def ml_pipeline():

    explore_data()
    preprocess_data()
    run_tests()
    train_model()
    # predict()


# Menjalankan pipeline
if __name__ == "__main__":
    ml_pipeline()

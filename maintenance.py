
import subprocess
import sys

from prefect import flow, task


# TASK 1: Memeriksa kondisi model
@task
def monitor_model():

    subprocess.run(
        [sys.executable, "monitor.py"],
        check=True
    )


# TASK 2: Melakukan retraining jika diperlukan
@task
def retrain_model():

    subprocess.run(
        [sys.executable, "retrain.py"],
        check=True
    )


# FLOW: Mengatur proses pemeliharaan model
@flow(name="Model Maintenance Pipeline")
def maintenance_pipeline():

    monitor_model()
    retrain_model()


# Menjalankan maintenance pipeline
if __name__ == "__main__":
    maintenance_pipeline()

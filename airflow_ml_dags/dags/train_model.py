import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _wait_for_files():
    if os.path.exists("data/processed/{{ ds }}/target.csv") and \
            os.path.exists("data/processed/{{ ds }}/data.csv"):
        return 0
    return 1


with DAG(
        "train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(5),
) as dag:

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
    )

    wait = PythonSensor(
        task_id="wait_for_files",
        python_callable=_wait_for_files,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    train_test_preparation = DockerOperator(
        image="airflow-train-test",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/processed/{{ ds }} --val-size 0.25",
        task_id="docker-airflow-train-test",
        depends_on_past=True,
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
    )

    train = DockerOperator(
        image="airflow-ml-train",
        command="--input-dir /data/raw/{{ ds }} --model-dir /data/model/{{ ds }}",
        task_id="docker-airflow-train",
        environment={'EXP_NAME': 'experiment', 'MODEL_NAME': 'model'},
        do_xcom_push=True,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
    )

    validate = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/processed/{{ ds }} --model-  --output-dir /data/predicted/{{ ds }}",
        task_id="docker-airflow-predict",
        environment={'EXP_NAME': 'experiment', 'MODEL_NAME': 'model'},
        do_xcom_push=True,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data", "D:/MADE/ml-in-prod/ngc436/logs:/logs"]
    )

    preprocess >> wait >> train_test_preparation >> train >> validate

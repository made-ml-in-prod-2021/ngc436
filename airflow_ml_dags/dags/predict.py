import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

EXP_NAME = "experiment_rf"
MODEL_NAME = "rf"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "generate_data",
        default_args=default_args,
        schedule_interval="@daily",
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

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/processed/{{ ds }} --model-dir /data/model/{{ ds }} --output-dir /data/predictions/{{ ds }}",
        task_id="docker-airflow-predict",
        environment={'EXP_NAME': EXP_NAME, 'MODEL_NAME': MODEL_NAME},
        do_xcom_push=True,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
    )

    preprocess >> predict

import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train_model",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    # TODO: rewrite dags
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
    )

    train_test_preparation = DockerOperator(
        image="airflow-train-test",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-train-test",
        do_xcom_push=False,
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
    )

    train = DockerOperator(
        image="airflow-ml-train",
        command="--input-dir /data/raw/{{ ds }} --output-dir /model/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/predicted/{{ ds }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=["/Users/mikhail.maryufich/PycharmProjects/airflow_examples/data:/data"]
    )

    download >> preprocess >> predict
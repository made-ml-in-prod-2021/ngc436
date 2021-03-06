import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago

DAG_NAME = "predict"

EXP_NAME = "experiment_rf"
MODEL_NAME = "rf"

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
        DAG_NAME,
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="preprocess"
                " --input-dir /data/raw/{{ ds }}"
                " --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        network_mode="bridge",
        api_version='auto',
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
        volumes=["/tmp/data:/data"]
    )

    wait = PythonSensor(
        task_id="wait_for_files",
        python_callable=_wait_for_files,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="predict "
                "--input-dir /data/processed/{{ ds }} "
                "--model-dir /data/model/{{ ds }} "
                "--output-dir /data/predictions/{{ ds }}",
        task_id="docker-airflow-predict",
        environment={'EXP_NAME': EXP_NAME, 'MODEL_NAME': MODEL_NAME},
        do_xcom_push=True,
        network_mode="bridge",
        api_version='auto',
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
        volumes=["/tmp/data:/data"]
    )

    preprocess >> wait >> predict

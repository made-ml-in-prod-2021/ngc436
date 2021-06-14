import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago

DAG_NAME = "train_model"

EXP_NAME = "experiment_rf"
MODEL_NAME = "rf"


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _wait_for_files(**kwargs):
    ds = kwargs["ds"]
    print(f"Checking path /data/processed/{ds}/target.csv")
    if os.path.exists(f"/data/processed/{ds}/target.csv") and \
            os.path.exists(f"/data/processed/{ds}/data.csv"):
        print("Cool! Paths found")
        return 1
    print("Not found the required paths")
    return 0


with DAG(
        DAG_NAME,
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(5),
) as dag:

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="preprocess "
                "--input-dir /data/raw/{{ ds }}"
                " --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
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

    train_test_preparation = DockerOperator(
        image="airflow-train-test",
        command="train-test "
                "--input-dir /data/processed/{{ ds }} "
                "--output-dir /data/processed/{{ ds }} "
                "--val-size 0.25",
        task_id="docker-airflow-train-test",
        do_xcom_push=False,
        api_version='auto',
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
        volumes=["/tmp/data:/data"]
    )

    train = DockerOperator(
        image="airflow-ml-train",
        command="train "
                "--input-dir /data/processed/{{ ds }} "
                "--output-model-dir /data/model/{{ ds }}",
        task_id="docker-airflow-train",
        environment={'EXP_NAME': EXP_NAME, 'MODEL_NAME': MODEL_NAME},
        network_mode="bridge",
        do_xcom_push=False,
        api_version='auto',
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["D:/MADE/ml-in-prod/ngc436/data:/data"]
        volumes=["/tmp/data:/data"]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="validate "
                "--input-dir /data/processed/{{ ds }} ",
        task_id="docker-airflow-validate",
        environment={'EXP_NAME': EXP_NAME, 'MODEL_NAME': MODEL_NAME},
        network_mode="bridge",
        do_xcom_push=False,
        api_version='auto',
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        # volumes=["D:/MADE/ml-in-prod/ngc436/data:/data", "D:/MADE/ml-in-prod/ngc436/logs:/logs"]
        volumes=["/tmp/data:/data", "/tmp/logs:/logs"]
    )

    preprocess >> wait >> train_test_preparation >> train >> validate

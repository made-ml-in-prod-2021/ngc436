from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

DAG_NAME = "generate_data"


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        DAG_NAME,
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    generate = DockerOperator(
        image="airflow-data-generation",
        command="generate /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        api_version='auto',
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=["/tmp/data:/data"]
    )

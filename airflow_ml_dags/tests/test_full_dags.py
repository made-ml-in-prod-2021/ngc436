import logging
import os
import shutil
import time

import pytest

from . import AirflowAPI

logger = logging.getLogger(os.path.basename(__file__))

pytest_plugins = ["docker_compose"]


@pytest.fixture(scope="session", autouse=True)
def initialize():
    data_path = "/tmp/data"
    logs_path = "/tmp/logs"

    for pth in [data_path, logs_path]:
        if os.path.exists(pth):
            shutil.rmtree(pth)
        os.makedirs(pth)

    yield

    for pth in [data_path, logs_path]:
        shutil.rmtree(pth)


@pytest.fixture(scope="session", autouse=True)
def get_airflow_api(session_scoped_container_getter):
    service = session_scoped_container_getter.get("webserver").network_info[0]
    api_url = f"http://localhost:{service.host_port}/"
    logging.warning(f"API URL: {api_url}")
    airflow_api = AirflowAPI(api_url)
    yield airflow_api


@pytest.mark.timeout(20)
def test_generator(get_airflow_api):
    airflow_api: AirflowAPI = get_airflow_api

    time.sleep(5)

    execution_date = "2021-06-11T12:00:00+00:00"
    dag_id = "generate_data"
    airflow_api.trigger_dag(dag_id, execution_date)
    is_running = True
    while is_running:
        is_running = airflow_api.is_dag_running(dag_id, execution_date)
        time.sleep(2)

    assert os.path.exists("/tmp/data/raw")

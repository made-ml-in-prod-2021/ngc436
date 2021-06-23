import logging
import os

import pytest
import sh
from airflow.providers.docker.operators.docker import DockerOperator


@pytest.fixture(scope="session", autouse=True)
def prepare_docker_images():
    # build images before the tests start
    # Note: comment it during debug due to Fork problems
    sh.docker_compose("build", no_rm=True)

    yield


def test_generator_task(tmpdir):
    tmp_gen_path = tmpdir.join("generator_data_raw")
    os.makedirs(tmp_gen_path)

    generate = DockerOperator(
        image="airflow-data-generation",
        command=f"{tmp_gen_path.strpath}",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        volumes=[f"{tmp_gen_path.strpath}:{tmp_gen_path.strpath}"]
    )

    generate.execute(context={})

    gen_data = os.path.join(tmp_gen_path, "data.csv")
    assert os.path.exists(gen_data) and os.path.isfile(gen_data), f"{gen_data} doesn't exist or is not file"
    gen_target = os.path.join(tmp_gen_path, "target.csv")
    assert os.path.exists(gen_target) and os.path.isfile(gen_target), f"{gen_target} doesn't exist or is not file"

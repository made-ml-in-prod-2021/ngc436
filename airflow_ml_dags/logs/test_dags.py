import glob
import os

import airflow
import pytest

from airflow.utils.dag_cycle_tester import test_cycle as check_for_cycle

# current directory should be set to the root of the project (e.g. airflow_ml_dags)
DAG_PATHS = glob.glob(os.path.join("dags", "*.py"))


@pytest.mark.parametrize("dag_path", DAG_PATHS)
def test_dag_integrity(dag_path):
    """Import DAG files and check for a valid DAG instance."""
    dag_name = os.path.basename(dag_path)
    module = _import_file(dag_name, dag_path)
    # Validate if there is at least 1 DAG object in the file
    dag_objects = [v for v in vars(module).values() if isinstance(v, airflow.models.DAG)]
    assert dag_objects
    # For every DAG object, test for cycles
    for dag in dag_objects:
        check_for_cycle(dag)


def _import_file(module_name, module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

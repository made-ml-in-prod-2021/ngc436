FROM airflow-ml-base:latest

COPY validate.py /validate.py

COPY logging.conf.yaml /logging.conf.yaml

ENTRYPOINT ["python", "validate.py"]
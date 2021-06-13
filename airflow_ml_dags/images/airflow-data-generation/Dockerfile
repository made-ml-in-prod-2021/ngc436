FROM airflow-ml-base:latest

COPY generate.py /generate.py

RUN pip install -r /build/requirements.txt
ENTRYPOINT ["python", "generate.py"]
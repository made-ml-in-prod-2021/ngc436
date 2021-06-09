FROM python:3.6
RUN mkdir /build/
COPY requirements.txt /build/requirements.txt

COPY generate.py /generate.py

RUN pip install -r /build/requirements.txt
ENTRYPOINT ["python", "generate.py"]
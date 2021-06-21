import _thread
import os
import random
import signal
import sys
import time
from threading import Thread

import requests
import uvicorn
from fastapi import FastAPI
from starlette.responses import Response
from starlette.status import HTTP_204_NO_CONTENT

app = FastAPI()

MODEL_FILE_ID = "1VrN3fFwtOKoh9xXufzC_sFwZ20Wz53QT"
SCALER_FILE_ID = "1aA-VKAjrEzXIYzAaotQDXDaKrl3mPD4w"
URL = "https://docs.google.com/uc?export=download"
MODELS_DIR = 'model'
CHUNK_SIZE = 32768


def handler(signum, frame):
    sys.exit(1)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
        return None


def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def download_file_from_google_drive(file_id, destination):
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


@app.on_event("startup")
def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_file_from_google_drive(MODEL_FILE_ID, destination=os.path.join(MODELS_DIR, "model.bin"))


@app.get("/healthz", status_code=204, response_class=Response)
def healthz():
    return Response(status_code=HTTP_204_NO_CONTENT)


@app.get("/readiness", status_code=204, response_class=Response)
def healthz():
    return Response(status_code=HTTP_204_NO_CONTENT)


# check by runnig  curl http://localhost:5000/predict
@app.get("/predict")
def make_prediction():
    return random.randint(1, 100)


def main():
    signal.signal(signal.SIGUSR1, handler)
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="debug")


def killer():
    if "EXEC_DURATION" in os.environ:
        time.sleep(int(os.environ["EXEC_DURATION"]))
        print("Sending signal to stop")
        os.kill(os.getpid(), signal.SIGUSR1)


if __name__ == "__main__":
    if "STARTUP_DELAY" in os.environ:
        time.sleep(int(os.environ["STARTUP_DELAY"]))
    thread = Thread(target=killer, name="killer")
    thread.start()

    main()

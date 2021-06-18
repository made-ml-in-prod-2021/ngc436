import os
from fastapi import FastAPI
import typing
import requests

app = FastAPI()

MODEL_FILE_ID = "1VrN3fFwtOKoh9xXufzC_sFwZ20Wz53QT"
SCALER_FILE_ID = "1aA-VKAjrEzXIYzAaotQDXDaKrl3mPD4w"
URL = "https://docs.google.com/uc?export=download"
MODELS_DIR = 'model'
CHUNK_SIZE = 32768


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


# TODO: data loading and preprocessing
@app.on_event("startup")
def download_models():
    os.makedirs(MODELS_DIR)
    download_file_from_google_drive()
    pass

def load_models():

    pass
    # return scaler, model

@app.get("/predict")
def make_prediction():
    return


download_models()

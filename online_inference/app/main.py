from fastapi import FastAPI
import typing

app = FastAPI()


@app.on_event("startup")
def download_models():
    pass


@app.get("/predict")
def make_prediction():
    return

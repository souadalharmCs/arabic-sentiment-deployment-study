from fastapi import FastAPI
from pydantic import BaseModel

from app.model import predict

app = FastAPI()


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def classify(data: TextRequest):

    prediction, inference_time = predict(data.text)

    return {
        "prediction": prediction,
        "inference_time": round(inference_time, 6)
    }
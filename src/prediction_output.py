from typing import List
from fastapi import FastAPI
from pydantic import BaseModel


class Mention(BaseModel):
    loc: str
    confidence: float
    score: float


class Prediction(BaseModel):
    key: str
    value: str
    mentions: List[Mention]


class Predictions(BaseModel):
    features: List[str]
    prediction: List[Prediction]


class PredictionOutput(BaseModel):
    time: str
    modelId: str
    predictions: List[Predictions]


app = FastAPI()

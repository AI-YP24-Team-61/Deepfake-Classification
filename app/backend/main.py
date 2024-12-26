from typing import Union, Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from http import HTTPStatus
from pydantic import BaseModel
import joblib


app = FastAPI(
    title="Team 61. Deepfake-Classification."
)

class Hyperparam(BaseModel):
    type_nn_pretrain: str | None = None
    count_fc: int
    end_activation_function: str
    type_loss: str
    lr: float
    C: float

class fit_request(BaseModel):
    hyperparameters: Hyperparam

class fit_response(BaseModel):
    model: None
    
class predict_request(BaseModel):
    hyperparameters: Hyperparam

class predict_response(BaseModel):
    results: List[float] # типа список вероятностей, а потом просто вернем argmax



@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/fit", response_model=fit_response, status_code=HTTPStatus.CREATED)
async def fit(request: fit_request):

    ...
    
@app.post("/predict", response_model=fit_response, status_code=HTTPStatus.CREATED)
async def fit(request: fit_request):
    ... 



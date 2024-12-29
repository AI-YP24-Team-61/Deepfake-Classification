from typing import Union, Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from http import HTTPStatus
from PIL import Image
from pydantic import BaseModel
import joblib
import torch


from model_preprocessing import CustomModel, data_pipeline, train_model, predict_real, make_eda

torch.manual_seed(0)

# Пространство с обученными моделями
models_fitted = {}
# Пространство инстансов моделей (хотел поместить их в models_fitted, но т.к. models_fitted помещается в response, то 'class type' (CustomModel) не сериализуется в JSON)
models_classes = {}
# Пространство инференса
models_inference = {}

app = FastAPI(
    title="Team 61. Deepfake-Classification."
)

class Hyperparam(BaseModel):
    type_nn_pretrain: str | None = 'ResNet18'
    end_activation_function: str | None = 'Sigmoid'
    batch_size: int = 128
    lr: float = 0.01
    C: float = 0.01

class Statistics(BaseModel):
    fake_cnt: float | int
    real_cnt: float | int
    avg_size: float | int
    min_size: float | int
    max_size: float | int
    mean_rgb: List[float|int]
    var_rgb: List[float|int]
    std_rgb: List[float|int]

# class load_data_request(BaseModel):
#     ...
# class load_data_response(BaseModel):
#     ...

class fit_request(BaseModel):
    id: str
    hyperparameters: Hyperparam

class fit_response(BaseModel):
    message: str | None = None

class set_request(BaseModel):
    id: str | None = None

class set_response(BaseModel):
    message: str | None = None
    
class predict_request(BaseModel):
    image_path: str = r"..\data\inference_image\3.jpg"


class predict_response(BaseModel):
    real_prob: float
    is_real: bool

class make_eda_request(BaseModel):
    _: None = None
class make_eda_response(BaseModel):
    train: Statistics
    test: Statistics




@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

# @app.post("/load_data", response_model=load_data_response, status_code=HTTPStatus.CREATED)
# async def load_data(request: load_data_request):
#     ...

@app.get("/models", response_model=List[Dict], status_code=HTTPStatus.CREATED)
async def models():
    return [models_fitted]

@app.post("/fit", response_model=fit_response, status_code=HTTPStatus.CREATED)
async def fit(request: fit_request):
    if request.id in models_fitted.keys():
        return {"message": f"Model with id '{request.id}' already exist. Enter another id."}
    else:
        data_dir = "D:\\1_Магистратура_ВШЭ\\1_AI_YP24_Team_61\\Deepfake-Classification\\app\\data\\cifake-real-and-ai-generated-synthetic-images"
        # Обучаем модель
        model = CustomModel(lr=request.hyperparameters.lr, 
                            weight_decay=request.hyperparameters.C)
        dataset_sizes, dataloaders_logreg = data_pipeline(data_dir=data_dir)
        model_inf, dict_stat, best_acc = train_model(model=model,
                                        id_model=request.id,
                                        dataloaders=dataloaders_logreg,
                                        dataset_sizes=dataset_sizes,
                                        num_epochs=1)
        models_fitted[request.id] = {
            'type_nn_pretrain': request.hyperparameters.type_nn_pretrain,
            'end_activation_function': request.hyperparameters.end_activation_function,
            'batch_size': request.hyperparameters.batch_size,
            'lr': request.hyperparameters.lr,
            'C': request.hyperparameters.C,
            'num_epochs': 1,
            'accuracy': float(best_acc)
        }
        models_classes[request.id] = model

        return {"message": f"Model '{request.id}' trained and saved"}
    
@app.post("/set", response_model=set_response, status_code=HTTPStatus.CREATED)
async def set(request: set_request):
    models_inference.clear()

    model_path = f"D:\\1_Магистратура_ВШЭ\\1_AI_YP24_Team_61\\Deepfake-Classification\\app\\backend\\model_weights\\{request.id}.pt"
    models_inference['id'] = request.id

    #Загружаем модель для инференса и добавляем ее в пространство инференса
    model_test = models_classes[request.id]
    model_test.load_state_dict(torch.load(model_path, weights_only=True))
    models_inference['model_inference'] = model_test

    return {"message": f"Model '{request.id}' is ready for inference"}
    
    
@app.post("/predict", response_model=predict_response, status_code=HTTPStatus.CREATED)
async def predict(request: predict_request):
    im = Image.open(request.image_path)
    model = models_inference['model_inference']
    #print('-'*10, model)
    pred = predict_real(im, model)
    return pred

@app.post("/eda", response_model=make_eda_response, status_code=HTTPStatus.CREATED)
async def eda(request: make_eda_request):
    return make_eda()
    
    



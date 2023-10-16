from contextlib import asynccontextmanager

import fastapi
import uvicorn
from fastapi import FastAPI

from .api_data import DataLoader
from .api_models import PredictionRequest
from .api_utils import base_exception_handler
from .api_service import predict
from .api_ml_model import MlModel

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Loads Data
#     DataLoader()
#     print(DataLoader.airlines())
#     yield

app = FastAPI(title='Airline delayed flight API')
app.add_exception_handler(Exception, base_exception_handler)


@app.on_event('startup')
async def startup_event():
    # Load Data
    DataLoader()
    # Load Model
    MlModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(prediction_params: PredictionRequest) -> dict:
    await predict(prediction_params)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")

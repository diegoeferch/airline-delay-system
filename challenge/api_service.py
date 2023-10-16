from .api_data import DataLoader
from .api_models import PredictionRequest
from .api_ml_model import MlModel


async def predict(params: PredictionRequest):
    await check_params(params)
    return await MlModel.predict_delay(params)


async def check_params(pred_request: PredictionRequest):
    for params in pred_request.flights:
        if params.month < 1 or params.month > 12:
            raise Exception

        if params.airline not in await DataLoader.airlines():
            raise Exception

        if params.flight_type not in await DataLoader.flight_types():
            raise Exception

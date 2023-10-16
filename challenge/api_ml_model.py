import os.path
import pickle
from pathlib import Path

import pandas as pd

from .api_models import PredictionRequest
from .model import DelayModel
from .api_data import DataLoader


class MlModel(object):
    _parent_path = str(Path(__file__).parent)
    _model_pickle_path = str(Path(_parent_path, 'model.pkl'))
    _model = None

    def __new__(cls, *args, **kwargs):
        if os.path.exists(cls._model_pickle_path):
            with open(cls._model_pickle_path, 'rb') as pkl_file:
                cls._model = pickle.load(pkl_file)
            print(f'Model loaded!: {type(cls._model)}')
        else:
            print('Initializing model...')
            cls._model = DelayModel()
            # Data for training
            features, target = cls._model.preprocess(
                data=DataLoader.data(),
                target_column="delay"
            )
            # Fitting
            cls._model.fit(
                features=features,
                target=target
            )
            with open(cls._model_pickle_path, 'wb') as pkl_file:
                pickle.dump(cls._model, pkl_file)

            print("Saved trained model!")

    @classmethod
    async def predict_delay(cls, params: PredictionRequest):
        df = pd.DataFrame(pd.json_normalize([pred.json(by_alias=True) for pred in params.flights]))
        return cls._model.predict(df)

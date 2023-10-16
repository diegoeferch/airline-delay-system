import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

from typing import Tuple, Union, List, Any



def scale(
        target: pd.DataFrame
) -> float:
    """
    The scale function calculates the scale factor for the XGBoost model's scale_pos_weight parameter.
    :param target:
    :return: Float
    """
    n_y0 = len(target[target['y'] == 0])
    n_y1 = len(target[target['y'] == 1])
    print(f"scale: {n_y0 / n_y1}")
    return n_y0 / n_y1


def get_min_diff(
        data: pd.DataFrame
) -> float:
    """
    Function that calculates the time difference in minutes between two dates in a pandas DataFrame.
    :param data:
    :return:
    """
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff


class DelayModel:
    TOP_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    THRESHOLD_IN_MINUTES = 15
    MODEL_LR = 0.01

    def __init__(
            self
    ):
        self._model = None  # Model should be saved in this attribute.
        self._scale = None  # Scale factor
        self._target = None # Target labels (just in case)

    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat(
            [pd.get_dummies(data['OPERA'], prefix='OPERA'),
             pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
             pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )[self.TOP_FEATURES]

        self._target = self._get_target(data)
        self._scale = scale(self._target)

        if target_column is not None:
            return features, self._target.rename(columns={"y": target_column})

        return features

    def fit(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._build_model()
        self._model.fit(features, target)

    def predict(
            self,
            features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        self._build_model()
        return self._model.predict(features).tolist()


    def _get_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        It calculates the target variable for the delay prediction model based on the time difference between two
        dates in the input data
        :param data:
        :return:
        """
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        target = np.where(data['min_diff'] > self.THRESHOLD_IN_MINUTES, 1, 0)
        return pd.DataFrame({"y": target})

    def _build_model(self) -> None:
        if self._model is None:
            self._model = xgb.XGBClassifier(random_state=1, learning_rate=self.MODEL_LR, scale_pos_weight=self._scale)

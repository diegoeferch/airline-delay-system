from pathlib import Path

import pandas as pd


class DataLoader(object):
    DATA_PATH = str(Path(Path(__file__).parent.parent, 'data', 'data.csv'))
    _data = None

    def __new__(cls, *args, **kwargs):
        if cls._data is None:
            print("Loading data...")
            cls._data = pd.read_csv(cls.DATA_PATH)
            print("Data loaded!")

    @classmethod
    def data(cls):
        return cls._data

    @classmethod
    async def airlines(cls):
        return list(cls._data['OPERA'].unique())

    @classmethod
    async def flight_types(cls):
        return list(cls._data['TIPOVUELO'].unique())

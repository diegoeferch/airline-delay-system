from typing import List

from pydantic import BaseModel, Field


class PredictionParams(BaseModel):
    airline: str = Field(..., alias="OPERA")
    flight_type: str = Field(..., alias="TIPOVUELO")
    month: int = Field(..., alias="MES")


class PredictionRequest(BaseModel):
    flights: List[PredictionParams]

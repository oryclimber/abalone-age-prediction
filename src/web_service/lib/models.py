from typing import Literal

from pydantic import BaseModel, Field


class AbaloneInput(BaseModel):
    """Physical measurements of one abalone."""

    sex: Literal["M", "F", "I"]
    length: float = Field(gt=0)
    diameter: float = Field(gt=0)
    height: float = Field(gt=0)
    whole_weight: float = Field(gt=0)
    shucked_weight: float = Field(gt=0)
    viscera_weight: float = Field(gt=0)
    shell_weight: float = Field(gt=0)


class PredictionOut(BaseModel):
    """Predicted number of rings and the corresponding age in years (rings + 1.5)."""

    predicted_rings: float
    predicted_age: float

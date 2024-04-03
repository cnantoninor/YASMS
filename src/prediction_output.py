from typing import List
from pydantic import BaseModel, ConfigDict


class Mention(BaseModel):
    """
    Represents a mention of a prediction in the input.

    Attributes:
    - loc: str (location of the mention. It can be a character range in the input text or a bounded region in an image.
        If the mention is about the entire text or region then the location value is "all")
    - confidence: float (confidence of the mention. It can be a probability or a score)
    """

    loc: str
    confidence: float


class Prediction(BaseModel):
    """
    Represents the prediction for a list of features.

    Attributes:
    - key: str (feature name)
    - value: any (predicted value can be a boolean for binary classification or a float for regression or a string for multi-class classification)
    - mentions: List[Mention] (optional list of mentions)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    value: any
    mentions: List[Mention]


class Predictions(BaseModel):
    """
    Represents the predictions for a set of features.

    Attributes:
    - featureNames: List[str] (list of feature names)
    - prediction: List[Prediction] (list of prediction objects)

    """

    featureNames: List[str]
    prediction: List[Prediction]


class PredictionOutput(BaseModel):
    """
    Represents the output of a prediction.

    Attributes:
    - timestamp: str (ISO 8601 format)
    - modelId: str (model instance identifier)
    - predictions: List[Predictions] (list of predictions)
    """

    timestamp: str
    modelId: str
    predictions: List[Predictions]

from typing import List
from pydantic import BaseModel


class Feature(BaseModel):
    name: str
    value: str


class PredictionInput(BaseModel):
    """
    Represents the input for a prediction.

    Attributes:
    - features: List[Feature] (list of features)
    """

    features: List[Feature]

    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    @property
    def feature_values(self) -> List[str]:
        return [f.value for f in self.features]

    def check_valid_features(self, training_feature_names: List[str]) -> None:

        # check if the features are the same as the training features using the features fields
        if set(self.feature_names) != set(training_feature_names):
            raise ValueError(
                f"Feature names provided for prediction do not match the training features: {set(self.feature_names)} != {set(training_feature_names)}"
            )
        # check if there is at least one feature and if the value is not empty for each feature
        if not all(self.feature_values):
            raise ValueError(
                f"Feature values cannot be empty: {self.model_dump_json()}"
            )


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

    key: str
    value: str
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

import unittest
from pydantic import ValidationError
from prediction_model import PredictionInput, Feature


class TestPredictionInput(unittest.TestCase):

    def test_prediction_input_initialization(self):
        # Test that the PredictionInput class initializes correctly with valid input
        prediction_input = PredictionInput(
            features=[Feature(name="text", value="Ciao ciccio!")]
        )
        self.assertEqual(
            prediction_input.model_dump(),
            {"features": [{"name": "text", "value": "Ciao ciccio!"}]},
        )

    def test_check_valid_features(self):
        # Test that the check_valid_features method raises a ValueError when the features provided do not match the training features
        prediction_input = PredictionInput(
            features=[Feature(name="text", value="Ciao ciccio!")]
        )
        with self.assertRaises(ValueError):
            prediction_input.check_valid_features(["not_text"])

    def test_check_valid_features_name_empty(self):
        # Test that the check_valid_features method raises a ValueError when the features provided do not match the training features
        prediction_input = PredictionInput(
            features=[Feature(name="", value="Ciao ciccio!")]
        )
        with self.assertRaises(ValueError):
            prediction_input.check_valid_features(["not_text"])

    def test_check_valid_features_value_empty(self):
        # Test that the check_valid_features method raises a ValueError when the features provided do not match the training features
        prediction_input = PredictionInput(features=[Feature(name="text", value="")])
        with self.assertRaises(ValueError):
            prediction_input.check_valid_features(["text"])

    def test_prediction_input_feature_names(self):
        # Test that the feature_names property returns the correct list of feature names
        prediction_input = PredictionInput(
            features=[
                Feature(name="text", value="Ciao ciccio!"),
                Feature(name="label", value="spam"),
            ]
        )
        self.assertEqual(prediction_input.feature_names, ["text", "label"])

    def test_prediction_input_feature_values(self):
        # Test that the feature_values property returns the correct list of feature values
        prediction_input = PredictionInput(
            features=[
                Feature(name="text", value="Ciao ciccio!"),
                Feature(name="label", value="spam"),
            ]
        )
        self.assertEqual(prediction_input.feature_values, ["Ciao ciccio!", "spam"])

    def test_prediction_input_feature_names_empty(self):
        # Test that the feature_names property returns an empty list when there are no features
        prediction_input = PredictionInput(features=[])
        self.assertEqual(prediction_input.feature_names, [])

    def test_prediction_input_feature_values_empty(self):
        # Test that the feature_values property returns an empty list when there are no features
        prediction_input = PredictionInput(features=[])
        self.assertEqual(prediction_input.feature_values, [])

    def test_prediction_input_no_features(self):
        # Test that the PredictionInput class raises a ValidationError when no features are provided
        with self.assertRaises(ValidationError):
            PredictionInput()

    def test_prediction_input_non_list_features(self):
        # Test that the PredictionInput class raises a ValidationError when features is not a list
        with self.assertRaises(ValidationError):
            PredictionInput(features="not a list")

    def test_prediction_input_non_feature_in_list(self):
        # Test that the PredictionInput class raises a ValidationError when the list contains non-Feature objects
        with self.assertRaises(ValidationError):
            PredictionInput(features=["not a feature"])

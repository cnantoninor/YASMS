from pathlib import Path
import unittest
from spam_classifier import SpamClassifierModelLogic
from model_instance import ModelInstance
from utils import import_class_from_string

from config import root_path


class TestUtils(unittest.TestCase):

    def test_import_class_from_string(self):
        self.assertEqual(
            import_class_from_string("spam_classifier.SpamClassifierModelLogic"),
            SpamClassifierModelLogic,
        )


def data_uploaded_mis_and_dir():
    data_uploaded_dir = test_data__data_uploaded_path.as_posix()
    mis = ModelInstance(data_uploaded_dir)
    return mis, data_uploaded_dir


def trained_ready_to_serve_mis_and_dir():
    ready_to_serve_dir = test_data__ready_to_serve_path.as_posix()
    mis = ModelInstance(ready_to_serve_dir)
    return mis, ready_to_serve_dir


def training_failed_mis_and_dir():
    training_failed_dir = test_data__training_failed_path.as_posix()
    mis = ModelInstance(training_failed_dir)
    return mis, training_failed_dir


def training_in_progress_mis_and_dir():
    training_in_progress_dir = test_data__training_in_progress_path.as_posix()
    mis = ModelInstance(training_in_progress_dir)
    return mis, training_in_progress_dir


test_data_path: Path = root_path / "test_data"
test_mdlc_data_path: Path = test_data_path / "MDLC"

test_data__data_uploaded_path: Path = (
    test_mdlc_data_path /
    "spam_classifier/test_model/test_project/DATA_UPLOADED")
test_data__training_in_progress_path: Path = (
    test_mdlc_data_path /
    "spam_classifier/test_model/test_project/TRAINING_IN_PROGRESS")
test_data__ready_to_serve_path: Path = (
    test_mdlc_data_path
    / "spam_classifier/test_model/test_project/TRAINED_READY_TO_SERVE"
)
test_data__training_failed_path: Path = (
    test_mdlc_data_path /
    "spam_classifier/test_model/test_project/TRAINING_FAILED")
test_data__invalid_path: Path = (
    test_mdlc_data_path / "spam_classifier/test_model/test_project/INVALID"
)


if __name__ == "__main__":
    unittest.main()

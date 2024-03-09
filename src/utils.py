from dataclasses import dataclass
from pathlib import Path
import sys
from config import Constants, test_data_path
from model_instance import ModelInstance

test_data__data_uploaded_path: Path = (
    test_data_path / "spam_classifier/test_model/test_project/DATA_UPLOADED"
)
test_data__training_in_progress_path: Path = (
    test_data_path / "spam_classifier/test_model/test_project/TRAINING_IN_PROGRESS"
)
test_data__ready_to_serve_path: Path = (
    test_data_path / "spam_classifier/test_model/test_project/TRAINED_READY_TO_SERVE"
)
test_data__training_failed_path: Path = (
    test_data_path / "spam_classifier/test_model/test_project/TRAINING_FAILED"
)
test_data__invalid_path: Path = (
    test_data_path / "spam_classifier/test_model/test_project/INVALID"
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


@dataclass
class UtilsData:
    test_environment = None


def is_test_environment():
    if UtilsData.test_environment is None:
        UtilsData.test_environment = False
        for module in sys.modules.values():
            if module.__name__ in ["unittest", "pytest"]:
                UtilsData.test_environment = True
    return UtilsData.test_environment


def check_valid_biz_task_model_pair(biz_task: str, model_type: str):
    task_model_pair = f"{biz_task}/{model_type}"
    valid_pairs = (
        Constants.VALID_BIZ_TASK_MODEL_PAIR
        if not is_test_environment()
        else Constants.VALID_BIZ_TASK_MODEL_PAIR_TEST
    )
    if not task_model_pair in valid_pairs:
        raise ValueError(
            f"Invalid business task model type pair: {task_model_pair}; Valid values: {Constants.VALID_BIZ_TASK_MODEL_PAIR}"
        )

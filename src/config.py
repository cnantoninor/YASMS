import logging
from logging.handlers import RotatingFileHandler
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure logging
LOG_FILE = "app.log"
LOG_ROTATION_INTERVAL = 10  # Number of log files before rotation

# Create a handler that rotates log files when they reach a certain size
log_handler = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=10)

LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s"
)
log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
log_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

src_path = os.path.dirname(os.path.abspath(__file__))
root_path: Path = Path(__file__).parent.parent
data_path: Path = root_path / "data"
train_data_path: Path = data_path / "train"
test_data_path: Path = root_path / "test_data"


@dataclass
class Paths:
    testnino1_classification_task_path: Path = (
        data_path / "wl_classif_testnino1/TESTNINO1.csv"
    )

    testnino1_classification_task_refined_path: Path = (
        data_path / "wl_classif_testnino1/TESTNINO1_without_StatoWorkflow_N.csv"
    )
    test_data__upload_train_data_csv_path: Path = (
        test_data_path / "upload_train_data.csv"
    )
    test_data__invalid_path: Path = (
        test_data_path / "spam_classifier/test_model/test_project/INVALID"
    )


@dataclass
class Constants:
    BIZ_TASK_SPAM = "spam_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "GradientBoostingClassifier"
    VALID_BIZ_TASK_MODEL_PAIR = [f"{BIZ_TASK_SPAM}/{GRADIENT_BOOSTING_CLASSIFIER}"]
    VALID_BIZ_TASK_MODEL_PAIR_TEST = [f"{BIZ_TASK_SPAM}/test_model"]
    MODEL_DATA_FILE = "model_data.csv"
    TRAINING_IN_PROGRESS_LOG = "training_in_progress.log"
    TRAINING_ERROR_LOG = "training_error.log"
    TRAINING_SUBDIR = "training"
    TRAINED_MODEL_FILE = "trained_model.pickle"


openai_api_key = os.getenv("OPENAI_API_KEY")

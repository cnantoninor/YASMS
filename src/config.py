import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",  # Include timestamps
    datefmt="%Y-%m-%d %H:%M:%S",  # Format of the timestamp
)


src_path = os.path.dirname(os.path.abspath(__file__))
root_path: Path = Path(__file__).parent.parent
data_path: Path = root_path / "data"
train_data_path: Path = data_path / "train"
test_data_path: Path = root_path / "test_data"


@dataclass
class Paths:
    mrau_weblive_project1: Path = (
        data_path / "wl_mrproject/mention_1305_2024-01-11-174801.csv"
    )
    mrau_weblive_project1_1k_mentions: Path = (
        data_path / "wl_mrproject/1k_mention_1305_2024-01-11-174801.csv"
    )

    mrau_weblive_project1_5k_mentions: Path = (
        data_path / "wl_mrproject/5k_mention_1305_2024-01-11-174801.csv"
    )

    testnino1_classification_task: Path = (
        data_path / "wl_classif_testnino1/TESTNINO1.csv"
    )

    testnino1_classification_task_refined: Path = (
        data_path / "wl_classif_testnino1/TESTNINO1_without_StatoWorkflow_N.csv"
    )
    test_data__upload_train_data_csv: Path = test_data_path / "upload_train_data.csv"


@dataclass
class Constants:
    model_spam_type = "spam_classifier"
    valid_model_types = [model_spam_type]


openai_api_key = os.getenv("OPENAI_API_KEY")

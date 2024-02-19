from enum import Enum
import os
from config import Constants


class ModelInstanceStateNames(Enum):
    DATA_UPLOADED = 1
    TRAINING_IN_PROGRESS = 2
    MODEL_TRAINED_READY_TO_SERVE = 3 # Final State
    MODEL_TRAINING_FAILED = 4 # Final State


class ModelInstanceState:

    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory {self.directory} not found")
        if not os.path.isdir(self.directory):
            raise NotADirectoryError(f"{self.directory} is not a directory")
        parts = self.directory.split(os.path.sep)
        if len(parts) < 3:
            raise ValueError(
                "Directory path must have at least three parts: {modelType}/{modelName}/{modelInstanceDate}"
            )
        self.__biz_task, self.__mod_type, self.__mod_instance_date = parts[-3:]
        self.__state = None

    def __determine_state(self):

        training_subdir = os.path.join(self.directory, Constants.TRAINING_SUBDIR)
        model_pickle_file = os.path.join(training_subdir, Constants.TRAINED_MODEL_FILE)
        training_error_file = os.path.join(
            training_subdir, Constants.TRAINING_ERROR_LOG
        )
        training_in_progress_file = os.path.join(
            training_subdir, Constants.TRAINING_IN_PROGRESS_LOG
        )

        if not os.path.exists(training_subdir) and os.path.exists(
            os.path.join(self.directory, Constants.MODEL_DATA_FILE)
        ):
            self.__state = ModelInstanceStateNames.DATA_UPLOADED
        elif os.path.exists(training_subdir) and os.path.exists(model_pickle_file):
            self.__state = ModelInstanceStateNames.MODEL_TRAINED_READY_TO_SERVE
        elif os.path.exists(training_subdir) and os.path.exists(training_error_file):
            self.__state = ModelInstanceStateNames.MODEL_TRAINING_FAILED
        elif os.path.exists(training_subdir) and os.path.exists(
            training_in_progress_file
        ):
            self.__state = ModelInstanceStateNames.TRAINING_IN_PROGRESS
            # todo add timed out state check
        else:
            directory_subtree = ""
            for root, dirs, files in os.walk(self.directory):
                directory_subtree += f"{root}\n"
                for file in files:
                    directory_subtree += f"  - {file}\n"
            raise ValueError(f"Could not determine state for {directory_subtree}")

    @property
    def type(self) -> str:
        return self.__biz_task

    @property
    def name(self) -> str:
        return self.__mod_type

    @property
    def instance_date(self) -> str:
        return self.__mod_instance_date

    @property
    def state(self) -> ModelInstanceStateNames:
        if self.__state is None:
            self.__determine_state()
        return self.__state

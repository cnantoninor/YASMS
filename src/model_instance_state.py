from enum import Enum
import os
from config import Constants
import logging


class ModelInstanceStateEnum(Enum):
    DATA_UPLOADED = 1
    TRAINING_IN_PROGRESS = 2
    TRAINED_READY_TO_SERVE = 3  # Final State
    TRAINING_FAILED = 4  # Final State


class ModelInstanceState:

    @staticmethod
    def from_train_directory(directory: str) -> list:
        model_instance_dirs = []
        model_instances = []
        for root, dirs, _ in os.walk(directory):
            if len(dirs) == 4:
                model_instance_dirs.append(root)
        for midir in model_instance_dirs:
            try:
                model_instances.append(ModelInstanceState(midir))
            except Exception as e:
                logging.error(
                    "Skipping dir `%s` due to error creating ModelInstanceState: %s",
                    midir,
                    e,
                )
        return model_instances

    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory {self.directory} not found")
        if not os.path.isdir(self.directory):
            raise NotADirectoryError(f"{self.directory} is not a directory")
        parts = self.directory.split(os.path.sep)
        if len(parts) < 4:
            raise ValueError(
                f"Passed directory path `{self.directory}` must have at least four parts: \
                    [businessTask]/[modelType]/[project]/[modelInstanceName]"
            )
        self.__biz_task, self.__mod_type, self.__project, self.__mod_instance = parts[
            -4:
        ]
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
            self.__state = ModelInstanceStateEnum.DATA_UPLOADED
        elif os.path.exists(training_subdir) and os.path.exists(model_pickle_file):
            self.__state = ModelInstanceStateEnum.TRAINED_READY_TO_SERVE
        elif os.path.exists(training_subdir) and os.path.exists(training_error_file):
            self.__state = ModelInstanceStateEnum.TRAINING_FAILED
        elif os.path.exists(training_subdir) and os.path.exists(
            training_in_progress_file
        ):
            self.__state = ModelInstanceStateEnum.TRAINING_IN_PROGRESS
            # todo add timed out state check
        else:
            directory_subtree = ""
            for root, _, files in os.walk(self.directory):
                directory_subtree += f"{root}\n"
                for file in files:
                    directory_subtree += f"  - {file}\n"
            raise ValueError(f"Could not determine state for {directory_subtree}")

    @property
    def task(self) -> str:
        return self.__biz_task

    @property
    def type(self) -> str:
        return self.__mod_type

    @property
    def instance(self) -> str:
        return self.__mod_instance

    @property
    def project(self) -> str:
        return self.__project

    @property
    def state(self) -> ModelInstanceStateEnum:
        if self.__state is None:
            self.__determine_state()
        return self.__state

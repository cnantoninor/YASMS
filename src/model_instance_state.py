from __future__ import annotations
import logging
from enum import Enum
import os
from config import Constants


class ModelInstanceStateEnum(Enum):
    DATA_UPLOADED = 1
    TRAINING_IN_PROGRESS = 2
    TRAINED_READY_TO_SERVE = 3  # Final State
    TRAINING_FAILED = 4  # Final State


class ModelInstanceState:

    @staticmethod
    def from_train_directory(root_dir: str) -> list[ModelInstanceState]:
        # check directory exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found")
        model_instances = []
        root_len = len(root_dir.split(os.path.sep))

        for subdir, _, _ in sorted(os.walk(root_dir), reverse=True):
            # if the difference between the number of parts in the root and
            # the number of parts in the directory is 4 then add it to the list
            subdirs = subdir.split(os.path.sep)
            if len(subdirs) - root_len == 4:
                try:
                    model_instances.append(ModelInstanceState(subdir))
                except Exception as e:
                    logging.error(
                        "Skipping dir `%s` due to error creating ModelInstanceState: %s",
                        subdir,
                        e,
                    )

        if len(model_instances) == 0:
            logging.warning("No model instances found in directory `%s`", root_dir)
        else:
            # concatenate the model instances into a string
            msg = ""
            for model_instance in model_instances:
                msg += (str(model_instance)) + "\n"
            logging.info(
                "Found %s model instances in directory `%s`:\n%s",
                len(model_instances),
                root_dir,
                msg,
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
        self.__determine_state()

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

    # Override the __str__ method to return a string representation of the object
    def __str__(self):
        return f"ModelInstanceState({self.__biz_task}, \
        {self.__mod_type}, {self.__project}, {self.__mod_instance}, {self.state})"

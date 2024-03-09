from __future__ import annotations
import logging
from enum import IntEnum
import os
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod

from pandas import DataFrame
from config import Constants
import json

from src import import_class_from_string


class ModelInstanceStateEnum(IntEnum):
    DATA_UPLOADED = 1
    TRAINING_IN_PROGRESS = 2
    TRAINED_READY_TO_SERVE = 3  # Final State
    TRAINING_FAILED = 4  # Final State


class ModelInterface(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def check_trainable(self):
        pass


class ModelInstance(ABC):

    @staticmethod
    def from_train_directory(root_dir: str) -> list[ModelInstance]:
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
                    model_instance = ModelInstance(subdir)
                    model_instances.append(model_instance)
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
        parts = Path(self.directory).parts
        if len(parts) < 4:
            raise ValueError(
                f"Passed directory path `{self.directory}` must have at least four parts: \
                    [businessTask]/[modelType]/[project]/[modelInstanceName]"
            )
        self.__biz_task, self.__mod_type, self.__project, self.__mod_instance = parts[
            -4:
        ]
        self.__features_fields = []
        self.__target_field = None
        self.__instance_logic = None
        self.__determine_state()

    def check_trainable(self):
        if (
            self.state != ModelInstanceStateEnum.DATA_UPLOADED
            and self.state != ModelInstanceStateEnum.TRAINING_IN_PROGRESS
        ):
            raise ValueError(f"Model instance `{self}` is not in a state to be trained")
        self.__logic.check_trainable()

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
            self.__load_features_and_target()
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

    def load_training_data(self) -> DataFrame:
        return pd.read_csv(self.directory + "/" + Constants.MODEL_DATA_FILE)

    def __load_features_and_target(self):
        features_fields_file = os.path.join(
            self.directory, Constants.FEATURES_FIELDS_FILE
        )
        # read the features fields file
        with open(features_fields_file, "r") as f:
            self.__features_fields = f.read().splitlines()
        target_field_file = os.path.join(self.directory, Constants.TARGET_FIELD_FILE)
        # read the target field file
        with open(target_field_file, "r") as f:
            self.__target_field = f.read()

    @staticmethod
    def snake_to_camel_case(snake_case_str: str) -> str:
        components = snake_case_str.split("_")
        return "".join(x.title() for x in components).strip()

    def predict(self):
        return self.__logic().predict()

    def train(self):
        return self.__logic().train()

    @property
    def __logic(self) -> ModelInterface:
        """
        Specific model logic instance for the model instance
        """
        if self.__instance_logic == None:
            camel_case_name = (
                f"{ModelInstance.snake_to_camel_case(self.task)}ModelLogic"
            )
            self.__instance_logic = import_class_from_string(
                f"{self.task}.{camel_case_name}"
            )(self)
        return self.__instance_logic

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

    @property
    def features_fields(self) -> list[str]:
        return self.__features_fields

    @property
    def target_field(self) -> str:
        return self.__target_field

    # Override the __str__ method to return a string representation of the object
    def __str__(self) -> str:
        return self.to_json()

    def to_json(self) -> str:
        data = {
            "task": self.task,
            "type": self.type,
            "project": self.project,
            "instance": self.instance,
            "state": self.state.name,
            "features": self.features_fields,
            "target": self.target_field,
        }
        return json.dumps(data)

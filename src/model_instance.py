from __future__ import annotations
from abc import ABC, abstractmethod
import json
import logging

from enum import IntEnum
import os
from pathlib import Path
import pickle
import traceback
import numpy
import pandas as pd

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from config import Constants

from utils import import_class_from_string

logger = logging.getLogger(__name__)


class ModelInstanceStateEnum(IntEnum):
    DATA_UPLOADED = 1
    TRAINING_IN_PROGRESS = 2
    TRAINED_READY_TO_SERVE = 3  # Final State
    TRAINING_FAILED = 4  # Final State


class ModelInterface(ABC):

    @abstractmethod
    def train(self) -> tuple[pd.DataFrame, numpy.ndarray, Pipeline, float, float]:
        """
        Train the model and return:
            - the metrics stats data frame
            - the confusion matrix
            - the trained model pipeline
            - the cross validation time
            - the fit time
        """

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def check_trainable(self) -> None:
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
        self.__directory = directory
        if not os.path.exists(self.__directory):
            raise FileNotFoundError(f"Directory {self.__directory} not found")
        if not os.path.isdir(self.__directory):
            raise NotADirectoryError(f"{self.__directory} is not a directory")
        parts = Path(self.__directory).parts
        if len(parts) < 4:
            raise ValueError(
                f"Passed directory path `{self.__directory}` must have at least four parts: \
                    [businessTask]/[modelType]/[project]/[modelInstanceName]"
            )
        self.__biz_task, self.__mod_type, self.__project, self.__mod_instance_name = (
            parts[-4:]
        )
        self.__features_fields = []
        self.__target_field = None
        self.__instance_logic = None
        self.__determine_state()

    def check_trainable(self):
        if not self.is_trainable():
            raise ValueError(f"Model instance `{self}` is not in a state to be trained")
        self.__logic.check_trainable()

    def is_trainable(self):
        return self.state in (
            ModelInstanceStateEnum.DATA_UPLOADED,
            ModelInstanceStateEnum.TRAINING_IN_PROGRESS,
        )

    def __determine_state(self):

        self.__training_subdir = os.path.join(
            self.__directory, Constants.TRAINING_SUBDIR
        )
        self.__model_pickle_file = os.path.join(
            self.__training_subdir, Constants.TRAINED_MODEL_FILE
        )
        self.__training_error_file = os.path.join(
            self.__training_subdir, Constants.TRAINING_ERROR_LOG
        )
        self.__training_in_progress_file = os.path.join(
            self.__training_subdir, Constants.TRAINING_IN_PROGRESS_LOG
        )

        if not os.path.exists(self.__training_subdir) and os.path.exists(
            os.path.join(self.__directory, Constants.MODEL_DATA_FILE)
        ):
            self.__state = ModelInstanceStateEnum.DATA_UPLOADED
        elif os.path.exists(self.__training_subdir) and os.path.exists(
            self.__model_pickle_file
        ):
            self.__state = ModelInstanceStateEnum.TRAINED_READY_TO_SERVE
        elif os.path.exists(self.__training_subdir) and os.path.exists(
            self.__training_error_file
        ):
            self.__state = ModelInstanceStateEnum.TRAINING_FAILED
        elif os.path.exists(self.__training_subdir) and os.path.exists(
            self.__training_in_progress_file
        ):
            self.__state = ModelInstanceStateEnum.TRAINING_IN_PROGRESS
        else:
            directory_subtree = ""
            for root, _, files in os.walk(self.__directory):
                directory_subtree += f"{root}\n"
                for file in files:
                    directory_subtree += f"  - {file}\n"
            raise ValueError(f"Could not determine state for {directory_subtree}")

        self.__load_features_and_target()

    def load_training_data(self) -> DataFrame:
        return pd.read_csv(self.__directory + "/" + Constants.MODEL_DATA_FILE)

    def __load_features_and_target(self):
        features_fields_file = os.path.join(
            self.__directory, Constants.FEATURES_FIELDS_FILE
        )
        # read the features fields file
        with open(features_fields_file, "r", encoding="utf8") as f:
            self.__features_fields = f.read().splitlines()
        target_field_file = os.path.join(self.__directory, Constants.TARGET_FIELD_FILE)
        # read the target field file
        with open(target_field_file, "r", encoding="utf8") as f:
            self.__target_field = f.read()

    @staticmethod
    def snake_to_camel_case(snake_case_str: str) -> str:
        components = snake_case_str.split("_")
        return "".join(x.title() for x in components).strip()

    def predict(self):
        return self.__logic.predict()

    def train(self) -> tuple[pd.DataFrame, numpy.ndarray, Pipeline, float, float]:
        # create training dir if not exists and training in progress file
        if not os.path.exists(self.__training_subdir):
            os.makedirs(self.__training_subdir)
        msg = f"Started Training the model instance:`{self.directory}`..."
        with open(self.__training_in_progress_file, "w", encoding="utf8") as f:
            f.write(msg)
        logger.info(msg)

        try:
            df_metrics, confusion_matrix, pipeline, cv_time, fit_time = (
                self.__logic.train()
            )

            # save the trained model to the model pickle file
            pickle.dump(pipeline, open(self.__model_pickle_file, "wb")

            df_metrics.to_csv(
                self.__training_subdir + "/metrics.stats", index=False, encoding="utf8"
            )
            # save the confusion matrix to a file
            numpy.savetxt(
                self.__training_subdir + "/confusion_matrix.stats",
                confusion_matrix,
                fmt="%d",
                delimiter=",",
            )

            timestats = f"Cross Validation Time: {cv_time} seconds; Fit Time: {fit_time} seconds"
            with open(
                self.__training_subdir + "/time.stats", "w", encoding="utf8"
            ) as f:
                f.write(timestats)

            logger.info(
                "Successfully Trained the model instance:`%s`. Model file is saved to `%s`",
                self.directory,
                self.__model_pickle_file,
            )
            # remove the training in progress file
            os.remove(self.__training_in_progress_file)

        except Exception as e:
            err_msg = traceback.format_exc()
            # write the error to the training error log file
            with open(self.__training_error_file, "w", encoding="utf8") as f:
                f.write(err_msg)
            logging.error("Error training model instance `%s`: %s", str(self), err_msg)
            raise e

    @property
    def __logic(self) -> ModelInterface:
        """
        Specific model logic instance for the model instance
        """
        if self.__instance_logic is None:
            camel_case_name = (
                f"{ModelInstance.snake_to_camel_case(self.task)}ModelLogic"
            )
            self.__instance_logic = import_class_from_string(
                f"{self.task}.{camel_case_name}"
            )(self)
        return self.__instance_logic

    @property
    def model_file(self) -> str:
        return self.__model_pickle_file

    @property
    def task(self) -> str:
        return self.__biz_task

    @property
    def type(self) -> str:
        return self.__mod_type

    @property
    def instance(self) -> str:
        return self.__mod_instance_name

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

    @property
    def directory(self) -> str:
        return self.__directory

    @property
    def training_error_file(self) -> str:
        return self.__training_error_file

    @property
    def training_subdir(self) -> str:
        return self.__training_subdir

    # Override the __str__ method to return a string representation of the object
    def __str__(self) -> str:
        return self.to_json()

    def to_json(self) -> str:
        data = {
            "task": self.task,
            "type": self.type,
            "project": self.project,
            "instance_name": self.instance,
            "state": self.state.name,
            "features": self.features_fields,
            "target": self.target_field,
        }
        return json.dumps(data)

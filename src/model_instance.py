from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import chain
import json
import logging

from enum import IntEnum
import os
from pathlib import Path
import pickle
import traceback
from typing import Dict, List
import numpy
import pandas as pd

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from config import Constants, data_path

from environment import is_test_environment
from prediction_model import PredictionInput, PredictionOutput
from utils import import_class_from_string
import time


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
    def check_trainable(self) -> None:
        pass

    @abstractmethod
    def predict(self, features: dict[str, any]) -> PredictionOutput:
        """
        Given the dictionary of features which must be the same of the train features,
        predict the output and return the PredictionOutput annotations.

        Args:
            features (dict[str, any]): The dictionary of features used for prediction

        Returns:
            PredictionOutput: The prediction output

        """


class _Models:

    def __init__(self, root_data_dir: str, debug: bool = False):
        # Dictionary of models for serving, i.e. project_name:str ->
        # model_instance : ModelInstance
        self._servable_dict: Dict[str, List[ModelInstance]] = {}
        # Dictionary of models for training, i.e. project_name:str ->
        # model_instance : ModelInstance
        self._trainable_dict: Dict[str, List[ModelInstance]] = {}
        # Dictionary of models in other states, i.e. project_name:str ->
        # model_instance : ModelInstance
        self._other_dict: Dict[str, List[ModelInstance]] = {}

        self._debug = debug

        self._populate(root_data_dir)
        self._root_data_dir = root_data_dir

    def __iter__(self):
        return chain(self._servable_dict.values(), self._trainable_dict.values())

    # size
    def __len__(self):
        # return the total number of models for each dict key sum the len of the value list
        tot = 0
        for _, v in self._servable_dict.items():
            tot += len(v)
        for _, v in self._trainable_dict.items():
            tot += len(v)
        for _, v in self._other_dict.items():
            tot += len(v)
        return tot

    def __str__(self) -> str:
        return json.dumps(self.to_json())

    def _add(self, model_instance: ModelInstance) -> None:
        model_type_id = model_instance.type_identifier
        if model_instance.is_servable():
            models_for_type = self._servable_dict.get(model_type_id, [])
            models_for_type.append(model_instance)
            self._servable_dict[model_type_id] = models_for_type
            logging.debug("Added model instance `%s` as servable", model_instance)
        elif model_instance.is_trainable():
            models_for_type = self._trainable_dict.get(model_type_id, [])
            models_for_type.append(model_instance)
            self._trainable_dict[model_type_id] = models_for_type
            logging.debug("Added model instance `%s` as trainable", model_instance)
        else:
            models_for_type = self._other_dict.get(model_type_id, [])
            models_for_type.append(model_instance)
            self._other_dict[model_type_id] = models_for_type
            logging.debug("Added model instance `%s` as other", model_instance)

    def get_active_model_for_type(self, model_type_id: str) -> ModelInstance:
        """
        Get the active model instance for the passed model type id, if not found raise a ValueError.
        An `active` model instance is the newest servable model instance if available for a model type.

        """
        if model_type_id in self._servable_dict:
            return self._servable_dict[model_type_id][0]
        raise ValueError(
            f"No trainable or servable model found for passed model type id:`{model_type_id}`, available model types are:`{self._servable_dict.keys()}`"
        )

    def get_active_models(self):
        """
        Get the active model instances for all model types, i.e. the newest servable model instance for each model type.
        """
        active_models = {}
        # pylint: disable=consider-using-dict-items
        for model_type_id in self._servable_dict.keys():
            active_models[model_type_id] = self.get_active_model_for_type(
                model_type_id
            ).to_json()
        return active_models

    def find_model_instance(self, model_instance_id: str) -> ModelInstance:
        for model_instances in chain(
            self._servable_dict.values(),
            self._trainable_dict.values(),
            self._other_dict.values(),
        ):
            for model_instance in model_instances:
                if model_instance.identifier == model_instance_id:
                    return model_instance
        return None

    @property
    def root_data_dir(self):
        return self._root_data_dir

    @property
    def servable(self):
        return self._servable_dict

    @property
    def other(self):
        return self._other_dict

    @property
    def trainable(self):
        return self._trainable_dict

    @property
    def trainable_model_instances(self):
        return list(chain(*self._trainable_dict.values()))

    @property
    def servable_model_instances(self):
        return list(chain(*self._servable_dict.values()))

    @property
    def other_model_instances(self):
        return list(chain(*self._other_dict.values()))

    def clear(self) -> _Models:
        self._servable_dict.clear()
        self._trainable_dict.clear()
        self._other_dict.clear()
        return self

    def reload(self) -> _Models:
        start_time = time.time()

        self.clear()
        self._populate(self._root_data_dir)

        duration = time.time() - start_time
        logging.debug("Reload method duration: %s seconds", duration)

        return self

    def to_json(self, verbose: bool = False):

        if not verbose:
            return {
                "servable": {k: str(v) for k, v in self._servable_dict.items()},
                "trainable": {k: str(v) for k, v in self._trainable_dict.items()},
                "other": {k: str(v) for k, v in self._other_dict.items()},
            }
        else:
            return {
                "servable": {
                    k: [item.to_json() for item in v]
                    for k, v in self._servable_dict.items()
                },
                "trainable": {
                    k: [item.to_json() for item in v]
                    for k, v in self._trainable_dict.items()
                },
                "other": {
                    k: [item.to_json() for item in v]
                    for k, v in self._other_dict.items()
                },
            }

    def _populate(self, dir_name: str) -> _Models:
        # check directory exists
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} not found")

        root_len = len(dir_name.split(os.path.sep))
        # walk the directory and get the model instances in reverse date order
        for subdir, _, _ in sorted(os.walk(dir_name), reverse=True):
            # if the difference between the number of parts in the root and
            # the number of parts in the directory is 4 then add it to the list
            subdirs = subdir.split(os.path.sep)
            if len(subdirs) - root_len == 4:
                try:
                    model_instance = ModelInstance(subdir)
                    self._add(model_instance)
                except Exception as e:
                    logging.error(
                        "Skipping dir `%s` due to error creating ModelInstance: %s\n%s",
                        subdir,
                        e,
                        traceback.format_exc(),  # This will log the full stack trace
                    )
        if not is_test_environment():
            logger.info("Models: %s", str(self))


class ModelInstance:

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

    def reload_state(self):
        self.__determine_state()
        return self

    def check_trainable(self):
        if not self.is_trainable():
            raise ValueError(f"Model instance `{self}` is not in a state to be trained")
        self.__logic.check_trainable()

    def check_servable(self):
        if not self.is_servable():
            raise ValueError(f"Model instance `{self}` is not in a state to be served")

    def is_trainable(self):
        return (
            self.state is ModelInstanceStateEnum.DATA_UPLOADED
            or self.state is ModelInstanceStateEnum.TRAINING_IN_PROGRESS
        )

    def is_servable(self):
        return self.state == ModelInstanceStateEnum.TRAINED_READY_TO_SERVE

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
        self.__training_progress_details_file = os.path.join(
            self.__training_subdir, Constants.TRAINING_PROGRESS_DETAILS_LOG
        )
        self.__training_in_progress_file = os.path.join(
            self.__training_subdir, Constants.TRAINING_IN_PROGRESS_LOG
        )
        if not os.path.exists(
            os.path.join(self.__directory, Constants.MODEL_DATA_FILE)
        ):
            raise FileNotFoundError(
                f"Could not determine state, `{Constants.MODEL_DATA_FILE}` not found in `{self.__directory}`"
            )

        # Determine the state of the model instance
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

            timestats = {"cvTimeSecs": cv_time, "fitTimeSecs": fit_time}
            with open(
                self.__training_subdir + "/time.stats", "w", encoding="utf8"
            ) as f:
                f.write(json.dumps(timestats))

            # save the trained model to the model pickle file
            pickle.dump(pipeline, open(self.__model_pickle_file, "wb"))

            logger.info(
                "Successfully Trained the model instance:`%s`. Model file is saved to `%s`",
                self.directory,
                self.__model_pickle_file,
            )
            # remove the training in progress file
            os.remove(self.__training_in_progress_file)
            self.reload_state()
        except Exception as e:
            err_msg = traceback.format_exc()
            # write the error to the training error log file
            with open(self.__training_error_file, "w", encoding="utf8") as f:
                f.write(err_msg)
            logging.error("Error training model instance `%s`: %s", str(self), err_msg)
            raise e

    def load_model(self) -> Pipeline:
        self.check_servable()
        return pickle.load(open(self.__model_pickle_file, "rb"))

    def predict(self, prediction_input: PredictionInput) -> PredictionOutput:
        """
        Given the dictionary of features which must be the same of the train features,
        predict the output and return the PredictionOutput annotations.

        Args:
            features (dict[str, any]): The dictionary of features used for prediction

        Returns:
            PredictionOutput: The prediction output

        """
        self.check_servable()
        prediction_input.check_valid_features(self.features_fields)

        logger.debug(
            "Predicting using algorithm `%s` and features `%s`",
            self.__logic,
            prediction_input.feature_values,
        )

        prediction_output: PredictionOutput = self.__logic.predict(prediction_input)
        logger.debug(
            "Predicted using algorithm `%s` and features `%s`; returned result: `%s`",
            self.__logic,
            prediction_input.feature_values,
            prediction_output,
        )
        return prediction_output

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
    def training_in_progress_file(self) -> str:
        return self.__training_in_progress_file

    @property
    def training_progress_details_file(self) -> str:
        return self.__training_progress_details_file

    @property
    def training_subdir(self) -> str:
        return self.__training_subdir

    @property
    def identifier(self) -> str:
        """
        The identifier of the model instance, i.e. task/type/project
        """
        return f"{self.task}/{self.type}/{self.project}/{self.instance}"

    @property
    def type_identifier(self) -> str:
        """
        The identifier of the model, i.e. task/type/project
        """
        return f"{self.task}/{self.type}/{self.project}"

    @property
    def stats_metrics(self) -> dict:
        if not self.is_servable():
            return None
        if not os.path.exists(self.__training_subdir + "/metrics.stats"):
            msg = f"Metrics file not found for model instance in `{self.__directory}`"
            logger.warning(msg)
            return msg

        return pd.read_csv(self.__training_subdir + "/metrics.stats").to_csv(
            index=False
        )

    @property
    def stats_confusion_matrix(self) -> str:
        if not self.is_servable():
            return None
        if not os.path.exists(self.__training_subdir + "/confusion_matrix.stats"):
            msg = f"Confusion Matrix file not found for model instance in `{self.__directory}`"
            logger.warning(msg)
            return msg

        confusion_matrix = numpy.loadtxt(
            self.__training_subdir + "/confusion_matrix.stats", delimiter=","
        )
        return confusion_matrix.tolist()

    @property
    def stats_time(self) -> dict:
        if not self.is_servable():
            return None
        if not os.path.exists(self.__training_subdir + "/time.stats"):
            msg = f"`time.stats` file not found for model instance in `{self.__directory}`"
            logger.warning(msg)
            return msg

        with open(self.__training_subdir + "/time.stats", "r", encoding="utf8") as f:
            return json.loads(f.read())

    # Override the __str__ method to return a string representation of the
    # object
    def __str__(self) -> str:
        return self.identifier

    def to_json(self) -> dict:

        training_details = None
        if os.path.exists(self.__training_progress_details_file):
            with open(
                self.__training_progress_details_file, "r", encoding="utf-8"
            ) as f:
                training_details = f.read()
            training_details = [
                line for line in training_details.split("\n") if line != ""
            ]
        metrics_details = None
        if self.stats_metrics:
            metrics_details = [
                line for line in self.stats_metrics.split("\n") if line != ""
            ]

        data = {
            "task": self.task,
            "type": self.type,
            "project": self.project,
            "instance_name": self.instance,
            "state": self.state.name,
            "training_log": training_details,
            "features": self.features_fields,
            "target": self.target_field,
            "stats": {
                "metrics": metrics_details,
                "confusion_matrix": self.stats_confusion_matrix,
                "time": self.stats_time,
            },
        }
        return data


models = _Models(data_path.as_posix())

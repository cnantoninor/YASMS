from datetime import datetime
import io
import json
import logging
import os
import glob
import zipfile
import shutil
from typing import List
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from app_startup import bootstrap_app
import config
from model_instance import ModelInstance, available_models
from utils import check_valid_biz_task_model_pair
from task_manager import tasks_queue
from trainer import TrainingTask

bootstrap_app()

app = FastAPI()


@app.get("/logs")
async def get_app_log():
    with open(config.LOG_FILE) as f:
        applog = f.read().split("\n")

    return {f"{config.LOG_FILE}": applog}


@app.get("/tasks/queue")
async def get_tasks_queue():
    return {"tasks_queue": tasks_queue.to_json()}


@app.get("/models/available")
async def get_available_models():
    """
    Retrieves the state of all the model instances.
    Torna solo ultima o quella attiva

    Returns:
        dict: A dictionary containing the state of all the model instances.
    """
    return {"available_models": available_models.to_json()}


@app.get("/models/registered_types")
async def get_registered_types():
    """
    Retrieves the available business tasks and model types registered in the server.

    Returns:
        json: A json containing the available business tasks and model types.

    """
    return {"available_biztasks_model_pair": config.Constants.VALID_BIZ_TASK_MODEL_PAIR}


@app.post("/models/{biz_task}/{mod_type}/{project}/upload_train_data")
async def upload_train_data(
    biz_task: str,
    mod_type: str,
    project: str,
    train_data: UploadFile = File(...),
    features_fields: List[str] = Form(...),
    target_field: str = Form(...),
):
    """
    Uploads the CSV training data file to the specified model type and model name directory.

    Args:
        train_data (UploadFile): The CSV file to be uploaded.
        biz_task (str): The business task, e.g. spam_classifier.
        mod_type (str): The type of the model, e.g. KNN, SVM, etc..
        project (str): The name of the project.
        features_fields (List[str]): the list of the fields in the
            CSV file `train_data` that will be used as features. Existence of fields will be checked.
        target_field (str): the field in the
            CSV file `train_data` that will be used as target. Existence of the field will be checked.

    Returns:
        dict: A dictionary containing the uploaded train data path.

    Raises:
        Error if the file is not a CSV or if the file does not contain the required fields.
    """

    check_valid_biz_task_model_pair(biz_task, mod_type)

    contents = await train_data.read()

    uploaded_data_dir = (
        config.data_path.joinpath(biz_task)
        .joinpath(mod_type)
        .joinpath(project)
        .joinpath(determine_model_instance_name_date_path())
    )

    os.makedirs(uploaded_data_dir, exist_ok=True)

    # Check if the file is a zip file
    if zipfile.is_zipfile(io.BytesIO(contents)):
        # If it's a zip file, extract it
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            zip_ref.extractall(path=uploaded_data_dir)
    else:
        # If it's not a zip file, write it directly
        with open(os.path.join(uploaded_data_dir, train_data.filename), "wb") as f:
            f.write(contents)

    __write_features_and_target_fields(uploaded_data_dir, features_fields, target_field)
    __check_csv_file(uploaded_data_dir, features_fields, target_field)
    __clean_train_data_dir_if_needed(uploaded_data_dir.parent)
    model_instance = ModelInstance(uploaded_data_dir.as_posix())
    tasks_queue.submit(TrainingTask(model_instance))

    logging.info(
        """Successfully uploaded train data and submitted train task for ModelInstance: `%s`""",
        model_instance,
    )

    return {"model_instance": model_instance.to_json(), "path": uploaded_data_dir}


def __write_features_and_target_fields(directory, features_fields, target_field):
    with open(
        os.path.join(directory, config.Constants.FEATURES_FIELDS_FILE),
        "w",
        encoding="utf8",
    ) as f:
        f.write("\n".join(features_fields))

    with open(
        os.path.join(directory, config.Constants.TARGET_FIELD_FILE),
        "w",
        encoding="utf8",
    ) as f:
        f.write(target_field)


def __check_csv_file(directory, features_fields, target_field):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if len(csv_files) == 0:
        raise ValueError(f"No CSV file found in the train data dir: {directory}.")

    df = pd.read_csv(csv_files[0])

    missing_columns = []
    for column in features_fields:
        if column not in df.columns:
            missing_columns.append(column)

    if target_field not in df.columns:
        missing_columns.append(target_field)

    if missing_columns:
        shutil.rmtree(directory)
        raise ValueError(
            f"The following columns are missing in the {csv_files[0]}: {missing_columns}"
        )

    if len(df) < 50:
        shutil.rmtree(directory)
        raise ValueError(f"The {csv_files[0]} must have at least 50 rows.")


def __clean_train_data_dir_if_needed(directory: str) -> None:
    # Get a list of all subdirectories
    subdirectories = [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]

    # If there are more than 10 subdirectories
    if len(subdirectories) > 10:
        # Sort the subdirectories alphabetically
        subdirectories.sort()

        # Remove the ones that are on top of the sorted list until only 10 remain
        for subdirectory in subdirectories[:-10]:
            shutil.rmtree(subdirectory)


def determine_model_instance_name_date_path() -> str:
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H-%M-%S-%f")
    return date_time_str


@app.get("/models/{biz_task}/{mod_type}/{project}/do_inference")
async def do_inference(
    biz_task: str,
    mod_type: str,
    project: str,
):
    """
    Perform the inference using the specified model type and name.

    Args:
        biz_task (str): The type of the model.
        mod_type (str): The name of the model.

    Returns:
        dict: A dictionary containing the inference results.
    """
    pass

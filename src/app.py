from datetime import datetime
import io
import logging
import os
import glob
import zipfile
import shutil
from typing import List
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
import config

logging.info(
    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STARTING APP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
)
app = FastAPI()


@app.get("/models")
async def get_models_state():
    """
    Retrieves the state of all the model instances.
    Torna solo ultima o quella attiva

    Returns:
        dict: A dictionary containing the state of all the model instances.
    """
    pass


@app.get("/models/registered_types_and_names")
async def get_available_biz_task_and_names():
    """
    Retrieves the available model types and model names registered in the server.

    Returns:
        dict: A dictionaries containing the model types and names that are currently available and registered in the server.
    """
    pass


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
        features_fields (List[str]): the list of the fields in the CSV file `train_data` that will be used as features. Existence of fields will be checked.
        target_field (str): the field in the CSV file `train_data` that will be used as target. Existence of the field will be checked.
        biz_task (str): The business task, e.g. spam_classifier.
        mod_type (str): The type of the model, e.g. KNN, SVM, etc..
        project (str): The name of the project.

    Returns:
        dict: A dictionary containing the uploaded train data path.

    Raises:
        Error if the file is not a CSV or if the file does not contain the required fields.
    """

    assert biz_task in config.Constants.VALID_MODEL_TYPES

    contents = await train_data.read()

    train_data_dir = (
        config.train_data_path.joinpath(biz_task)
        .joinpath(mod_type)
        .joinpath(project)
        .joinpath(determine_model_instance_name_date_path())
    )

    os.makedirs(train_data_dir, exist_ok=True)

    # Check if the file is a zip file
    if zipfile.is_zipfile(io.BytesIO(contents)):
        # If it's a zip file, extract it
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            zip_ref.extractall(path=train_data_dir)
    else:
        # If it's not a zip file, write it directly
        with open(os.path.join(train_data_dir, train_data.filename), "wb") as f:
            f.write(contents)

    __check_csv_file(train_data_dir, features_fields, target_field)
    __clean_train_data_dir_if_needed(train_data_dir.parent)

    logging.info(
        """Successfully Uploaded train data for business task `%s` 
        and model type `%s` and project `%s`
        in %s, with features:%s and target:%s""",
        biz_task,
        mod_type,
        project,
        train_data_dir,
        features_fields,
        target_field,
    )

    return {"uploaded_train_data_path": train_data_dir}


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


logging.info(
    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    STARTED   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
)


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

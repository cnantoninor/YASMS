from datetime import datetime
import io
import logging
import os
import glob
from fastapi import FastAPI, File, UploadFile, Form
import zipfile
import config
import shutil
import pandas as pd
from typing import List

logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STARTING APP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
app = FastAPI()
logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    STARTED   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

@app.post("/upload_train_data")
async def upload_train_data(
    train_data: UploadFile = File(...),
    mod_type: str = Form(...),
    mod_name: str = Form(...),
    features_fields: List[str] = Form(...),
    target_field: str = Form(...),
):
    """
    Uploads a file to the specified model type and model name directory.

    Args:
        train_data (UploadFile): The file to be uploaded.
        mod_type (str): The type of the model.
        mod_name (str): The name of the model.

    Returns:
        dict: A dictionary containing the uploaded train data path.
    """

    assert mod_type in config.Constants.VALID_MODEL_TYPES

    contents = await train_data.read()

    train_data_dir = (
        config.train_data_path.joinpath(mod_type)
        .joinpath(mod_name)
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
        "Successfully Uploaded train data for model type {} and model name {} in {}, with features:{} and target:{}",
        mod_type,
        mod_name,
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

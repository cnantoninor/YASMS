from datetime import datetime
import io
import os
import glob
from fastapi import FastAPI, File, UploadFile, Form
import zipfile
import config
import shutil

app = FastAPI()


@app.post("/upload_train_data")
async def upload_file(
    train_data: UploadFile = File(...),
    model_type: str = Form(...),
    model_name: str = Form(...),
):
    """
    Uploads a file to the specified model type and model name directory.

    Args:
        train_data (UploadFile): The file to be uploaded.
        model_type (str): The type of the model.
        model_name (str): The name of the model.

    Returns:
        dict: A dictionary containing the uploaded train data path.
    """

    contents = await train_data.read()

    assert model_type is not None
    assert model_type in config.Constants.valid_model_types
    assert model_name is not None

    train_data_path = (
        config.train_data_path.joinpath(model_type)
        .joinpath(model_name)
        .joinpath(__determine_date_path())
    )

    os.makedirs(train_data_path, exist_ok=True)

    # Check if the file is a zip file
    if zipfile.is_zipfile(io.BytesIO(contents)):
        # If it's a zip file, extract it
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            zip_ref.extractall(path=train_data_path)
    else:
        # If it's not a zip file, write it directly
        with open(os.path.join(train_data_path, train_data.filename), "wb") as f:
            f.write(contents)

    __check_csv_exists(train_data_path)

    __check_fields_exists_in_csv_file(train_data_path, model_type)

    __clean_train_data_dir_if_needed(train_data_path.parent)

    return {"uploaded_train_data_path": train_data_path}


def __check_fields_exists_in_csv_file(directory, model_type):
    # TODO Add your implementation here
    pass


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


def __check_csv_exists(directory: str) -> None:
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in the train data dir: {directory}.")


def __determine_date_path() -> str:
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H:%M:%S_%f")
    return date_time_str

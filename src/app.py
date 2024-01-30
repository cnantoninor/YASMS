from datetime import datetime
from io import BufferedReader
import io
import os
from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import zipfile
import config

os.makedirs(config.data_path, exist_ok=True)

app = FastAPI()


@app.post("/upload_model_data")
async def upload_file(model_data: UploadFile = File(...), model_name: str = Form(...)):
    """
    Uploads a file and returns the filename and model name.

    Args:
        model_data (UploadFile): The file to be uploaded.
        model_name (str): The name of the model.

    Returns:
        dict: A dictionary containing the filename and model name.
    """

    contents = await model_data.read()

    # Check if the file is a zip file
    if zipfile.is_zipfile(io.BytesIO(contents)):
        # If it's a zip file, extract it
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            zip_ref.extractall(path=config.data_path)
    else:
        # If it's not a zip file, write it directly
        with open(os.path.join(config.data_path, model_data.filename), "wb") as f:
            f.write(contents)

    return {"filename": model_data.filename, "model_name": model_name}


def __date_path() -> str:
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_str = now.strftime("%Y/%m/%d_%H:%M:%S_%f")

    # Concatenate the model name and the date and time string
    return date_time_str

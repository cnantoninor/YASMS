from datetime import datetime
import io
from fastapi.testclient import TestClient
import pytest
import os
from io import BytesIO
import zipfile
import config
from app import app, __date_path

client = TestClient(app)

os.makedirs(config.test_data_path, exist_ok=True)


def test_upload_data_model_zipped():
    """
    Test case for uploading a file.

    This function sends a POST request to the '/upload' endpoint with a model name and file data.
    It asserts that the response status code is 200 and the response JSON contains the expected filename and model name.

    Returns:
        None
    """
    model_name = "test_model"
    with open(config.Paths.test_data__upload_train_data_csv, "rb") as file:
        file_data = file.read()

    zip_data = io.BytesIO()
    with zipfile.ZipFile(zip_data, "w") as zip_file:
        zip_file.writestr("model_data.csv", file_data)

    # Go to the start of the BytesIO stream
    zip_data.seek(0)

    response = client.post(
        "/upload",
        data={"model_name": model_name},
        files={
            "model_data": (
                "test_file.zip",
                zip_data,
                "application/zip",
            )
        },
    )
    print(response)
    print(response.content)
    assert response.status_code == 200
    assert response.json() == {
        "filename": "test_file.zip",
        "model_name": model_name,
    }


def test___date_path():
    now = datetime.now()
    assert __date_path() is not None
    assert __date_path().startswith(now.strftime("%Y/%m/%d_%H:%M:%S"))
    print(__date_path())
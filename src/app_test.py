from datetime import datetime
import io
from fastapi.testclient import TestClient
import os
import zipfile
import config
from app import app, __determine_date_path, __check_csv_file

client = TestClient(app)

os.makedirs(config.test_data_path, exist_ok=True)


def test_upload_training_data_zipped():
    """
    Test case for uploading a file.

    This function sends a POST request to the '/upload' endpoint with a model name and file data.
    It asserts that the response status code is 200 and the response JSON contains the expected filename and model name.

    Returns:
        None
    """
    model_name = "test_model"
    model_type = config.Constants.model_spam_type

    with open(config.Paths.test_data__upload_train_data_csv, "rb") as file:
        file_data = file.read()

    zip_data = io.BytesIO()
    with zipfile.ZipFile(zip_data, "w") as zip_file:
        zip_file.writestr("model_data.csv", file_data)

    # Go to the start of the BytesIO stream
    zip_data.seek(0)

    response = client.post(
        "/upload_train_data",
        data={
            "model_name": model_name,
            "model_type": model_type,
            "features_fields": ["Testo"],
            "target_field": "Stato Workflow",
        },
        files={
            "train_data": (
                "test_file.zip",
                zip_data,
                "application/zip",
            )
        },
    )

    assert_upload_response(response)


def assert_upload_response(response):
    assert response.status_code == 200
    uploaded_train_data_path = response.json()["uploaded_train_data_path"]
    assert uploaded_train_data_path is not None
    assert os.path.exists(uploaded_train_data_path)
    now = datetime.now()
    assert now.strftime("%Y%m%d_%H-%M") in uploaded_train_data_path


def test___determine_date_path():
    dt_path = __determine_date_path()
    assert dt_path is not None
    now = datetime.now()
    assert dt_path.startswith(now.strftime("%Y%m%d_%H-%M-%S"))
    print(dt_path)


def test_upload_training_data_unzipped():
    model_name = "test_model"
    model_type = config.Constants.model_spam_type

    with open(config.Paths.test_data__upload_train_data_csv, "rb") as file:
        file_data = file.read()

    response = client.post(
        "/upload_train_data",
        data={
            "model_name": model_name,
            "model_type": model_type,
            "features_fields": ["Testo"],
            "target_field": "Stato Workflow",
        },
        files={
            "train_data": (
                "test_file.csv",
                file_data,
                "application/csv",
            )
        },
    )
    assert_upload_response(response)

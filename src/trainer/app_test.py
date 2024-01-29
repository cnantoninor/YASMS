from fastapi.testclient import TestClient
import pytest
from app import app
import os
from io import BytesIO
import zipfile

client = TestClient(app)


def test_upload_file():
    """
    Test case for uploading a file.

    This function sends a POST request to the '/upload' endpoint with a model name and file data.
    It asserts that the response status code is 200 and the response JSON contains the expected filename and model name.

    Returns:
        None
    """
    model_name = "test_model"
    file_data = b"Some file data"

    # Create a zip file
    with zipfile.ZipFile("test_file.zip", "w") as zip_file:
        zip_file.writestr("model.csv", file_data)

    response = client.post(
        "/upload",
        data={"model_name": model_name},
        files={
            "model_data": (
                "test_file.zip",
                open("test_file.zip", "rb"),
                "application/zip",
            )
        },
    )
    print(response)
    print(response.content)
    assert response.status_code == 200
    assert response.json() == {"filename": "test_file.zip", "model_name": model_name}

from datetime import datetime
import io
import unittest
from fastapi.testclient import TestClient
import os
import zipfile
import config
from app import app, determine_model_instance_name_date_path
from model_instance import ModelInstance, ModelInstanceStateEnum

client = TestClient(app)

os.makedirs(config.test_data_path, exist_ok=True)


class TestApp(unittest.TestCase):

    def test_upload_training_data_zipped(self):
        """
        Test case for uploading a file.

        This function sends a POST request to the '/upload' endpoint with a model name and file data.
        It asserts that the response status code is 200 and the response JSON contains the expected filename and model name.

        Returns:
            None
        """
        mod_type = "test_model"
        biz_task = config.Constants.BIZ_TASK_SPAM
        project = "test_project"

        with open(config.Paths.test_data__upload_train_data_csv_path, "rb") as file:
            file_data = file.read()

        zip_data = io.BytesIO()
        with zipfile.ZipFile(zip_data, "w") as zip_file:
            zip_file.writestr(config.Constants.MODEL_DATA_FILE, file_data)

        # Go to the start of the BytesIO stream
        zip_data.seek(0)

        response = client.post(
            f"/models/{biz_task}/{mod_type}/{project}/upload_train_data",
            data={
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

        self.assert_upload_response(response)

    def assert_upload_response(self, response):
        print(response.json())
        assert response.status_code == 200
        uploaded_train_data_path = response.json()["path"]
        assert uploaded_train_data_path is not None
        assert os.path.exists(uploaded_train_data_path)
        now = datetime.now()
        assert now.strftime("%Y%m%d_%H-%M") in uploaded_train_data_path
        mis = ModelInstance(uploaded_train_data_path)
        self.assertEqual(mis.state, ModelInstanceStateEnum.DATA_UPLOADED)
        self.assertEqual(mis.task, config.Constants.BIZ_TASK_SPAM)
        self.assertEqual(mis.features_fields, ["Testo"])
        self.assertEqual(mis.target_field, "Stato Workflow")
        model_instance_str = response.json()["model_instance"]
        self.assertEqual(model_instance_str, mis.__str__())

    def test_determine_model_instance_name_date_path(self):
        dt_path = determine_model_instance_name_date_path()
        assert dt_path is not None
        now = datetime.now()
        assert dt_path.startswith(now.strftime("%Y%m%d_%H-%M-%S"))
        print(dt_path)

    def test_upload_training_data_unzipped(self):
        mod_type = "test_model"
        biz_task = config.Constants.BIZ_TASK_SPAM
        project = "test_project"

        with open(config.Paths.test_data__upload_train_data_csv_path, "rb") as file:
            file_data = file.read()

        response = client.post(
            f"/models/{biz_task}/{mod_type}/{project}/upload_train_data",
            data={
                "features_fields": ["Testo"],
                "target_field": "Stato Workflow",
            },
            files={
                "train_data": (
                    "model_data.csv",
                    file_data,
                    "application/csv",
                )
            },
        )
        self.assert_upload_response(response)

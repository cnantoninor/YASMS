import os
from time import sleep
import zipfile
from datetime import datetime
import io
import unittest
from fastapi.testclient import TestClient
import config
from app import app, determine_model_instance_name_date_path
from model_instance import ModelInstance, ModelInstanceStateEnum, models
from prediction_model import Feature, PredictionInput
from utils_test import test_data__data_uploaded_path, test_data_path
from task_manager import tasks_executor, tasks_queue

client = TestClient(app)

os.makedirs(test_data_path, exist_ok=True)


class TestApp(unittest.TestCase):

    def test_upload_training_data_zipped(self):
        """
        Test case for uploading a file.

        This function sends a POST request to the '/upload' endpoint with a model name and file data.
        It asserts that the response status code is 200 and the response JSON
        contains the expected filename and model name.

        Returns:
            None
        """
        mod_type = "test_model"
        biz_task = config.Constants.BIZ_TASK_SPAM
        project = "test_project"

        with open(test_data__data_uploaded_path / "model_data.csv", "rb") as file:
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

    def assert_upload_response(
        self, response, features_fields=["Testo"], target_field="Stato Workflow"
    ):
        print(response.json())
        assert response.status_code == 200
        uploaded_data_path = response.json()["path"]
        assert uploaded_data_path is not None
        assert os.path.exists(uploaded_data_path)
        now = datetime.now()
        assert now.strftime("%Y%m%d_%H-%M") in uploaded_data_path
        mis = ModelInstance(uploaded_data_path)
        self.assertEqual(mis.state, ModelInstanceStateEnum.DATA_UPLOADED)
        self.assertEqual(mis.task, config.Constants.BIZ_TASK_SPAM)
        self.assertEqual(mis.features_fields, features_fields)
        self.assertEqual(mis.target_field, target_field)
        model_instance_str = response.json()["modelInstance"]
        self.assertEqual(model_instance_str, mis.to_json())

    def test_determine_model_instance_name_date_path(self):
        dt_path = determine_model_instance_name_date_path()
        assert dt_path is not None
        now = datetime.now()
        assert dt_path.startswith(now.strftime("%Y%m%d_%H-%M-%S"))
        print(dt_path)

    def test_upload_training_data_unzipped(self):
        response = self.upload_test_data()
        self.assert_upload_response(response)

    # pylint: disable=dangerous-default-value
    def upload_test_data(
        self,
        file_path=(test_data__data_uploaded_path / "model_data.csv").as_posix(),
        features_fields=["Testo"],
        target_field="Stato Workflow",
    ):
        mod_type = "test_model"
        biz_task = config.Constants.BIZ_TASK_SPAM
        project = "test_project"

        with open(file_path, "rb") as file:
            file_data = file.read()

        response = client.post(
            f"/models/{biz_task}/{mod_type}/{project}/upload_train_data",
            data={
                "features_fields": features_fields,
                "target_field": target_field,
            },
            files={
                "train_data": (
                    file_path,
                    file_data,
                    "application/csv",
                )
            },
        )

        return response

    def test_upload_training_data_unzipped_csv_filename_should_be_renamed(self):
        mod_type = "test_model"
        biz_task = config.Constants.BIZ_TASK_SPAM
        project = "test_project"

        with open("test_data/23457ec5-79c6-4542-a14a-14a3c96d90cb.csv", "rb") as file:
            file_data = file.read()

        response = client.post(
            f"/models/{biz_task}/{mod_type}/{project}/upload_train_data",
            data={
                "features_fields": "text",
                "target_field": "status",
            },
            files={
                "train_data": (
                    "23457ec5-79c6-4542-a14a-14a3c96d90cb.csv",
                    file_data,
                    "application/csv",
                )
            },
        )
        self.assert_upload_response(
            response, features_fields=["text"], target_field="status"
        )

    def test_upload_csv_with_tab_return_json_error(self):
        # Test that uploading a CSV file with a tab character in it returns a
        # JSON error
        mod_type = "test_model"
        biz_task = config.Constants.BIZ_TASK_SPAM
        project = "test_project"
        with open(test_data_path / "tab_separated_model_data.csv", "rb") as file:
            file_data_with_tab = file.read()
        response = client.post(
            f"/models/{biz_task}/{mod_type}/{project}/upload_train_data",
            data={
                "features_fields": ["text"],
                "target_field": "status",
            },
            files={
                "train_data": (
                    "model_data_with_tab.csv",
                    file_data_with_tab,
                    "application/csv",
                )
            },
        )
        assert response.status_code == 400
        assert response.json() is not None
        print(response.json())
        assert "tab separated" in response.json()["error"]["message"]

    def test_isalive(self):
        response = client.get("/isalive")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"alive": True})

    def test_predict(self):

        tasks_executor.reset()

        response = self.upload_test_data(
            "test_data/23457ec5-79c6-4542-a14a-14a3c96d90cb.csv",
            features_fields=["text"],
            target_field="status",
        )

        self.assertEqual(response.status_code, 200)

        biz_task = response.json()["modelInstance"]["task"]
        mod_type = response.json()["modelInstance"]["type"]
        project = response.json()["modelInstance"]["project"]
        instance_name = response.json()["modelInstance"]["instance_name"]
        model_instance_id = (
            biz_task + "/" + mod_type + "/" + project + "/" + instance_name
        )

        count = 0
        # Wait for the model instance to be created
        while models.find_model_instance(model_instance_id) is None:
            sleep(1)
            count += 1
            # after 5 seconds we should have the model instance created, raise an error otherwise
            if count > 5:
                raise Exception("Model instance not created in 5 seconds")

        mi = models.find_model_instance(model_instance_id)
        self.assertTrue(mi.is_trainable())

        mi.train()
        print(mi.to_json())
        self.assertTrue(mi.is_servable())

        prediction_input = PredictionInput(
            features=[Feature(name="text", value="Ciao ciccio!")]
        )

        response = client.post(
            f"/models/{biz_task}/{mod_type}/{project}/predict",
            json=prediction_input.model_dump(),
        )
        # Check the response status code
        self.assertEqual(response.status_code, 200)

        value = response.json()["predictions"][0]["prediction"][0]["value"]
        self.assertEqual(value, "spam")

    def test_get_active_models(self):
        response = client.get("/models/active")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("activeModels", data)
        self.assertIn("spam_classifier/test_model/test_project", data["activeModels"])
        model_data = data["activeModels"]["spam_classifier/test_model/test_project"]
        self.assertEqual(model_data["task"], "spam_classifier")
        self.assertEqual(model_data["type"], "test_model")
        self.assertEqual(model_data["project"], "test_project")
        self.assertEqual(model_data["state"], "TRAINED_READY_TO_SERVE")
        self.assertIn("training_log", model_data)
        self.assertIn("features", model_data)
        self.assertEqual(model_data["features"], ["text"])
        self.assertEqual(model_data["target"], "status")
        self.assertIn("stats", model_data)
        stats = model_data["stats"]
        self.assertIn("metrics", stats)
        self.assertIn("confusion_matrix", stats)
        self.assertIn("time", stats)

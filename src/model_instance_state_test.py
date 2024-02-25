import unittest
import os
import tempfile
from model_instance_state import ModelInstanceState, ModelInstanceStateEnum
from app import determine_model_instance_name_date_path
from config import Constants
from test_utils import (
    data_uploaded_mis_and_dir,
    trained_ready_to_serve_mis_and_dir,
    training_failed_mis_and_dir,
    training_in_progress_mis_and_dir,
    test_data__invalid_path,
)


class TestModelInstanceState(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.biz_task = Constants.BIZ_TASK_SPAM
        self.mod_type = "test_model"
        self.project = "test_project"


    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    @classmethod
    def setup_class(cls):
        os.remove("app.log")

    def test_data_uploaded_state_dir(self):
        mis, data_uploaded_dir = data_uploaded_mis_and_dir()
        self.assertEqual(mis.directory, data_uploaded_dir)
        self.assertEqual(mis.state, ModelInstanceStateEnum.DATA_UPLOADED)
        self.assertEqual(mis.task, "spam_classifier")
        self.assertEqual(mis.type, "test_model")
        self.assertEqual(mis.project, "test_project")

    def test_init_nonexistent_directory(self):
        # Test initializing ModelInstanceState with a non-existing directory
        with self.assertRaises(FileNotFoundError):
            ModelInstanceState("/non/existent/directory")

    def test_init_non_directory(self):
        # Test initializing ModelInstanceState with a non-directory
        not_a_dir = self.test_dir.name + "afile.txt"
        with open(not_a_dir, "w", encoding="utf-8") as f:
            f.write("This is not a directory")
        with self.assertRaises(NotADirectoryError):
            ModelInstanceState(not_a_dir)

    def test_init_invalid_directory_less_4_parts(self):
        # Test initializing ModelInstanceState with an invalid directory
        with self.assertRaises(ValueError):
            ModelInstanceState("/")

    def test_properties(self):
        # Test the properties of ModelInstanceState
        mod_instance_name = determine_model_instance_name_date_path()
        mod_type = "knn_123"
        biz_task = Constants.BIZ_TASK_SPAM
        project = "test_project"
        os.makedirs(
            os.path.join(
                self.test_dir.name, biz_task, mod_type, project, mod_instance_name
            )
        )
        fullpath = os.path.join(
            self.test_dir.name, biz_task, mod_type, project, mod_instance_name
        )

        mis = ModelInstanceState(fullpath)

        self.assert_mis_properties(biz_task, mod_type, project, mod_instance_name, mis)

    def assert_mis_properties(
        self,
        biz_task: str,
        mod_type: str,
        project: str,
        instance: str,
        mis: ModelInstanceState,
    ):
        self.assertEqual(
            mis.task,
            biz_task,
            f"ModelInstanceState.type should return the business task: {biz_task}",
        )

        self.assertEqual(
            mis.type,
            mod_type,
            f"ModelInstanceState.type should return the model type: {mod_type}",
        )

        self.assertEqual(
            mis.instance,
            instance,
            f"ModelInstanceState.instance_date should return \
                the model instance name: {instance}",
        )

        self.assertEqual(
            mis.project,
            project,
            f"ModelInstanceState.name should return the project name: {project}",
        )

    def test_determine_state_when_state_cannot_be_determined(self):
        # Test when the state cannot be determined
        with self.assertRaises(ValueError) as cm:
            ModelInstanceState(test_data__invalid_path.as_posix()).state
            self.assertIn("Could not determine state for", str(cm.exception))

    def test_determine_state_when_data_uploaded(self):
        # Test DATA_UPLOADED: when the dir contains only a model.csv file
        mod_instance_name = ModelInstanceStateEnum.DATA_UPLOADED.name
        mis, _ = data_uploaded_mis_and_dir()
        self.assert_mis_properties(self.biz_task, self.mod_type, self.project, mod_instance_name, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.DATA_UPLOADED,
            f"ModelInstanceState.state should be \
                DATA_UPLOADED when {Constants.MODEL_DATA_FILE} exists",
        )

    def test_determine_state_when_training_in_progress(self):
        # Test TRAINING_IN_PROGRESS: when the dir contains also a training subdir and the training_in_progress_file
        mod_instance_name = ModelInstanceStateEnum.TRAINING_IN_PROGRESS.name
        mis, _ = training_in_progress_mis_and_dir()
        self.assert_mis_properties(self.biz_task, self.mod_type, self.project, mod_instance_name, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINING_IN_PROGRESS,
            "ModelInstanceState.state should be TRAINING_IN_PROGRESS \
                when training_in_progress_file exists",
        )

    def test_determine_state_when_trained_ready_to_serve(self):
        # Test TRAINED_READY_TO_SERVE: when the dir contains the pickle model file and NOT the error file
        mod_instance_name = ModelInstanceStateEnum.TRAINED_READY_TO_SERVE.name
        mis, _ = trained_ready_to_serve_mis_and_dir()
        self.assert_mis_properties(self.biz_task, self.mod_type, self.project, mod_instance_name, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINED_READY_TO_SERVE,
            "ModelInstanceState.state should be TRAINED_READY_TO_SERVE \
                when TRAINED_MODEL_FILE file exists",
        )

    def test_determine_state_when_training_failed(self):
        # Test TRAINING_FAILED: when the dir contains training error log file AND NOT the pickle model file
        mod_instance_name = ModelInstanceStateEnum.TRAINING_FAILED.name
        mis, _ = training_failed_mis_and_dir()
        self.assert_mis_properties(self.biz_task, self.mod_type, self.project, mod_instance_name, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINING_FAILED,
            "ModelInstanceState.state should be TRAINING_FAILED \
                when TRAINING_ERROR_LOG file exists",
        )


if __name__ == "__main__":
    unittest.main()

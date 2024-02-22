import unittest
import os
import tempfile
from model_instance_state import ModelInstanceState, ModelInstanceStateNames
from app import determine_model_instance_name_date_path
from config import Constants


class TestModelInstanceState(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    @classmethod
    def setup_class(cls):
        os.remove("app.log")

    def test_DATA_UPLOADED_state_dir(self):
        # Test initializing ModelInstanceState with an existing directory
        existent_test_data_dir = (
            "test_data/spam_classifier/test_model/test_project/20240219_15-12-42-634770"
        )
        mis = ModelInstanceState(existent_test_data_dir)
        self.assertEqual(mis.directory, existent_test_data_dir)
        self.assertEqual(mis.state, ModelInstanceStateNames.DATA_UPLOADED)
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

        self.assert_mis_properties(mod_instance_name, mod_type, biz_task, project, mis)

    def assert_mis_properties(
        self,
        mod_instance_name: str,
        mod_type: str,
        biz_task: str,
        project,
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
            f"ModelInstanceState.name should return the model type: {mod_type}",
        )

        self.assertEqual(
            mis.instance,
            mod_instance_name,
            f"ModelInstanceState.instance_date should return \
                the model instance name: {mod_instance_name}",
        )

        self.assertEqual(
            mis.project,
            project,
            f"ModelInstanceState.name should return the project name: {project}",
        )

    def test_determine_state(self):
        # Test the __determine_state method of ModelInstanceState
        biz_task = Constants.BIZ_TASK_SPAM
        mod_type = "kmeans_123"
        project = "test_project"
        mod_instance_name = determine_model_instance_name_date_path()

        os.makedirs(
            os.path.join(
                self.test_dir.name, biz_task, mod_type, project, mod_instance_name
            )
        )
        fullpath = os.path.join(
            self.test_dir.name, biz_task, mod_type, project, mod_instance_name
        )

        # Test when the state cannot be determined
        with self.assertRaises(ValueError) as cm:
            _ = ModelInstanceState(fullpath).state
            self.assertIn("Could not determine state for", str(cm.exception))

        # Test when the dir contains only a model.csv file
        with open(
            os.path.join(fullpath, Constants.MODEL_DATA_FILE), "w", encoding="utf-8"
        ) as f:
            f.write("model data")
        mis = ModelInstanceState(fullpath)
        self.assert_mis_properties(mod_instance_name, mod_type, biz_task, project, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateNames.DATA_UPLOADED,
            f"ModelInstanceState.state should be \
                DATA_UPLOADED when {Constants.MODEL_DATA_FILE} exists",
        )

        # Test when the dir contains also a training subdir and the training_in_progress_file
        training_subdir = os.path.join(fullpath, Constants.TRAINING_SUBDIR)
        os.makedirs(training_subdir)
        with open(
            os.path.join(training_subdir, Constants.TRAINING_IN_PROGRESS_LOG),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("training in progress")
        mis = ModelInstanceState(fullpath)
        self.assert_mis_properties(mod_instance_name, mod_type, biz_task, project, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateNames.TRAINING_IN_PROGRESS,
            "ModelInstanceState.state should be TRAINING_IN_PROGRESS \
                when training_in_progress_file exists",
        )

        # Test when the dir contains the pickle model file and NOT the error file
        with open(
            os.path.join(training_subdir, Constants.TRAINED_MODEL_FILE),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("trained model")
        mis = ModelInstanceState(fullpath)
        self.assert_mis_properties(mod_instance_name, mod_type, biz_task, project, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateNames.TRAINED_READY_TO_SERVE,
            "ModelInstanceState.state should be TRAINED_READY_TO_SERVE \
                when TRAINED_MODEL_FILE file exists",
        )

        # Test when the dir contains training error log file AND NOT the pickle model file
        os.remove(os.path.join(training_subdir, Constants.TRAINED_MODEL_FILE))
        with open(
            os.path.join(training_subdir, Constants.TRAINING_ERROR_LOG),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("training error")
        mis = ModelInstanceState(fullpath)
        self.assert_mis_properties(mod_instance_name, mod_type, biz_task, project, mis)
        self.assertEqual(
            mis.state,
            ModelInstanceStateNames.TRAINING_FAILED,
            "ModelInstanceState.state should be TRAINING_FAILED \
                when TRAINING_ERROR_LOG file exists",
        )


if __name__ == "__main__":
    unittest.main()

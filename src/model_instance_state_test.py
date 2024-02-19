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

    def test_init_existing_directory(self):
        # Test initializing ModelInstanceState with an existing directory
        mis = ModelInstanceState(self.test_dir.name)
        self.assertEqual(mis.directory, self.test_dir.name)

    def test_init_nonexistent_directory(self):
        # Test initializing ModelInstanceState with a non-existing directory
        with self.assertRaises(FileNotFoundError):
            ModelInstanceState("/non/existent/directory")

    def test_init_non_directory(self):
        # Test initializing ModelInstanceState with a non-directory
        not_a_dir = self.test_dir.name + "afile.txt"
        with open(not_a_dir, "w") as f:
            f.write("This is not a directory")
        with self.assertRaises(NotADirectoryError):
            ModelInstanceState(not_a_dir)

    def test_init_invalid_directory_less_3_parts(self):
        # Test initializing ModelInstanceState with an invalid directory
        with self.assertRaises(ValueError):
            ModelInstanceState("/")

    def test_properties(self):
        # Test the properties of ModelInstanceState
        mod_instance_name = determine_model_instance_name_date_path()
        mod_type = "kmeans_123"
        biz_task = Constants.MODEL_SPAM_TYPE
        os.makedirs(
            os.path.join(self.test_dir.name, biz_task, mod_type, mod_instance_name)
        )
        fullpath = os.path.join(
            self.test_dir.name, biz_task, mod_type, mod_instance_name
        )

        mis = ModelInstanceState(fullpath)
        self.assertEqual(
            mis.name,
            mod_type,
            f"ModelInstanceState.name should return the model name {mod_type}",
        )

        self.assertEqual(
            mis.type,
            biz_task,
            f"ModelInstanceState.type should return the model type {biz_task}",
        )

        self.assertEqual(
            mis.instance_date,
            mod_instance_name,
            f"ModelInstanceState.instance_date should return the model instance date {mod_instance_name}",
        )

    def test_determine_state(self):
        # Test the __determine_state method of ModelInstanceState
        mod_instance_name = determine_model_instance_name_date_path()
        mod_type = "kmeans_123"
        biz_task = Constants.MODEL_SPAM_TYPE
        os.makedirs(
            os.path.join(self.test_dir.name, biz_task, mod_type, mod_instance_name)
        )
        fullpath = os.path.join(
            self.test_dir.name, biz_task, mod_type, mod_instance_name
        )

        # Test when the state cannot be determined
        with self.assertRaises(ValueError) as cm:
            _ = ModelInstanceState(fullpath).state
            self.assertIn("Could not determine state for", str(cm.exception))

        # Test when the dir contains only a model.csv file
        with open(os.path.join(fullpath, Constants.MODEL_DATA_FILE), "w") as f:
            f.write("model data")
        mis = ModelInstanceState(fullpath)
        self.assertEqual(
            mis.state,
            ModelInstanceStateNames.DATA_UPLOADED,
            f"ModelInstanceState.state should be DATA_UPLOADED when {Constants.MODEL_DATA_FILE} exists",
        )

        # Test when the dir contains also a training subdir and the training_in_progress_file
        training_subdir = os.path.join(fullpath, Constants.TRAINING_SUBDIR)
        os.makedirs(training_subdir)
        with open(
            os.path.join(training_subdir, Constants.TRAINING_IN_PROGRESS_LOG), "w"
        ) as f:
            f.write("training in progress")
        mis = ModelInstanceState(fullpath)
        self.assertEqual(
            mis.state,
            ModelInstanceStateNames.TRAINING_IN_PROGRESS,
            "ModelInstanceState.state should be TRAINING_IN_PROGRESS when training_in_progress_file exists",
        )


if __name__ == "__main__":
    unittest.main()

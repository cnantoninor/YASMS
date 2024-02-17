import unittest
import os
import tempfile
from model_instance_state import ModelInstanceState
from app import determine_model_instance_name_date_path


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
        mod_name = "kmeans_123"
        mod_type = "spam_classifier"
        os.makedirs(
            os.path.join(self.test_dir.name, mod_type, mod_name, mod_instance_name)
        )
        fullpath = os.path.join(
            self.test_dir.name, mod_type, mod_name, mod_instance_name
        )

        mis = ModelInstanceState(fullpath)
        self.assertEqual(
            mis.name,
            mod_name,
            f"ModelInstanceState.name should return the model name {mod_name}",
        )

        self.assertEqual(
            mis.type,
            mod_type,
            f"ModelInstanceState.type should return the model type {mod_type}",
        )

        self.assertEqual(
            mis.instance_date,
            mod_instance_name,
            f"ModelInstanceState.instance_date should return the model instance date {mod_instance_name}",
        )


if __name__ == "__main__":
    unittest.main()

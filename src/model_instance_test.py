import unittest
import tempfile
from src.model_instance import ModelInstance, ModelInstanceStateEnum
from config import Constants, test_data_path
from src.utils import (
    data_uploaded_mis_and_dir,
    trained_ready_to_serve_mis_and_dir,
    training_failed_mis_and_dir,
    training_in_progress_mis_and_dir,
    test_data__invalid_path,
)


class TestModelInstance(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.biz_task = Constants.BIZ_TASK_SPAM
        self.mod_type = "test_model"
        self.project = "test_project"

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

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
            ModelInstance("/non/existent/directory")

    def test_init_non_directory(self):
        # Test initializing ModelInstanceState with a non-directory
        not_a_dir = self.test_dir.name + "afile.txt"
        with open(not_a_dir, "w", encoding="utf-8") as f:
            f.write("This is not a directory")
        with self.assertRaises(NotADirectoryError):
            ModelInstance(not_a_dir)

    def test_init_invalid_directory_less_4_parts(self):
        # Test initializing ModelInstanceState with an invalid directory
        with self.assertRaises(ValueError):
            ModelInstance("/")

    def assert_mis_properties(
        self,
        biz_task: str,
        mod_type: str,
        project: str,
        instance: str,
        mis: ModelInstance,
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
            _ = ModelInstance(test_data__invalid_path.as_posix()).state
            self.assertIn("Could not determine state for", str(cm.exception))

    def test_determine_state_when_data_uploaded(self):
        # Test DATA_UPLOADED: when the dir contains only a model.csv file
        mod_instance_name = ModelInstanceStateEnum.DATA_UPLOADED.name
        mis, _ = data_uploaded_mis_and_dir()
        self.assert_mis_properties(
            self.biz_task, self.mod_type, self.project, mod_instance_name, mis
        )
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
        self.assert_mis_properties(
            self.biz_task, self.mod_type, self.project, mod_instance_name, mis
        )

        print("****************")
        print(mis.state)
        print(type(mis.state))

        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINING_IN_PROGRESS,
            f"ModelInstanceState.state should be TRAINING_IN_PROGRESS when training_in_progress_file exists but is:`{mis.state}`",
        )

    def test_determine_state_when_trained_ready_to_serve(self):
        # Test TRAINED_READY_TO_SERVE: when the dir contains the pickle model file and NOT the error file
        mod_instance_name = ModelInstanceStateEnum.TRAINED_READY_TO_SERVE.name
        mis, _ = trained_ready_to_serve_mis_and_dir()
        self.assert_mis_properties(
            self.biz_task, self.mod_type, self.project, mod_instance_name, mis
        )
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINED_READY_TO_SERVE,
            "ModelInstanceState.state should be TRAINED_READY_TO_SERVE when TRAINED_MODEL_FILE file exists",
        )

    def test_determine_state_when_training_failed(self):
        # Test TRAINING_FAILED: when the dir contains training error log file AND NOT the pickle model file
        mod_instance_name = ModelInstanceStateEnum.TRAINING_FAILED.name
        mis, _ = training_failed_mis_and_dir()
        self.assert_mis_properties(
            self.biz_task, self.mod_type, self.project, mod_instance_name, mis
        )
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINING_FAILED,
            "ModelInstanceState.state should be TRAINING_FAILED \
                when TRAINING_ERROR_LOG file exists",
        )

    def test_from_train_directory__invalid_should_raise_file_not_found(self):
        # Test from_train_directory when the directory does not exist
        with self.assertRaises(FileNotFoundError):
            ModelInstance.from_train_directory("/sdfsdfdsf")

    def test_from_train_directory(self):
        # Test from_train_directory when the directory is a training directory
        mis_list = ModelInstance.from_train_directory(test_data_path.as_posix())

        for mis in mis_list:
            print(mis)

        self.assertEqual(
            len(mis_list),
            4,
            "ModelInstanceState.from_train_directory should return 4 model instances",
        )

        for mis in mis_list:
            self.assertEqual(
                mis.state.name,
                mis.instance,
                f"ModelInstanceState.state should be f{mis.instance} but is f{mis.state}",
            )

    def test_check_trainable(self):
        # Test check_trainable
        mis, _ = data_uploaded_mis_and_dir()
        mis.check_trainable()

        mis, _ = training_in_progress_mis_and_dir()
        mis.check_trainable()

        mis, _ = trained_ready_to_serve_mis_and_dir()
        with self.assertRaises(ValueError):
            mis.check_trainable()

        mis, _ = training_failed_mis_and_dir()
        with self.assertRaises(ValueError):
            mis.check_trainable()

    def test_load_training_data(self):
        # Test load_training_data
        mis, _ = data_uploaded_mis_and_dir()
        df = mis.load_training_data()
        self.assertEqual(
            df.shape[0],
            99,
            "ModelInstanceState.load_training_data should return a dataframe with 5572 rows",
        )
        self.assertEqual(
            df.shape[1],
            47,
            "ModelInstanceState.load_training_data should return a dataframe with 2 columns",
        )

    def test__load_features_and_target(self):
        # Test __load_features_and_target
        mis, _ = data_uploaded_mis_and_dir()
        self.assertEqual(
            mis.features_fields,
            ["Testo"],
            "ModelInstanceState.features_fields should return ['Testo']",
        )
        self.assertEqual(
            mis.target_field,
            "Stato Workflow",
            "ModelInstanceState.target_field should return 'Stato Workflow'",
        )

    def test_snake_to_camel_case(self):
        self.assertEqual(
            ModelInstance.snake_to_camel_case("spam_classifier"), "SpamClassifier"
        )


if __name__ == "__main__":
    unittest.main()

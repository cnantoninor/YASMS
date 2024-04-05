import os
import unittest
import tempfile

from mock import MagicMock, patch
from model_instance import ModelInstance, ModelInstanceStateEnum, Models
from config import Constants
from utils_test import (
    test_mdlc_data_path,
    test_data__invalid_path,
    data_uploaded_mis_and_dir,
    trained_ready_to_serve_mis_and_dir,
    training_failed_mis_and_dir,
    training_in_progress_mis_and_dir,
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
        # Test initializing ModelInstance with a non-existing directory
        with self.assertRaises(FileNotFoundError):
            ModelInstance("/non/existent/directory")

    def test_init_non_directory(self):
        # Test initializing ModelInstance with a non-directory
        not_a_dir = self.test_dir.name + "afile.txt"
        with open(not_a_dir, "w", encoding="utf-8") as f:
            f.write("This is not a directory")
        with self.assertRaises(NotADirectoryError):
            ModelInstance(not_a_dir)

    def test_init_invalid_directory_less_4_parts(self):
        # Test initializing ModelInstance with an invalid directory
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
            f"ModelInstance.type should return the business task: {biz_task}",
        )

        self.assertEqual(
            mis.type,
            mod_type,
            f"ModelInstance.type should return the model type: {mod_type}",
        )

        self.assertEqual(
            mis.instance,
            instance,
            f"ModelInstance.instance_date should return \
                the model instance name: {instance}",
        )

        self.assertEqual(
            mis.project,
            project,
            f"ModelInstance.name should return the project name: {project}",
        )

    def test_determine_state_when_state_cannot_be_determined(self):
        # Test when the state cannot be determined
        with self.assertRaises(Exception) as cm:
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
            f"ModelInstance.state should be \
                DATA_UPLOADED when {Constants.MODEL_DATA_FILE} exists",
        )

    def test_determine_state_when_training_in_progress(self):
        # Test TRAINING_IN_PROGRESS: when the dir contains also a training
        # subdir and the training_in_progress_file
        mod_instance_name = ModelInstanceStateEnum.TRAINING_IN_PROGRESS.name
        mis, _ = training_in_progress_mis_and_dir()
        self.assert_mis_properties(
            self.biz_task, self.mod_type, self.project, mod_instance_name, mis
        )
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINING_IN_PROGRESS,
            f"""ModelInstance.state should be TRAINING_IN_PROGRESS \
            when training_in_progress_file exists but is:`{mis.state}`""",
        )

    def test_determine_state_when_trained_ready_to_serve(self):
        # Test TRAINED_READY_TO_SERVE: when the dir contains the pickle model
        # file and NOT the error file
        mod_instance_name = ModelInstanceStateEnum.TRAINED_READY_TO_SERVE.name
        mis, _ = trained_ready_to_serve_mis_and_dir()
        self.assert_mis_properties(
            self.biz_task, self.mod_type, self.project, mod_instance_name, mis
        )
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINED_READY_TO_SERVE,
            "ModelInstance.state should be TRAINED_READY_TO_SERVE when TRAINED_MODEL_FILE file exists",
        )
        self.assertEqual(mis.stats_time["cvTimeSecs"], 5.0)
        self.assertEqual(mis.stats_time["fitTimeSecs"], 1.0)
        self.assertIsNotNone(mis.stats_metrics)

    def test_determine_state_when_training_failed(self):
        # Test TRAINING_FAILED: when the dir contains training error log file
        # AND NOT the pickle model file
        mod_instance_name = ModelInstanceStateEnum.TRAINING_FAILED.name
        mis, _ = training_failed_mis_and_dir()
        self.assert_mis_properties(
            self.biz_task, self.mod_type, self.project, mod_instance_name, mis
        )
        self.assertEqual(
            mis.state,
            ModelInstanceStateEnum.TRAINING_FAILED,
            "ModelInstance.state should be TRAINING_FAILED \
                when TRAINING_ERROR_LOG file exists",
        )

    def test_from_train_directory__invalid_should_raise_file_not_found(self):
        # Test from_train_directory when the directory does not exist
        with self.assertRaises(FileNotFoundError):
            Models("/sdfsdfdsf")

    def test_from_train_directory(self):
        data_dir = test_mdlc_data_path.as_posix().replace("/", os.path.sep)
        models = Models(data_dir)
        trainable_models = models.trainable.values()
        servable_models = models.servable.values()
        other_models = models.other.values()
        self.assertEqual(
            len(trainable_models),
            2,
            "Models should return 2 trainable model instances",
        )

        self.assertEqual(
            len(servable_models),
            1,
            "Models should return 1 servable model instances",
        )

        self.assertEqual(
            len(other_models),
            1,
            "Models should return 1 `other` model instances",
        )

        for model in trainable_models:
            self.assertEqual(
                model.state.name,
                model.instance,
                f"ModelInstance.state should be f{model.instance} but is f{model.state}",
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

    def test_check_servable(self):
        # Test check_servable
        mis, _ = data_uploaded_mis_and_dir()
        with self.assertRaises(ValueError):
            mis.check_servable()

        mis, _ = training_in_progress_mis_and_dir()
        with self.assertRaises(ValueError):
            mis.check_servable()

        mis, _ = trained_ready_to_serve_mis_and_dir()
        mis.check_servable()

        mis, _ = training_failed_mis_and_dir()
        with self.assertRaises(ValueError):
            mis.check_servable()

    def test_load_training_data(self):
        # Test load_training_data
        mis, _ = data_uploaded_mis_and_dir()
        df = mis.load_training_data()
        self.assertEqual(
            df.shape[0],
            99,
            "ModelInstance.load_training_data should return a dataframe with 5572 rows",
        )
        self.assertEqual(
            df.shape[1],
            47,
            "ModelInstance.load_training_data should return a dataframe with 2 columns",
        )

    def test__load_features_and_target(self):
        # Test __load_features_and_target
        mis, _ = data_uploaded_mis_and_dir()
        self.assertEqual(
            mis.features_fields,
            ["Testo"],
            "ModelInstance.features_fields should return ['Testo']",
        )
        self.assertEqual(
            mis.target_field,
            "Stato Workflow",
            "ModelInstance.target_field should return 'Stato Workflow'",
        )

    def test_snake_to_camel_case(self):
        self.assertEqual(
            ModelInstance.snake_to_camel_case("spam_classifier"), "SpamClassifier"
        )

    mock_walk_data = [
        (
            "data_dir"
            + os.path.sep
            + "spam_classifier"
            + os.path.sep
            + "test_model"
            + os.path.sep
            + "test_project"
            + os.path.sep
            + "20240311_19-52-03-239251",
            [ModelInstanceStateEnum.TRAINING_IN_PROGRESS],
            [],
        ),
        (
            "data_dir"
            + os.path.sep
            + "spam_classifier"
            + os.path.sep
            + "test_model"
            + os.path.sep
            + "test_project"
            + os.path.sep
            + "20240312_19-52-03-239251",
            [ModelInstanceStateEnum.TRAINED_READY_TO_SERVE],
            [],
        ),
    ]

    @patch("os.path.exists", return_value=True)
    @patch("os.walk", return_value=mock_walk_data)
    def test_active_models(self, mock_exists, mock_walk):
        data_dir = "data_dir"
        mock_instances = []
        for data in TestModelInstance.mock_walk_data:
            model_instance = MagicMock()
            model_instance.state = data[1][0]
            model_instance.instance = data[0].split(os.path.sep)[-1]
            mock_instances.append(model_instance)
        with patch("model_instance.ModelInstance", side_effect=mock_instances):
            models = Models(data_dir)
            self.assertEqual(
                len(models),
                len(TestModelInstance.mock_walk_data),
                "ModelInstance.load_models_from should return 4 model instances",
            )


if __name__ == "__main__":
    unittest.main()

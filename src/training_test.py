# pylint: disable=protected-access

import unittest
from unittest.mock import patch
from src.model_instance import ModelInstance
from src.utils import data_uploaded_mis_and_dir, test_data__invalid_path
from trainer import TrainingTask


class TestTrainingTask(unittest.TestCase):

    def setUp(self):
        mis, _ = data_uploaded_mis_and_dir()
        self.task = TrainingTask(mis)

    @patch("trainer.logging")
    def test_execute(self, mock_logging):
        self.task.execute()
        mock_logging.info.assert_called_once_with(
            "Executing training task `%s`", self.task
        )

    def test_check_state_not_data_uploaded(self):
        with self.assertRaises(ValueError):
            self.task = TrainingTask(
                ModelInstance(test_data__invalid_path.as_posix()),
            )


if __name__ == "__main__":
    unittest.main()

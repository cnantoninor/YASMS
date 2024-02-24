import unittest
from unittest.mock import Mock, patch
from model_instance_state import ModelInstanceState, ModelInstanceStateEnum
from trainer import TrainingTask
from config import Paths


class TestTrainingTask(unittest.TestCase):

    def setUp(self):
        mis = ModelInstanceState(Paths.test_data__data_uploaded_dir.as_posix())
        self.task = TrainingTask("TestTrainingTask", mis)

    @patch("trainer.logging")
    def test_execute(self, mock_logging):
        self.task.execute()
        mock_logging.info.assert_called_once_with(
            "Executing training task %s", self.task
        )

    def test_check_state_data_uploaded(self):
        try:
            self.task._check_state()  # Should not raise any exception
        except ValueError:
            self.fail("_check_state() raised ValueError unexpectedly!")

    def test_check_state_not_data_uploaded(self):
        self.task.model_instance_state = Mock(spec=ModelInstanceState)
        self.task.model_instance_state.state = (
            ModelInstanceStateEnum.TRAINING_IN_PROGRESS
        )
        with self.assertRaises(ValueError):
            self.task._check_state()


if __name__ == "__main__":
    unittest.main()

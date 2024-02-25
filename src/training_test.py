import unittest
from unittest.mock import patch
from test_utils import data_uploaded_mis_and_dir
from trainer import TrainingTask


class TestTrainingTask(unittest.TestCase):

    def setUp(self):
        mis, _ = data_uploaded_mis_and_dir()
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

        # todo from here

        with self.assertRaises(ValueError):
            self.task._check_state()


if __name__ == "__main__":
    unittest.main()

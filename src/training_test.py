# pylint: disable=protected-access

import datetime
import unittest
from unittest.mock import patch
from model_instance import ModelInstance
from utils_test import data_uploaded_mis_and_dir, test_data__invalid_path
from trainer import TrainingTask


class TestTrainingTask(unittest.TestCase):

    def setUp(self):
        model_instance, _ = data_uploaded_mis_and_dir()
        self.task = TrainingTask(model_instance)

    @patch("trainer.logging")
    def test_execute(self, mock_logging):
        self.task.execute()
        mock_logging.debug.assert_called()

    def test_check_state_not_data_uploaded(self):
        with self.assertRaises(Exception):
            self.task = TrainingTask(
                ModelInstance(test_data__invalid_path.as_posix()),
            )

    def test_task_to_json(self):
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=10)
        self.task._time_started = start
        self.task._time_ended = end
        json = self.task.to_json()
        self.assertEqual(
            json["name"],
            "TrainingTask::spam_classifier/test_model/test_project/DATA_UPLOADED",
        )
        self.assertEqual(
            json["timeStarted"],
            start.isoformat(),
        )
        self.assertEqual(
            json["timeEnded"],
            end.isoformat(),
        )
        self.assertEqual(
            json["durationSecs"],
            10,
        )

        self.assertIsNone(json["error"])
        self.assertIsNotNone(json["modelInstance"])


if __name__ == "__main__":
    unittest.main()

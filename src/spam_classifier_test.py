import os
import shutil
import unittest
from prediction_output import PredictionOutput
from spam_classifier import SpamClassifierModelLogic
from utils_test import data_uploaded_mis_and_dir


class TestSpamClassifierModelLogic(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.data_uploaded_mis = data_uploaded_mis_and_dir()[0]

    def setUp(self):
        # delete the ../app/log file based on the current file path
        applog = os.path.join(os.path.dirname(__file__), "../app.log")
        # reset the applog file to empty
        with open(applog, "w", encoding="utf8") as f:
            f.write("")

    # when each test finish delete the train dir
    def tearDown(self):
        if os.path.exists(self.data_uploaded_mis.training_subdir):
            shutil.rmtree(self.data_uploaded_mis.training_subdir)

    def test_check_trainable(self):
        spam_classifier = SpamClassifierModelLogic(data_uploaded_mis_and_dir()[0])
        try:
            spam_classifier.check_trainable()
        except ValueError as e:
            self.fail(e)

    def test_train(self):
        try:
            self.data_uploaded_mis.train()
            print(self.data_uploaded_mis.reload_state().to_json())
        except Exception as e:
            self.fail(e)

    def test_predict(self):
        try:
            mi = self.data_uploaded_mis
            mi.train()
            mi.reload_state()
            pred_output: PredictionOutput = mi.predict({"Testo": "Hello World!"})
            print(pred_output.model_dump_json())
        except Exception as e:
            self.fail(e)


if __name__ == "__main__":
    unittest.main()

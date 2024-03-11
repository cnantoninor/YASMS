import unittest
from spam_classifier import SpamClassifierModelLogic
from utils_test import data_uploaded_mis_and_dir


class TestSpamClassifierModelLogic(unittest.TestCase):

    def test_check_trainable(self):
        spam_classifier = SpamClassifierModelLogic(data_uploaded_mis_and_dir()[0])
        try:
            spam_classifier.check_trainable()
        except ValueError as e:
            self.fail(e)


if __name__ == "__main__":
    unittest.main()

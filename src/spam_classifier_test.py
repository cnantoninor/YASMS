import unittest
from unittest.mock import Mock, patch
from spam_classifier import SpamClassifierModelLogic
from utils import data_uploaded_mis_and_dir


class TestSpamClassifierModelLogic(unittest.TestCase):

    def test_check_trainable(self):
        self.spam_classifier = SpamClassifierModelLogic(data_uploaded_mis_and_dir()[0])
        try:
            self.spam_classifier.check_trainable()
        except ValueError as e:
            self.fail(e)


if __name__ == "__main__":
    unittest.main()

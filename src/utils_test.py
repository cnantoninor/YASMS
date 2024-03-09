import unittest
import sys
from src import import_class_from_string
from spam_classifier import SpamClassifierModelLogic
from utils import UtilsData, is_test_environment


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Backup sys.modules
        self.original_sys_modules = sys.modules.copy()

    def tearDown(self):
        # Restore sys.modules
        sys.modules = self.original_sys_modules

    def test_is_test_environment_in_unittest(self):
        UtilsData.test_environment = None
        self.assertTrue(is_test_environment())

    def test_is_test_environment_in_non_test_environment(self):
        # Remove 'unittest' and 'pytest' from sys.modules to simulate a non-test environment
        UtilsData.test_environment = None
        sys.modules.pop("unittest", None)
        sys.modules.pop("pytest", None)
        self.assertFalse(is_test_environment())

    def test_import_class_from_string(self):
        self.assertEqual(
            import_class_from_string("spam_classifier.SpamClassifierModelLogic"),
            SpamClassifierModelLogic,
        )


if __name__ == "__main__":
    unittest.main()

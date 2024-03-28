import sys
import unittest

from environment import Environment, is_test_environment


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Backup sys.modules
        self.original_sys_modules = sys.modules.copy()

    def tearDown(self):
        # Restore sys.modules
        sys.modules = self.original_sys_modules

    def test_is_test_environment_in_unittest(self):
        Environment.is_test = None
        self.assertTrue(is_test_environment())

    def test_is_test_environment_in_non_test_environment(self):
        # Remove 'unittest' and 'pytest' from sys.modules to simulate a
        # non-test environment
        Environment.is_test = None
        for module in sys.modules.copy():
            if module.startswith("unittest.") or module.startswith("pytest."):
                sys.modules.pop(module)
        self.assertFalse(is_test_environment())


if __name__ == "__main__":
    unittest.main()

from dataclasses import dataclass
import sys


@dataclass
class Environment:
    is_test = None


def is_test_environment():
    if Environment.is_test is None:
        Environment.is_test = False
        for module in sys.modules.values():
            if module.__name__.startswith("unittest."):
                Environment.is_test = True
    return Environment.is_test

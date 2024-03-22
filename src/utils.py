from dataclasses import dataclass
from importlib import import_module
import sys
from config import Constants


@dataclass
class UtilsData:
    test_environment = None


def is_test_environment():
    if UtilsData.test_environment is None:
        UtilsData.test_environment = False
        for module in sys.modules.values():
            if module.__name__ in ["pytest"]:
                UtilsData.test_environment = True
    return UtilsData.test_environment


def check_valid_biz_task_model_pair(biz_task: str, model_type: str):
    task_model_pair = f"{biz_task}/{model_type}"
    valid_pairs = (
        Constants.VALID_BIZ_TASK_MODEL_PAIR
        if not is_test_environment()
        else Constants.VALID_BIZ_TASK_MODEL_PAIR_TEST
    )
    if not task_model_pair in valid_pairs:
        raise ValueError(
            f"""Invalid business task model type pair: {task_model_pair}; /
            Valid values: {Constants.VALID_BIZ_TASK_MODEL_PAIR}"""
        )


def import_class_from_string(path: str):
    module_path, _, class_name = path.rpartition(".")
    mod = import_module(module_path)
    klass = getattr(mod, class_name)
    return klass

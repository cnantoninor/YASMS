from importlib import import_module
from config import Constants
from environment import is_test_environment


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

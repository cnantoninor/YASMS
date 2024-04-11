import logging
from environment import is_test_environment
from model_instance import ModelInstance

from task_manager import Task


class TrainingTask(Task):

    def __init__(self, model_instance: ModelInstance):
        super().__init__(model_instance)
        self.model_instance.check_trainable()

    def execute(self):
        logging.debug("START Execute training task `%s`", self)
        self.model_instance.train()
        if not is_test_environment():
            self.model_instance.train()
        else:
            logging.info("Running in test environment. Skipping TrainingTask!")
        logging.debug("END Execute training task `%s`", self)

import logging
from src.model_instance import ModelInstance

from task_manager import Task


class TrainingTask(Task):

    def __init__(self, model_instance: ModelInstance):
        super().__init__(model_instance)
        self.model_instance.check_trainable()

    def execute(self):
        logging.info("Executing training task `%s`", self)

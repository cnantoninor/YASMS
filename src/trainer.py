import logging
from src.model_instance import ModelInstance

from task_manager import Task


class TrainingTask(Task):

    def __init__(self, model_instance_state: ModelInstance):
        super().__init__(model_instance_state.__str__, model_instance_state)

    def execute(self):
        logging.info("Executing training task %s", self)

    def _check_state(self):
        self.model_instance_state.check_trainable()

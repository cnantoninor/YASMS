import logging
from model_instance_state import ModelInstanceState

from task_manager import Task


class TrainingTask(Task):

    def __init__(self, model_instance_state: ModelInstanceState):
        super().__init__(model_instance_state.__str__, model_instance_state)

    def execute(self):
        logging.info("Executing training task %s", self)

    def _check_state(self):
        self.model_instance_state.check_trainable()

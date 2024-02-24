import logging
from model_instance_state import ModelInstanceStateEnum

from task_manager import Task


class TrainingTask(Task):

    def execute(self):
        logging.info("Executing training task %s", self)

    def _check_state(self):
        if self.model_instance_state.state != ModelInstanceStateEnum.DATA_UPLOADED:
            raise ValueError(
                f"Training task can only be executed when model instance state is DATA_UPLOADED, but it is {self.model_instance_state.state}"
            )

import logging
from model_instance_state import ModelInstanceState, ModelInstanceStateEnum

from task_manager import Task


class TrainingTask(Task):

    def __init__(self, model_instance_state: ModelInstanceState):
        super().__init__(model_instance_state.__str__, model_instance_state)

    def execute(self):
        logging.info("Executing training task %s", self)

    def _check_state(self):
        if (
            self.model_instance_state.state != ModelInstanceStateEnum.DATA_UPLOADED
            and self.model_instance_state.state
            != ModelInstanceStateEnum.TRAINING_IN_PROGRESS
        ):
            raise ValueError(
                f"Training task can only be executed when model instance state is DATA_UPLOADED or \
                TRAINING_IN_PROGRESS, but it is {self.model_instance_state.state}"
            )

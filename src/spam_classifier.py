import logging
from model_instance import ModelInstance, ModelInterface

logger = logging.getLogger(__name__)


class SpamClassifierModelLogic(ModelInterface):

    def __init__(self, model_instance: ModelInstance) -> None:
        self.__model_instance = model_instance

    def check_trainable(self):
        logger.debug(
            "Checking if the spam classifier model instance:`%s` is trainable",
            self.__model_instance,
        )
        # TODO: Implement the check_trainable method

    def train(self):
        logger.info(
            "Training the spam classifier model instance:`%s`",
            self.model_instance_state,
        )
        df = self.__model_instance.load_training_data()
        # TODO: Implement the training logic

    def predict(self):
        logger.info(
            "Predicting using the spam classifier model instance:`%s`",
            self.model_instance_state,
        )

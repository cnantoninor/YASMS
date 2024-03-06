import logging
from src.model_instance import ModelInstance

logger = logging.getLogger(__name__)


class SpamClassifier:
    def __init__(self, model_instance_state: ModelInstance) -> None:
        model_instance_state.checkTrainable()
        self.model_instance_state = model_instance_state

    def train(self):
        logger.info(
            "Training the spam classifier model `%s`", self.model_instance_state
        )
        df = self.model_instance_state.load_training_data()

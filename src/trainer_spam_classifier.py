import logging
from model_instance_state import ModelInstanceState

logger = logging.getLogger(__name__)


class SpamClassifier:
    def __init__(self, model_instance_state: ModelInstanceState) -> None:
        model_instance_state.checkTrainable()
        self.model_instance_state = model_instance_state

    def train(self):
        logger.info(
            "Training the spam classifier model `%s`", self.model_instance_state
        )
        df = self.model_instance_state.load_training_data()

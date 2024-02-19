import logging


class Trainer:
    def __init__(self, train_dir: str) -> None:
        self.train_dir = train_dir
        self.logger = logging.getLogger(__name__)
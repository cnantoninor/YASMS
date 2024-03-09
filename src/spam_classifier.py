import logging
from model_instance import ModelInstance, ModelInterface

logger = logging.getLogger(__name__)


class SpamClassifierModelLogic(ModelInterface):

    def __init__(self, model_instance: ModelInstance) -> None:
        self.__model_instance = model_instance
        self.__df = None

    def check_trainable(self):
        logger.info(
            "Checking if the spam classifier model instance:`%s` is trainable",
            self.__model_instance,
        )

        target_column = self.__model_instance.target_field
        if target_column == "Stato Workflow":
            # remove rows not validated by the analyst
            self.__df = self.df[self.df[target_column] != "N"]

            self.__df["Stato Workflow"] = self.df["Stato Workflow"].replace("Y", 1)
            self.__df["Stato Workflow"] = self.df["Stato Workflow"].replace("D", 0)
            logger.info(
                "Fixed `%s`: it has been transformed to binary values: `%s` and `N` data points have been removed.",
                target_column,
                self.df[target_column].value_counts(),
            )

        target_counts = self.df[target_column].value_counts()

        total_samples = len(self.df)
        only_zero_and_ones = set(target_counts.index) == {0, 1}
        if not only_zero_and_ones:
            err_msg = (
                "The target column must have only zeros and ones instead `%s` found"
                % self.df[target_column].unique()
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        zero_count_gt_25percent = target_counts[0] >= total_samples * 0.25
        one_count_gt_25percent = target_counts[1] >= total_samples * 0.25
        more_than_50_data_points = total_samples > 50
        if (
            not only_zero_and_ones
            or not zero_count_gt_25percent
            or not one_count_gt_25percent
            or not more_than_50_data_points
        ):
            reasons = "; ".join(
                [
                    f"only_zero_and_ones: {only_zero_and_ones}",
                    f"zero_count_gt_25percent: {zero_count_gt_25percent}",
                    f"one_count_gt_25percent: {one_count_gt_25percent}",
                    f"more_than_50_data_points_with_valid_zero_and_ones: {more_than_50_data_points}",
                ]
            )
            err_msg = "The data is not trainable, reasons: %s", reasons
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.info(
            "The data is trainable. Target counts: %s; zero count: %s; one count: %s; total samples: %s",
            target_counts,
            target_counts[0],
            target_counts[1],
            total_samples,
        )

    def train(self):
        logger.info(
            "Training the spam classifier model instance:`%s`",
            self.model_instance_state,
        )
        self.df = self.__model_instance.load_training_data()
        # TODO: Implement the training logic

    def predict(self):
        logger.info(
            "Predicting using the spam classifier model instance:`%s`",
            self.model_instance_state,
        )

    @property
    def model_instance(self):
        return self.__model_instance

    @property
    def df(self):
        if self.__df is None:
            self.__df = self.__model_instance.load_training_data()
        return self.__df

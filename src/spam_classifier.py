import logging
import time
import numpy
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
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
            self.__fix_stato_workflow()

        target_counts = self.df[target_column].value_counts()

        total_samples = len(self.df)
        only_zero_and_ones = set(target_counts.index) == {0, 1}
        if not only_zero_and_ones:
            err_msg = f"""The target column must have only zeros and ones instead \
                `{self.df[target_column].unique()}` found"""
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

    def __fix_stato_workflow(self) -> None:
        # remove rows not validated by the analyst
        self.__df = self.df[self.df["Stato Workflow"] != "N"]

        self.__df["Stato Workflow"] = self.df["Stato Workflow"].replace(
            {"Y": 1, "D": 0}
        )
        logger.info(
            "Fixed `%s`: it has been transformed to binary values: `%s` and `N` data points have been removed.",
            "Stato Workflow",
            self.df["Stato Workflow"].value_counts(),
        )

    def train(self) -> tuple[pd.DataFrame, numpy.ndarray, Pipeline, float, float]:
        """
        Train the spam classifier model instance.
        """
        text_input_feature_field_name = "_text_input_feature_"
        # for each feature field, check if it is a string and concatenate all the feature fields with a new line
        for feature_field in self.model_instance.features_fields:
            if self.df[feature_field].dtypes != "object":
                raise ValueError(
                    f"Feature field `{feature_field}` must be a string, but it is a `{self.df[feature_field].dtypes}`"
                )
            self.df[text_input_feature_field_name] = self.df[feature_field].str.cat(
                sep="\n"
            )

        # Split the data into input features (X) and target variable (y)
        X = self.df[text_input_feature_field_name]  # pylint: disable=invalid-name
        if self.model_instance.target_field == "Stato Workflow":
            self.__fix_stato_workflow()
        y = self.df[self.model_instance.target_field]

        # cross validate the pipeline
        start_time = time.time()
        df_metrics, cm = self.cross_validate(X, y)
        cv_time = time.time() - start_time

        # (re)fit the pipeline
        pipeline = self.new_pipeline()
        start_time = time.time()
        pipeline.fit(X, y)
        fit_time = time.time() - start_time

        return df_metrics, cm, pipeline, cv_time, fit_time

    def new_pipeline(self):
        return Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("clf", GradientBoostingClassifier()),
            ]
        )

    def cross_validate(self, X, y):  # pylint: disable=invalid-name
        pipeline = self.new_pipeline()
        # Define the metrics to evaluate
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "log_loss": "neg_log_loss",
            "balanced_accuracy": "balanced_accuracy",
        }

        # Perform k-fold cross-validation and calculate the scores
        scores = cross_validate(pipeline, X, y, cv=5, scoring=scoring)

        # Create a list of dictionaries to store the metric values
        metrics_data = []

        # Iterate over each metric
        for metric in scoring:
            # Create a dictionary for each metric
            metric_dict = {
                "Metric": metric,
                "Value": scores["test_" + metric].mean(),
                "Standard Deviation": scores["test_" + metric].std(),
            }
            # Append the metric dictionary to the list
            metrics_data.append(metric_dict)

        # Create a pandas DataFrame from the list of dictionaries
        metrics_df = pd.DataFrame(metrics_data)

        # split test and train and create a confusion matrix
        # pylint: disable=invalid-name
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Fit the pipeline to the training data and make predictions on the test data
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        return metrics_df, cm

    def predict(self):
        logger.info(
            "Predicting using the spam classifier model instance:`%s`",
            self.model_instance,
        )

    @property
    def model_instance(self):
        return self.__model_instance

    @property
    def df(self):
        if self.__df is None:
            self.__df = self.__model_instance.load_training_data()
        return self.__df

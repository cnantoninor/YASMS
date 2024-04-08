import logging
import config
from model_instance import models
from task_manager import tasks_queue, tasks_executor
from trainer import TrainingTask
from utils import is_test_environment


def bootstrap_app():
    logging.info(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STARTING APP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    )

    if not is_test_environment():
        logging.info(
            "Creating TrainingTask(s) from ModelInstance instances in train data dir: %s...",
            config.data_path,
        )
        for model in models.trainable_model_instances:
            training_task = TrainingTask(model)
            # TasksQueue is singleton
            tasks_queue.submit(training_task)

        logging.info(
            "Successfully submitted %s Task(s) from instances in train data dir: %s; \nTasks in queue: %s",
            tasks_queue,
            config.data_path,
            tasks_queue.task_list_str,
        )
    else:
        logging.info("Running in test environment. No tasks submitted.")

    logging.info(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    STARTED   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    )

import logging
import config
from model_instance import Models
from task_manager import TasksExecutor, tasks_queue
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
        models = Models(config.data_path.as_posix())
        for model in models:
            if model.is_trainable():
                training_task = TrainingTask(model)
                # TasksQueue is singleton
                tasks_queue.submit(training_task)

        # Make sure that the singleton TasksExecutor is running
        task_size = TasksExecutor().queue.size
        task_list_str = TasksExecutor().queue.task_list_str

        logging.info(
            "Successfully submitted %s Task(s) from instances in train data dir: %s; \nTasks in queue: %s",
            task_size,
            config.data_path,
            task_list_str,
        )
    else:
        logging.info("Running in test environment. No tasks submitted.")

    logging.info(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    STARTED   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    )

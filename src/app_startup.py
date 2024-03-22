import logging
import config
from model_instance import ModelInstance
from task_manager import TasksExecutor, TasksQueue
from trainer import TrainingTask
from utils import is_test_environment


def bootstrap_app():
    logging.info(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STARTING APP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    )

    if not is_test_environment():
        logging.info(
            "Creating TrainingTask(s) from ModelInstance instances in train data dir: %s...",
            config.train_data_path,
        )
        list_mis = ModelInstance.populate_available_models(
            config.train_data_path.as_posix()
        )
        for mis in list_mis:
            if mis.is_trainable():
                training_task = TrainingTask(mis)
                # TasksQueue is singleton
                TasksQueue().submit(training_task)

        # Make sure that the singleton TasksExecutor is running
        task_list = TasksExecutor().queue.task_list_str

        logging.info(
            "Successfully submitted %s Task(s) from instances in train data dir: %s; \nTasks in queue: %s",
            len(task_list),
            config.train_data_path,
            task_list,
        )
    else:
        logging.info("Running in test environment. No tasks submitted.")

    logging.info(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    STARTED   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    )

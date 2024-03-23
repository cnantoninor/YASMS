import queue
import threading
import logging
import time
from abc import ABC, abstractmethod
from model_instance import ModelInstance


class Task(ABC):
    """
    Abstract base class representing a task.

    Attributes:
        name (str): The name of the task.
        model_instance_state (ModelInstance): The state of the model instance associated with the task.
    """

    def __init__(self, model_instance: ModelInstance):
        self._name = model_instance.__str__()
        self._model_instance = model_instance

    @property
    def name(self):
        """
        Get the name of the task.

        Returns:
            str: The name of the task.
        """
        return self._name

    @property
    def model_instance(self) -> ModelInstance:
        """
        Get the state of the model instance associated with the task.

        Returns:
            ModelInstance: The state of the model instance.
        """
        return self._model_instance

    @abstractmethod
    def execute(self):
        """
        Execute the task.
        """

    def __str__(self):
        return f"{self.__class__.__name__}::{self.name}"


class _TasksQueue:

    def __init__(self):
        self.tasks = queue.Queue(maxsize=0)

    def submit(self, task: Task):
        assert task is not None
        assert isinstance(task, Task)
        assert task.name is not None
        logging.info("Adding task to queue: `%s`", task)
        self.tasks.put_nowait(task)

    def __len__(self):
        return self.size

    def clear(self):
        self.tasks.queue.clear()

    @property
    def size(self):
        return self.tasks.qsize()

    @property
    def is_empty(self):
        return self.tasks.empty()

    @property
    def task_list_str(self) -> str:
        return [str(task) for task in list(self.tasks.queue)]


tasks_queue = _TasksQueue()


class TasksExecutor:

    # make TasksExecutor a singleton
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(TasksExecutor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @property
    def queue(self):
        return tasks_queue

    def run(self):
        while True:
            try:
                # Blocking call
                task = tasks_queue.tasks.get(block=True, timeout=None)
                logging.info("Executing task: `%s`", task)
                task.execute()
                logging.info("Task executed succesfully: `%s`", task)
            except Exception as e:
                logging.error("Error executing task `%s`: %s", task, e)
                time.sleep(5)
            finally:
                # free the thread resource from eventual poisoned tasks
                tasks_queue.tasks.task_done()
                logging.info(
                    "Removed task due to the previous error. Remaining tasks in queue: %s",
                    tasks_queue.size,
                )

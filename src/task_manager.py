import queue
import threading
import logging
import time
from abc import ABC, abstractmethod
from src.model_instance import ModelInstance


class Task(ABC):
    """
    Abstract base class representing a task.

    Attributes:
        name (str): The name of the task.
        model_instance_state (ModelInstanceState): The state of the model instance associated with the task.
    """

    def __init__(self, name, model_instance_state):
        self._name = name
        self._model_instance_state = model_instance_state
        self._check_state()

    @property
    def name(self):
        """
        Get the name of the task.

        Returns:
            str: The name of the task.
        """
        return self._name

    @property
    def model_instance_state(self) -> ModelInstance:
        """
        Get the state of the model instance associated with the task.

        Returns:
            ModelInstanceState: The state of the model instance.
        """
        return self._model_instance_state

    @abstractmethod
    def execute(self):
        """
        Execute the task.
        """

    @abstractmethod
    def _check_state(self):
        """
        Check the state of the task.
        Subclasses should Raise ValueError if the state is not valid for the task.
        """

    def __str__(self):
        return f"{self.__class__.__name__}::{self.name}"


class TasksQueue:

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton class to manage the tasks queue"""
        if not isinstance(cls._instance, cls):
            cls._instance = super(TasksQueue, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.tasks = queue.Queue(maxsize=0)

    def submit(self, task: Task):
        assert task is not None
        assert isinstance(task, Task)
        assert task.name is not None
        logging.info("Adding task to queue: `%s`", task)
        self.tasks.put_nowait(task)

    @property
    def size(self):
        return self.tasks.qsize()


class TasksExecutor:

    # make TasksExecutor a singleton
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(TasksExecutor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, tasks_queue: TasksQueue = TasksQueue()):
        self.tasks_queue = tasks_queue
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while True:
            try:
                # Blocking call
                task = self.tasks_queue.tasks.get(block=True, timeout=None)
                logging.info("Executing task: `%s`", task)
                task.execute()
                logging.info("Task executed succesfully: `%s`", task)
            except Exception as e:
                logging.error("Error executing task `%s`: %s", task, e)
                time.sleep(5)
            finally:
                # free the thread resource from eventual poisoned tasks
                self.tasks_queue.tasks.task_done()
                logging.info("Remaining tasks in queue: %s", self.tasks_queue.size)

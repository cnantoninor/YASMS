import queue
import threading
import logging
import time
from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def execute(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}::{self.name}"


class TrainingTask(Task):
    def execute(self):
        logging.info("Executing training task %s", self)


class TasksQueue:
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


class Subscriber:
    def __init__(self, tasks_queue: TasksQueue):
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
                logging.info(
                    "Remaining tasks in queue: %s", self.tasks_queue.size
                )

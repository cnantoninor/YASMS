from datetime import datetime
import queue
import threading
import logging
import time
from abc import ABC, abstractmethod
from environment import is_test_environment
from model_instance import ModelInstance, models


class Task(ABC):
    """
    Abstract base class representing a task.

    Attributes:
        name (str): The name of the task.
        model_instance_state (ModelInstance): The state of the model instance associated with the task.
    """

    def __init__(self, model_instance: ModelInstance):
        self._name = str(model_instance)
        self._model_instance = model_instance
        self._error: Exception = None
        self._time_started = None
        self._time_ended = None
        self._duration = 0

    @property
    def error(self) -> Exception:
        """
        Get the error associated with the task.

        Returns:
            Any: The error associated with the task.
        """
        return self._error

    @error.setter
    def error(self, value: Exception):
        """
        Set the error associated with the task.

        Args:
            value (Any): The error to be associated with the task.
        """
        self._error = value

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

    @property
    def time_started(self) -> datetime:
        """
        Get the time the task was started.

        Returns:
            datetime: The time the task was started.
        """
        return self._time_started

    @property
    def time_ended(self) -> datetime:
        """
        Get the time the task was ended.

        Returns:
            datetime: The time the task was ended.
        """
        return self._time_ended

    @property
    def duration_secs(self) -> float:
        """
        Get the duration of the task in seconds.

        Returns:
            float: The duration of the task in seconds.
        """
        if self._time_started and self._time_ended:
            duration = self._time_ended - self._time_started
            return duration.total_seconds()
        else:
            return -1.0

    def run(self):
        """
        Run the task.
        """
        self._time_started = datetime.now()
        self.execute()
        self._time_ended = datetime.now()
        self._duration = self._time_ended - self._time_started

    @abstractmethod
    def execute(self):
        """
        Execute the task.
        """

    def __str__(self):
        return f"{self.__class__.__name__}::{self.name}"

    def to_json(self):
        return {
            "name": str(self),
            "timeStarted": (
                self.time_started.isoformat() if self.time_started is not None else None
            ),
            "timeEnded": (
                self.time_ended.isoformat() if self.time_ended is not None else None
            ),
            "durationSecs": self.duration_secs,
            "error": str(self.error) if self.error is not None else None,
            "modelInstance": self.model_instance.to_json(),
        }


class _TasksQueue:

    def __init__(self):
        self.tasks = queue.Queue(maxsize=0)
        self._successfully_executed_tasks = []
        self._unsuccessfully_executed_tasks = []
        self._current_executing_tasks = []

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
        self._successfully_executed_tasks = []
        self._unsuccessfully_executed_tasks = []
        self._current_executing_tasks = []

    @property
    def size(self):
        return self.tasks.qsize()

    @property
    def empty(self):
        return self.tasks.empty()

    @property
    def task_list_str(self) -> str:
        return [str(task) for task in list(self.tasks.queue)]

    def to_json(self):
        return {
            "currenttime": datetime.now().isoformat(),
            "enqued": {"count": self.size, "tasks": self.task_list_str},
            "executing": {
                "count": len(self._current_executing_tasks),
                "tasks": [task.to_json() for task in self._current_executing_tasks][
                    ::-1
                ],
            },
            "succeed": {
                "count": len(self._successfully_executed_tasks),
                "tasks": [task.to_json() for task in self._successfully_executed_tasks][
                    ::-1
                ],
            },
            "failed": {
                "count": len(self._unsuccessfully_executed_tasks),
                "tasks": [
                    task.to_json() for task in self._unsuccessfully_executed_tasks
                ][::-1],
            },
        }


class _TasksExecutor:

    def __init__(self):
        self._should_run = None
        self._start()

    def _start(self):
        if self.is_running is True:
            raise Exception("Executor is already running")
        self._thread = threading.Thread(target=self.run)
        self._thread.daemon = True
        self._should_run = True
        self._thread.start()

    @property
    def is_running(self):
        return self._should_run and self._thread.is_alive()

    def reset(self) -> None:
        """
        DANGER: Clean the queue and restart the executor.
        This method is intended for testing purposes only.
        """
        self._stop()
        tasks_queue.clear()
        self._start()

    def _stop(self):
        self._should_run = False
        self._thread.join()

    def run(self):
        while self._should_run:
            if tasks_queue.empty:
                logging.debug("No tasks in queue. Waiting for tasks...")
                self.wait()
                continue
            try:
                task: Task = tasks_queue.tasks.get(block=True, timeout=None)
                # pylint: disable=protected-access
                tasks_queue._current_executing_tasks.append(task)
                logging.info("Executing task: `%s`", task)
                task.run()
                task.model_instance.reload_state()
                tasks_queue._current_executing_tasks.remove(task)
                tasks_queue._successfully_executed_tasks.append(task)
                logging.info("Task executed succesfully: `%s`", task)
            except Exception as e:
                logging.error("Error executing task `%s`: %s", task, e)
                task.error = e
                # pylint: disable=protected-access
                tasks_queue._unsuccessfully_executed_tasks.append(task)
            finally:
                try:
                    # free the thread resource from eventual poisoned tasks
                    if tasks_queue.empty is False:
                        tasks_queue.tasks.task_done()
                        logging.info(
                            "Removed task. Remaining tasks in queue: %s",
                            tasks_queue.size,
                        )
                    logging.debug("Reloading models at the end of Task execution")
                    models.reload()
                    logging.debug("Models reloaded and the end of Task execution")
                except Exception as e:
                    logging.error(
                        "Error in cleaning up resources and reloading models: %s", e
                    )

    def wait(self):
        if is_test_environment():
            time.sleep(0.1)
        else:
            time.sleep(10)


tasks_queue = _TasksQueue()
tasks_executor = _TasksExecutor()

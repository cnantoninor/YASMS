import time
import unittest
from unittest.mock import patch
from task_manager import Task, tasks_queue, TasksExecutor


class DummyTask(Task):

    def execute(self):
        pass


class TestTask(unittest.TestCase):
    def test_task_name(self):
        task = DummyTask(None)
        self.assertEqual(task.name, "None")


class TestTasksQueue(unittest.TestCase):

    def setUp(self):
        tasks_queue.clear()

    def test_publish(self):
        task = DummyTask(None)
        with patch("logging.info") as mocked_log:
            tasks_queue.submit(task)
            self.assertEqual(tasks_queue.size, 1)
            mocked_log.assert_called_once_with("Adding task to queue: `%s`", task)

    def test_size(self):
        self.assertEqual(tasks_queue.size, 0)
        task = DummyTask(None)
        tasks_queue.submit(task)
        self.assertEqual(tasks_queue.size, 1)

    def test_singleton(self):
        tasks_queue_1 = tasks_queue
        tasks_queue_2 = tasks_queue
        self.assertEqual(tasks_queue_1, tasks_queue_2)


class TestSubscriber(unittest.TestCase):
    def setUp(self):
        tasks_queue.clear()

    def test_run(self):
        self.assertEqual(tasks_queue.size, 0)
        task = DummyTask(None)
        with patch("logging.info") as mocked_log:
            tasks_queue.submit(task)
            self.assertEqual(tasks_queue.size, 1)
            TasksExecutor()
            time.sleep(0.2)
            self.assertEqual(tasks_queue.size, 0)
            mocked_log.assert_any_call("Adding task to queue: `%s`", task)
            mocked_log.assert_any_call("Executing task: `%s`", task)
            mocked_log.assert_any_call("Task executed succesfully: `%s`", task)


if __name__ == "__main__":
    unittest.main()

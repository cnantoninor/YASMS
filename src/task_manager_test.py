import time
import unittest
from unittest.mock import patch
from task_manager import Task, TrainingTask, TasksQueue, Subscriber


class DummyTask(Task):
    def execute(self):
        pass


class TestTask(unittest.TestCase):
    def test_task_name(self):
        task = DummyTask("TestTask")
        self.assertEqual(task.name, "TestTask")


class TestTrainingTask(unittest.TestCase):
    def test_execute(self):
        task = TrainingTask("TestTrainingTask")
        with patch("logging.info") as mocked_log:
            task.execute()
            mocked_log.assert_called_once()


class TestTasksQueue(unittest.TestCase):
    def test_publish(self):
        tasks_queue = TasksQueue()
        task = TrainingTask("TestTrainingTask")
        with patch("logging.info") as mocked_log:
            tasks_queue.submit(task)
            self.assertEqual(tasks_queue.size, 1)
            mocked_log.assert_called_once_with("Adding task to queue: `%s`", task)

    def test_size(self):
        tasks_queue = TasksQueue()
        self.assertEqual(tasks_queue.size, 0)
        task = TrainingTask("TestTrainingTask")
        tasks_queue.submit(task)
        self.assertEqual(tasks_queue.size, 1)


class TestSubscriber(unittest.TestCase):
    def test_run(self):
        tasks_queue = TasksQueue()
        self.assertEqual(tasks_queue.size, 0)
        task = DummyTask("dummy_task")
        with patch("logging.info") as mocked_log:
            tasks_queue.submit(task)
            self.assertEqual(tasks_queue.size, 1)
            _ = Subscriber(tasks_queue)
            time.sleep(0.2)
            self.assertEqual(tasks_queue.size, 0)
            mocked_log.assert_any_call("Adding task to queue: `%s`", task)
            mocked_log.assert_any_call("Executing task: `%s`", task)
            mocked_log.assert_any_call("Task executed succesfully: `%s`", task)
            mocked_log.assert_any_call("Remaining tasks in queue: %s", 0)


if __name__ == "__main__":
    unittest.main()

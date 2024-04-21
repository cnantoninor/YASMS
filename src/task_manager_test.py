import time
import unittest
from unittest.mock import MagicMock, patch
from task_manager import Task, tasks_queue, tasks_executor


class DummyTask(Task):

    # pylint: disable=useless-super-delegation
    def __init__(self, modelInstanceMock=MagicMock()):
        super().__init__(modelInstanceMock)

    def execute(self):
        pass


class TestTask(unittest.TestCase):
    def test_task_name(self):
        task = DummyTask()
        self.assertIn("MagicMock", str(task))


class TestTasksQueue(unittest.TestCase):

    def setUp(self):
        tasks_queue.clear()

    def test_publish(self):
        task = DummyTask()
        with patch("logging.info") as mocked_log:
            tasks_queue.submit(task)
            self.assertEqual(tasks_queue.size, 1)
            mocked_log.assert_called_once_with("Adding task to queue: `%s`", task)

    def test_size(self):
        self.assertEqual(tasks_queue.size, 0)
        task = DummyTask()
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
        tasks_executor.reset()
        self.assertTrue(tasks_executor.is_running)
        self.assertEqual(tasks_queue.size, 0)
        task = DummyTask()
        with patch("logging.info") as mocked_log:
            tasks_queue.submit(task)
            self.assertEqual(tasks_queue.size, 1)
            self.assertTrue(tasks_executor.is_running)
            time.sleep(2.2)
            print(tasks_queue.to_json())
            # pylint: disable=protected-access
            self.assertEqual(
                len(tasks_queue._successfully_executed_tasks),
                1,
            )
            mocked_log.assert_any_call("Adding task to queue: `%s`", task)
            mocked_log.assert_any_call("Executing task: `%s`", task)
            mocked_log.assert_any_call("Task executed succesfully: `%s`", task)


if __name__ == "__main__":
    unittest.main()

import unittest
from model_state_machine import State, WorkflowManager


class TestWorkflowManager(unittest.TestCase):
    def setUp(self):
        self.workflow_manager = WorkflowManager()

    def test_add_state(self):
        state = State("state1", "data1")
        self.workflow_manager.add_state(state)
        self.assertEqual(len(self.workflow_manager.get_all_states()), 1)

    def test_get_state(self):
        state1 = State("state1", "data1")
        state2 = State("state2", "data2")
        self.workflow_manager.add_state(state1)
        self.workflow_manager.add_state(state2)
        self.assertEqual(self.workflow_manager.get_state("state1"), state1)
        self.assertEqual(self.workflow_manager.get_state("state2"), state2)

    def test_remove_state(self):
        state1 = State("state1", "data1")
        state2 = State("state2", "data2")
        self.workflow_manager.add_state(state1)
        self.workflow_manager.add_state(state2)
        self.workflow_manager.remove_state("state1")
        self.assertEqual(len(self.workflow_manager.get_all_states()), 1)
        self.assertIsNone(self.workflow_manager.get_state("state1"))

    def test_get_all_states(self):
        state1 = State("state1", "data1")
        state2 = State("state2", "data2")
        self.workflow_manager.add_state(state1)
        self.workflow_manager.add_state(state2)
        all_states = self.workflow_manager.get_all_states()
        self.assertEqual(len(all_states), 2)
        self.assertIn(state1, all_states)
        self.assertIn(state2, all_states)


if __name__ == "__main__":
    unittest.main()

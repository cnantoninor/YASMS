from enum import Enum


class ModelState(Enum):
    DataUploaded = 1
    TrainingModel = 2
    ModelTrainedReadyToServe = 3
    ModelTrainingFailed = 4


class Transition:
    def __init__(self, from_state: ModelState, to_state: ModelState):
        self.from_state = from_state
        self.to_state = to_state


class WorkflowManager:
    def __init__(self):
        self.states = []
        self.transitions = []

    def add_state(self, state):
        self.states.append(state)

    def add_transition(self, transition):
        self.transitions.append(transition)

    def get_state(self, name):
        for state in self.states:
            if state.name == name:
                return state

    def remove_state(self, name):
        for state in self.states:
            if state.name == name:
                self.states.remove(state)
                break

    def get_all_states(self):
        return self.states

    def get_possible_transitions(self, current_state):
        possible_transitions = []
        for transition in self.transitions:
            if transition.from_state == current_state:
                possible_transitions.append(transition.to_state)
        return possible_transitions

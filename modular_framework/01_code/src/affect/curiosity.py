from collections import deque


class Curiosity:
    def __init__(self, window = 5):
        self.transition_history = []

    def update(self, policy_spread):

        return 0
    
    def get_curiosity(self):

        return 0
    
    def reset(self):
        self.buffer.clear()
        self._sum = 0.0

    def clear_histories(self):
        pass

    def get_histories(self):
        return list(), list()
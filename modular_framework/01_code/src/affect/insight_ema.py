from collections import deque
import numpy as np


class Insight:
    def __init__(self, window=5, decay=0.7):
        self.window = window
        self.decay = decay
        self.buffer = deque(maxlen=self.window)
        self.abs_td_history = []
        self.insight_history = []
        self.value = 0.0

    def update(self, td_error):
        abs_td = abs(float(td_error))
        self.buffer.append(abs_td)

        self.value = self.decay * self.value + (1 - self.decay) * abs_td #new insight = part of the previous one (70%) + part of the new td error
        self.abs_td_history.append(abs_td)
        self.insight_history.append(self.value)
        return self.value

    def get(self):
        return self.value
    def reset(self):
        self.buffer.clear()
        self.value = 0.0

    def clear_histories(self):
        self.abs_td_history.clear()
        self.insight_history.clear()
        self.buffer.clear()

    def get_histories(self):
        return list(self.abs_td_history), list(self.insight_history)
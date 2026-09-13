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
        self.value_pos = 0.0
        self.value_neg = 0.0 

    def update(self, td_error):
        td_error = (float(td_error))
        if td_error > 0:
            td_pos = td_error
        else:
            td_pos = 0

        if td_error < 0:
            td_neg = -td_error
        else:
            td_neg = 0

        self.value_pos = self.decay * self.value_pos + (1 - self.decay) * td_pos
        self.value_neg = self.decay * self.value_neg + (1 - self.decay) * td_neg
        self.value = self.value_pos + self.value_neg

        self.abs_td_history.append(abs(td_error))
        self.insight_history.append(self.value)
        return self.value
        

    def get_pos(self):
        return self.value_pos
    def get_neg(self):
        return self.value_neg
    
    def reset(self):
        self.buffer.clear()
        self.value = 0.0
        self.value_pos = 0.0
        self.value_neg = 0.0

    def clear_histories(self):
        self.abs_td_history.clear()
        self.insight_history.clear()
        self.buffer.clear()

    def get_histories(self):
        return list(self.abs_td_history), list(self.insight_history)
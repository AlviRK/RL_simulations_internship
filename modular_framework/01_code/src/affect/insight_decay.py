from collections import deque
import numpy as np


class Insight:
    def __init__(self, window=5, decay=0.7):
        self.window = window
        self.decay = decay
        self.buffer = deque(maxlen=self.window)
        self.abs_td_history = []
        self.insight_history = []

    def update(self, td_error):
        abs_td = abs(float(td_error))
        self.buffer.append(abs_td)

        insight = self.get()
        self.abs_td_history.append(abs_td)
        self.insight_history.append(insight)
        return insight

    def get(self):
        n = len(self.buffer)
        if n == 0:
            return 0.0

        values = list(self.buffer)

        # older values get smaller weights, recent values get larger weights
        weights = np.array([self.decay ** i for i in range(n - 1, -1, -1)])
        weights = weights / weights.sum()

        return float(np.sum(np.array(values) * weights))

    def reset(self):
        self.buffer.clear()

    def clear_histories(self):
        self.abs_td_history.clear()
        self.insight_history.clear()
        self.buffer.clear()

    def get_histories(self):
        return list(self.abs_td_history), list(self.insight_history)
from collections import deque

class Insight:
    def __init__(self, window=None):
        self.window = window
        self.buffer = deque(maxlen=self.window)
        self._sum = 0.0
        self.abs_td_history = []
        self.insight_history = []

    def update(self, td_error, policy_changed=0):
        abs_td = abs(float(td_error))

        if policy_changed ==1:
            self.buffer.append(abs_td)
        else:
            self.buffer.append(abs_td * 0.2)

        insight = self.get()
        self.abs_td_history.append(abs_td)
        self.insight_history.append(insight)
        return insight

    def get(self):
        n = len(self.buffer)
        return sum(self.buffer) / n if n > 0 else 0.0

    def reset(self):
        self.buffer.clear()
        self._sum = 0.0

    def clear_histories(self):
        self.abs_td_history.clear()
        self.insight_history.clear()
        self.buffer.clear()
        self._sum = 0.0

    def get_histories(self):
        return list(self.abs_td_history), list(self.insight_history)
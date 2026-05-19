from collections import deque


class Insight:
    def __init__(self, window = None):
        self.window = window
        self.buffer = deque(maxlen=self.window)
        self._sum = 0.0
        self.abs_td_history = []
        self.insight_history = []

    def update(self, td_error):
        abs_td = abs(float(td_error))
        if len(self.buffer) == self.window:
            self._sum -= self.buffer[0]
        self.buffer.append(abs_td)
        self._sum += abs_td

        insight = self.get()
        self.abs_td_history.append(abs_td)
        self.insight_history.append(insight)
        return insight
    
    def get(self):
        n = len(self.buffer)
        insight = self._sum / n if n > 0 else 0.0
        #print('insight: ', insight)
        return insight
    
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
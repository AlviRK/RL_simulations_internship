from collections import deque, defaultdict


class Confidence:
    def __init__(self, window = 1):
        self.window = window
        self.uncertainty_history = []
        self.confidence_history = []
        self.buffer = deque(maxlen=self.window)
        self._sum = 0.0
        self.counts = defaultdict(int)

    def update(self, state, action, n_actions):
        self.counts[(state, action)] += 1 #s-a-counter (post increment)
        counts_for_state = [self.counts[(state, a)] for a in range(n_actions)] # list with all counts for that state
        # calculate uncertainty at t
        uncertainty = sum(1/(c+1) for c in counts_for_state) / n_actions

        # update buffer
        if len(self.buffer) == self.window:
            self._sum -= self.buffer[0]
        self.buffer.append(uncertainty)
        self._sum += uncertainty

        # update confidence
        confidence = self.get_confidence()
        self.uncertainty_history.append(uncertainty)
        self.confidence_history.append(confidence)

        return confidence
    
    def get_confidence(self):
        n = len(self.buffer)
        confidence = 1-(self._sum / n if n > 0 else 0.0)
        return confidence
    
    def reset(self):
        self.buffer.clear()
        self._sum = 0.0

    def reset_counts(self):
        self.counts.clear()

    def clear_histories(self):
        self.uncertainty_history.clear()
        self.confidence_history.clear()
        self.buffer.clear()
        self._sum = 0.0

    def get_histories(self):
        return list(self.uncertainty_history), list(self.confidence_history)
import numpy as np

class Uncertainty:
    def __init__(self, threshold = 0.7):
        self.threshold = threshold

    def update(self, q_values):
        if np.all(q_values == q_values[0]):
            return 0.0
            
        sorted_q = np.sort(q_values)
        gap_q = sorted_q[-1] - sorted_q[-2]
        max_q = np.max(q_values)
        return 1.0 if gap_q < self.threshold and max_q > 0.2 else 0.0
         
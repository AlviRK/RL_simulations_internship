from collections import deque
import numpy as np

class Conflict:
    #creation of the variables
    def __init__(self, td_window=5,return_window=10, td_threshold=0.4, improvement_threshold=0.01, min_td_count=2):
        self.td_window = td_window 
        self.return_window = return_window
        self.td_threshold = td_threshold
        self.improvement_threshold = improvement_threshold
        self.min_td_count = min_td_count

        self.td_buffer = deque(maxlen=td_window)
        self.return_buffer = deque(maxlen=return_window)
        self.conflict_history = []
    
    def update_td(self, td_error):
        negativeTD = max(0.0, -float(td_error)) #only get the negative TD errors (the ones that contibute to conflict)
        self.td_buffer.append(negativeTD)

    def update_return(self, episode_return):
        self.return_buffer.append(episode_return) 

    def get(self):
        if len(self.td_buffer) < self.td_window: #wait to have 10 td errors to assess conflict
            return 0.0
        
        sum_neg_td = np.sum(self.td_buffer) #sum of negative TD errors within the time window
        num_neg_td = sum(v > 0 for v in self.td_buffer) # count of negative TD errors in the time window
        high_td = (sum_neg_td >= self.td_threshold) and (num_neg_td >= self.min_td_count)

        #if len(self.return_buffer) < 2:
            #low_improvement = True
        #else:
            #values = list(self.return_buffer)
            #current_return = values[-1]
            #recent_mean = np.mean(values[:-1])

            #recent_improvement = current_return - recent_mean
            #low_improvement = recent_improvement <= self.improvement_threshold

        conflict = 1.0 if high_td else 0.0
        self.conflict_history.append(conflict)

        return conflict
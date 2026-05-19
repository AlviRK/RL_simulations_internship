import numpy as np


class TDAgent:
    def __init__(self, env, setting):
        self.num_states = env.observation_space_n
        self.num_actions = env.n_actions()
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.alpha = setting["alpha"]      # Learning rate
        self.gamma = setting["gamma"]      # Discount factor
        self.epsilon = setting["epsilon"]    # Exploration rate
        self.epsilon_decay = setting.get("epsilon_decay", 1.5e-05)  # Decay rate for exploration   # was 4.75e-05 before
        self.epsilon_min = setting.get("epsilon_min", 0.001)  # Minimum exploration rate
        self.seed = setting.get("seed", None)
        self.rng = np.random.default_rng(self.seed) 

    # Define action selection
    def action_selection(self, state):
        exploratory = (self.rng.random() < self.epsilon)
        if exploratory:
            action = int(self.rng.integers(self.num_actions))
        else:
            q_row = self.q_table[state]
            best = np.flatnonzero(q_row == q_row.max())
            action = int(self.rng.choice(best))
        return action, exploratory

    # Define update rule
    def update_rule(self, state, action, reward, next_state, done):
        if done:
            # If terminal, there is no future Q-value
            td_target = reward 
        else:
            # Standard Bellman equation
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return td_error, td_target
import numpy as np

class SarsaAgent:

    requires_next_action = True

    def __init__(self, env, setting):
        self.num_states = env.observation_space_n
        self.num_actions = env.n_actions()
        self.alpha = setting["alpha"]
        self.gamma = setting["gamma"]
        self.epsilon = setting["epsilon"]
        self.epsilon_decay = setting.get("epsilon_decay", 4.75e-05)
        self.epsilon_min = setting.get("epsilon_min", 0.01)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.seed = setting.get("seed", None)
        self.rng = np.random.default_rng(self.seed)

    def action_selection(self, state):
        exploratory = (self.rng.random() < self.epsilon)
        if exploratory:
            action = int(self.rng.integers(self.num_actions))
        else:
            q_row = self.q_table[state]
            best = np.flatnonzero(q_row == q_row.max())
            action = int(self.rng.choice(best))
        return action, exploratory

    def update_rule(self, state, action, reward, next_state, done):
        """SARSA update: Q[s,a] ← Q[s,a] + α[r + γ Q[s',a'] − Q[s,a]]"""
        if not done:
            next_action, _ = self.action_selection(next_state)
        else:
            next_action = None
        td_target = reward if next_action is None else reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return td_error, td_target
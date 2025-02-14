import numpy as np

class Dynamic_Programming:
    def __init__(self, env):
        self.env = env

    def policy_eval(self, policy, gamma=0.99, threshold=1e-3, max_iter=100, history=False):
        if history:
            v_history = []
            v_history.append(np.zeros((self.env.state_space_dim, 1)))
        v = np.zeros((self.env.state_space_dim, 1))  # S x 1
        iter_num = 0
        v_old = v.copy()
        while True:
            for state in range(self.env.state_space_dim):
                v_next = 0
                for action in range(self.env.action_space_dim):
                    pi_sa = policy[state, action]
                    for next_state in range(self.env.state_space_dim):
                        p_sas = self.env.transition_prob[state, action, next_state]
                        v_next += pi_sa * p_sas * (self.env.reward[state, next_state] + gamma * v[next_state])
                v[state] = v_next
            if history:
                v_history.append(v.copy())
            if np.max(np.abs(v - v_old)) < threshold:
                break
            v_old = v.copy()
            iter_num += 1
            if iter_num > max_iter:
                break
        if history:
            return v, v_history
        return v

    def policy_improvement(self, policy, v, gamma=0.99, threshold=1e-3, max_iter=10):
        new_policy = np.zeros_like(policy)  # S x A x 1
        iter_num = 0
        while True:
            for state in range(self.env.state_space_dim):
                pi_s = policy[state].copy()
                bellman_v = np.zeros(self.env.action_space_dim)  # A
                for action in range(self.env.action_space_dim):
                    p_sas = self.env.transition_prob[state, action]
                    for next_state in range(self.env.state_space_dim):
                        bellman_v[action] += p_sas[next_state] * (self.env.reward[state, next_state] + gamma * v[next_state])
                argmax_action = np.argmax(bellman_v)
                new_policy[state, argmax_action] = 1
                error = np.max(np.abs(new_policy[state] - pi_s))
                policy[state] = new_policy[state]
            if error < threshold:
                break
            iter_num += 1
            if iter_num > max_iter:
                break
        return policy

    def policy_iteration(self, init_policy, gamma=0.99, threshold=1e-3, max_iter=10):
        policy_old = init_policy
        while True:
            v = self.policy_eval(policy_old, gamma=gamma, threshold=threshold, max_iter=max_iter)
            policy = self.policy_improvement(policy_old, v, gamma=gamma, threshold=threshold, max_iter=max_iter)
            if np.max(np.abs(policy - policy_old)) < threshold:
                break
            policy_old = policy
        return policy
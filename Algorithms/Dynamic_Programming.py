import numpy as np
from utils import Policy
class Dynamic_Programming:
    def __init__(self, env):
        self.env = env
        self.policy_class = Policy(env)

    def policy_eval(self, policy, gamma=0.99, threshold=1e-3, max_iter=1000, history=False):

        # Return the value function of the policy via Dynamic Programming
        # If history is True, return the history of the value function

        """
        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape (state_space_dim, action_space_dim)
        gamma : float
            Discount factor
        threshold : float
            Threshold for the value function
        max_iter : int
            Maximum number of iterations
        history : bool
            Whether to return the history of the value function
        """

        if history:
            v_history = []
            v_history.append(np.zeros((self.env.state_space_dim, 1)))
        v = np.zeros((self.env.state_space_dim)) # S
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

        # Return the improved policy given the value function via Dynamic Programming

        """
        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape (state_space_dim, action_space_dim)
        v : numpy.ndarray
            Value function of shape (state_space_dim, 1)
        gamma : float
            Discount factor
        threshold : float
            Threshold for the policy
        max_iter : int
            Maximum number of iterations
        """

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

    def policy_iteration(self, init_policy = None, gamma=0.99, threshold=1e-3, max_iter=10):
        """
        Parameters
        ----------
        init_policy : numpy.ndarray
            Initial policy matrix of shape (state_space_dim, action_space_dim)
        gamma : float
            Discount factor
        threshold : float
            Threshold for the policy
        max_iter : int
            Maximum number of iterations
        """
        if init_policy is None:
            policy_old = self.policy_class.uniform_random_policy()
        else:
            policy_old = init_policy
        while True:
            v = self.policy_eval(policy_old, gamma=gamma, threshold=threshold, max_iter=max_iter)
            policy = self.policy_improvement(policy_old, v, gamma=gamma, threshold=threshold, max_iter=max_iter)
            if np.max(np.abs(policy - policy_old)) < threshold:
                break
            policy_old = policy
        return v, policy
    
    def value_iteration(self, gamma=0.99, threshold=1e-3, max_iter=10):
        """
        Parameters
        ----------
        gamma : float
            Discount factor
        threshold : float
            Threshold for the value function
        max_iter : int
            Maximum number of iterations
        """
        v = np.zeros((self.env.state_space_dim))
        policy = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        iter_num = 0
        v_old = v.copy()
        while True:
            for state in range(self.env.state_space_dim):
                v_next_list = []
                for action in range(self.env.action_space_dim):
                    v_next = 0
                    p_sas = self.env.transition_prob[state, action]
                    for next_state in range(self.env.state_space_dim):
                        v_next += p_sas[next_state] * (self.env.reward[state, next_state] + gamma * v[next_state])
                    v_next_list.append(v_next)
                argmax_action = np.argmax(v_next_list)
                policy[state, argmax_action] = 1
                v[state] = np.max(v_next_list)
            if np.max(np.abs(v - v_old)) < threshold:
                break
            v_old = v.copy()
            iter_num += 1
            if iter_num > max_iter:
                break
        return v, policy

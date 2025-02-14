from Algorithms.Dynamic_Programming import Dynamic_Programming
from Algorithms.Monte_Calro import Monte_Calro
from Algorithms.TD import TD
from utils import Policy

class Tabular:
    def __init__(self, env):
        self.env = env
        self.dynamic_programming = Dynamic_Programming(env)
        self.monte_calro = Monte_Calro(env)
        self.td = TD(env)
        self.policy_class = Policy(env)

    def prediction(self, policy, method, **kwargs):
        """
        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape (state_space_dim, action_space_dim)
        method : str
            Method to use for prediction. One of ["DP", "MC", "TD"]
        **kwargs : dict
            Additional arguments for specific methods
            - DP: gamma (float), threshold (float), max_iter (int), history (bool)
            - on-policy MC: max_episode_num (int), max_episode_length (int), gamma (float)
            - off-policy MC: max_episode_num (int), max_episode_length (int), gamma (float)
            - TD: gamma (float), alpha (float), max_episode_num (int), max_episode_length (int)
        """
        if method == "DP":
            return self.dynamic_programming.policy_eval(policy, **kwargs)
        elif method == "on-policy MC":
            return self.monte_calro.first_visit_mc_prediction(policy, **kwargs)
        elif method == "off-policy MC":
            return self.monte_calro.off_policy_mc_prediction(policy, **kwargs)
        elif method == "TD":
            return self.td.td_prediction(policy, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")
        
    def control(self, method, **kwargs):
        """
        Parameters
        ----------
        method : str
            Method to use for control. One of ["DP", "on-policy MC", "off-policy MC", "SARSA", "Q-LEARNING", "DOUBLE-Q-LEARNING"]
        **kwargs : dict
            Additional arguments for specific methods
            - DP: gamma (float), threshold (float), max_iter (int), history (bool)
            - on-policy MC: max_episode_num (int), max_episode_length (int), gamma (float)
            - off-policy MC: max_episode_num (int), max_episode_length (int), gamma (float), epsilon (float), epsilon_greedy (bool)
            - SARSA: gamma (float), alpha (float), max_episode_num (int), max_episode_length (int)
            - Q-LEARNING: gamma (float), alpha (float), max_episode_num (int), max_episode_length (int)
            - DOUBLE-Q-LEARNING: gamma (float), alpha (float), max_episode_num (int), max_episode_length (int)
        """
        if method == "policy iteration":
            return self.dynamic_programming.policy_iteration(**kwargs)
        elif method == "value iteration":
            return self.dynamic_programming.value_iteration(**kwargs)
        elif method == "on-policy MC":
            return self.monte_calro.on_policy_first_visit_mc_control(**kwargs)
        elif method == "off-policy MC":
            return self.monte_calro.off_policy_mc_control(**kwargs)
        elif method == "SARSA" or method == 'sarsa':
            return self.td.sarsa(**kwargs)
        elif method == "Q-LEARNING" or method == 'q-learning':
            return self.td.q_learning(**kwargs)
        elif method == "DOUBLE-Q-LEARNING" or method == 'double-q-learning':
            return self.td.double_q_learning(**kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")

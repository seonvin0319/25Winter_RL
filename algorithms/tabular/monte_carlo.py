import numpy as np
from tqdm import tqdm
from utils.tabular_utils import TabularPolicy, rollout

class Monte_Carlo:
    def __init__(self, env):
        self.env = env
        self.policy_class = TabularPolicy(env)

    def first_visit_mc_prediction(self, policy, max_episode_num = 100, max_episode_length = 1000, gamma = 0.99, show_progress = False):
        """
        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape (state_space_dim, action_space_dim)
        max_episode_num : int
            Maximum number of episodes
        max_episode_length : int
            Maximum length of an episode
        gamma : float
            Discount factor
        show_progress : bool
            Whether to show progress
        """
        v = np.zeros(self.env.state_space_dim)
        returns = {}
        for state in range(self.env.state_space_dim):
            returns[state] = []
        for _ in tqdm(range(max_episode_num), desc = "Episode", disable = not show_progress):
            buffer = rollout(self.env, policy, max_episode_length = max_episode_length)
            G = 0
            while buffer:
                state, action, reward = buffer.pop()
                G = reward + gamma*G
                if not any(state == s and action == a for s, a, _ in buffer):
                    returns[state].append(G)
                    v[state] = np.mean(returns[state])
        return v

    def off_policy_mc_prediction(self, policy, max_episode_num=100, max_episode_length=1000, gamma=0.99, show_progress = False):
        """
        Parameters
        ----------
        max_episode_num : int
            Maximum number of episodes
        max_episode_length : int
            Maximum length of an episode
        gamma : float
            Discount factor
        epsilon : float
            Epsilon for epsilon-greedy policy
        epsilon_greedy : bool
            Whether to use epsilon-greedy policy
        show_progress : bool
            Whether to show progress
        """
        self.policy_class.iter_num = 0
        uniform_policy = self.policy_class.uniform_random_policy() # Behavior Policy
        q = np.random.randn(self.env.state_space_dim, self.env.action_space_dim)
        c = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        for _ in tqdm(range(max_episode_num), desc="Episode", disable=not show_progress):
            buffer = rollout(self.env, uniform_policy, max_episode_length=max_episode_length)
            G = 0
            W = 1
            while buffer:
                state, action, reward = buffer.pop()
                G = reward + gamma*G
                c[state, action] += W
                q[state, action] += (W/c[state, action])*(G - q[state, action])
                W = W * (policy[state, action]/uniform_policy[state, action])
                if W == 0:
                    break
        return q


    def on_policy_first_visit_mc_control(self, max_episode_num=100, max_episode_length=1000, gamma=0.99, epsilon=0.1, exploring_start = True, show_progress = False):
        """
        Parameters
        ----------
        max_episode_num : int
            Maximum number of episodes
        max_episode_length : int
            Maximum length of an episode
        gamma : float
            Discount factor
        epsilon : float
            Epsilon for epsilon-greedy policy
        exploring_start : bool
            Whether to use exploring start
        show_progress : bool
            Whether to show progress
        """
        self.policy_class.iter_num = 0
        policy = self.policy_class.uniform_random_policy()
        q = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        returns = {}
        for state in range(self.env.state_space_dim):
            for action in range(self.env.action_space_dim):
                returns[(state, action)] = []
        if exploring_start:
            init_state = np.random.choice(self.env.state_space)
            init_action = np.random.choice(self.env.action_space)
        else:
            init_state = None
            init_action = None
        for _ in tqdm(range(max_episode_num), desc="Episode", disable=not show_progress):
            buffer = rollout(self.env, policy, init_state, init_action, max_episode_length)
            G = 0
            while buffer:
                state, action, reward = buffer.pop()
                G = reward + gamma*G
                if not any(state == s and action == a for s, a, _ in buffer):
                    returns[(state, action)].append(G)
                    q[state, action] = np.mean(returns[(state, action)])
                    policy = self.policy_class.epsilon_greedy_policy(q, epsilon)
        return q, policy

    def off_policy_mc_control(self, behavior_policy = None, max_episode_num=100, max_episode_length=1000, gamma=0.99, show_progress = False, Polyak_Ruppert = False):
        """
        Parameters
        ----------
        behavior_policy : numpy.ndarray
            Behavior policy matrix of shape (state_space_dim, action_space_dim)
        max_episode_num : int
            Maximum number of episodes
        max_episode_length : int
            Maximum length of an episode
        gamma : float
            Discount factor
        epsilon : float
            Epsilon for epsilon-greedy policy
        epsilon_greedy : bool
            Whether to use epsilon-greedy policy
        show_progress : bool
            Whether to show progress
        Polyak_Ruppert : bool
            Whether to use Polyak-Ruppert
        """
        self.policy_class.iter_num = 0
        if behavior_policy is None:
            behavior_policy = self.policy_class.uniform_random_policy()
        policy = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        q = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        c = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        if Polyak_Ruppert:
            q_avg = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
            n_updates = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        for _ in tqdm(range(max_episode_num), desc="Episode", disable=not show_progress):
            buffer = rollout(self.env, behavior_policy, max_episode_length=max_episode_length)
            G = 0
            W = 1
            while buffer:
                state, action, reward = buffer.pop()
                G = reward + gamma*G
                c[state, action] += W
                q[state, action] += (W/c[state, action])*(G - q[state, action])
                max_q = np.max(q[state])
                best_actions = np.where(q[state] == max_q)[0]
                policy[state, best_actions] = 1 / len(best_actions)
                if Polyak_Ruppert:
                    n_updates[state, action] += 1
                    step_size = 1 / n_updates[state, action]
                    q_avg[state, action] += step_size * (q[state, action] - q_avg[state, action])
                    max_q = np.max(q_avg[state])
                    best_actions = np.where(q_avg[state] == max_q)[0]
                    policy[state, best_actions] = 1 / len(best_actions)

                if len(best_actions) == 1:
                    if action == best_actions[0]:
                        W = W * (1 / behavior_policy[state, action])
                else:
                    break
        if Polyak_Ruppert:
            return q_avg, policy
        else:
            return q, policy
import numpy as np
from tqdm import tqdm
from utils import Policy, rollout

class Monte_Carlo:
    def __init__(self, env):
        self.env = env
        self.policy_class = Policy(env)

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

    def off_policy_mc_control(self, max_episode_num=100, max_episode_length=1000, gamma=0.99, epsilon=0.1, epsilon_greedy = True, show_progress = False):
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
        uniform_policy = self.policy_class.uniform_random_policy()
        policy = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
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
                if epsilon_greedy:
                    policy[state] = self.policy_class.epsilon_greedy_policy(q, epsilon)[state]
                else:
                    policy[state] = self.policy_class.greedy_policy(q)[state]
                best_action = np.argmax(policy[state])
                if action == best_action:
                    W = W * (1 / uniform_policy[state, action])
                else:
                    break

        return q, policy
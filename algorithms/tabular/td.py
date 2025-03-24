import numpy as np
from utils.tabular_utils import TabularPolicy

class TD:
    def __init__(self, env):
        self.env = env
        self.policy_class = TabularPolicy(env)

    def td_prediction(self, policy, gamma=0.99, alpha=0.1, max_episode_num = 100, max_episode_length = 1000):
        """
        Parameters
        ----------
        policy : np.ndarray
            Policy to use for TD prediction
        gamma : float
            Discount factor
        """
        v = np.zeros(self.env.state_space_dim)
        for _ in range(max_episode_num):
            S = self.env.reset()
            for _ in range(max_episode_length):
                A = np.random.choice(self.env.action_space, p=policy[S])
                S_next, R, done = self.env.step(A)
                v[S] = v[S] + alpha * (R + gamma * v[S_next] - v[S])
                S = S_next
                if done:
                    break
        return v

    def sarsa(self, gamma=0.99, alpha=0.1, epsilon=0.1, max_episode_num = 100, max_episode_length = 1000, epsilon_greedy = True):
        """
        Parameters
        ----------
        gamma : float
            Discount factor
        alpha : float
            Learning rate
        epsilon : float
            Epsilon for epsilon-greedy policy
        max_episode_num : int
            Maximum number of episodes
        max_episode_length : int
            Maximum length of episode
        epsilon_greedy : bool
            Whether to use epsilon-greedy policy
        """
        q = np.random.randn(self.env.state_space_dim, self.env.action_space_dim)
        self.policy_class.iter_num = 0
        policy = self.policy_class.epsilon_greedy_policy(q, epsilon)
        for _ in range(max_episode_num):
            S = self.env.reset()
            A = np.random.choice(self.env.action_space, p=policy[S])
            for _ in range(max_episode_length):
                S_next, R, done = self.env.step(A)
                A_next = np.random.choice(self.env.action_space, p=policy[S_next])
                q[S, A] = q[S, A] + alpha * (R + gamma * q[S_next, A_next] - q[S, A])
                if epsilon_greedy:
                    policy = self.policy_class.epsilon_greedy_policy(q, epsilon)
                else:
                    policy = self.policy_class.greedy_policy(q)
                S = S_next
                A = A_next
                if done:
                    break
        return q, policy

    def q_learning(self, gamma=0.99, alpha=0.1, epsilon=0.1, max_episode_num = 100, max_episode_length = 1000, epsilon_greedy = True):
        """
        Parameters
        ----------
        gamma : float
            Discount factor
        alpha : float
            Learning rate
        epsilon : float
            Epsilon for epsilon-greedy policy
        max_episode_num : int
            Maximum number of episodes
        max_episode_length : int
            Maximum length of episode
        epsilon_greedy : bool
            Whether to use epsilon-greedy policy
        """
        q = np.random.randn(self.env.state_space_dim, self.env.action_space_dim)
        self.policy_class.iter_num = 0
        policy = self.policy_class.epsilon_greedy_policy(q, epsilon)
        for _ in range(max_episode_num):
            S = self.env.reset()
            for _ in range(max_episode_length):
                A = np.random.choice(self.env.action_space, p=policy[S])
                S_next, R, done = self.env.step(A)
                q[S, A] = q[S, A] + alpha * (R + gamma * np.max(q[S_next]) - q[S, A])
                if epsilon_greedy:
                    policy = self.policy_class.epsilon_greedy_policy(q, epsilon)
                else:
                    policy = self.policy_class.greedy_policy(q)
                S = S_next
                if done:
                    break
        return q, policy

    def double_q_learning(self, gamma=0.99, alpha=0.1, epsilon=0.1, max_episode_num = 100, max_episode_length = 1000, epsilon_greedy = True):
        """
        Parameters
        ----------
        gamma : float
            Discount factor
        alpha : float
            Learning rate
        epsilon : float
            Epsilon for epsilon-greedy policy
        max_episode_num : int
            Maximum number of episodes
        max_episode_length : int
            Maximum length of episode
        epsilon_greedy : bool
            Whether to use epsilon-greedy policy
        """
        q1 = np.random.randn(self.env.state_space_dim, self.env.action_space_dim)
        q2 = np.random.randn(self.env.state_space_dim, self.env.action_space_dim)
        self.policy_class.iter_num = 0
        policy = self.policy_class.epsilon_greedy_policy(q1+q2, epsilon)
        for _ in range(max_episode_num):
            S = self.env.reset()
            for _ in range(max_episode_length):
                A = np.random.choice(self.env.action_space, p=policy[S])
                S_next, R, done = self.env.step(A)
                random_sample = np.random.rand()
                if random_sample < 0.5:
                    q1[S, A] = q1[S, A] + alpha * (R + gamma * q2[S_next, np.argmax(q1[S_next])] - q1[S, A])
                else:
                    q2[S, A] = q2[S, A] + alpha * (R + gamma * q1[S_next, np.argmax(q2[S_next])] - q2[S, A])
                if epsilon_greedy:
                    policy = self.policy_class.epsilon_greedy_policy(q1+q2, epsilon)
                else:
                    policy = self.policy_class.greedy_policy(q1+q2)
                S = S_next
                if done:
                    break
        return (q1 + q2) / 2, policy


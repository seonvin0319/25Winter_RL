import numpy as np
from tqdm import tqdm
from utils import Policy, rollout

class Monte_Calro:
    def __init__(self, env):
        self.env = env
        self.policy_class = Policy(env)

    def first_visit_mc_prediction(self, policy, max_episode_num = 100, max_episode_length = 1000, gamma = 0.99):
        v = np.zeros(self.env.state_space_dim)
        returns = {}
        for state in range(self.env.state_space_dim):
            returns[state] = []
        for _ in tqdm(range(max_episode_num), desc = "Episode"):
            buffer = rollout(self.env, policy, max_episode_length = max_episode_length)
            G = 0
            while buffer:
                state, action, reward = buffer.pop()
                G = reward + gamma*G
                if not any(state == s and action == a for s, a, _ in buffer):
                    returns[state].append(G)
                v[state] = np.mean(returns[state])
        return v

    def on_policy_first_visit_mc_control_ES(self, max_episode_num=100, max_episode_length=1000, gamma=0.99, epsilon=0.1):
        policy = self.policy_class.uniform_random_policy()
        q = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        returns = {}
        init_state = np.random.choice(self.env.state_space)
        init_action = np.random.choice(self.env.action_space)
        for _ in tqdm(range(max_episode_num), desc="Episode"):
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

    def on_policy_first_visit_mc_control(self, max_episode_num=100, max_episode_length=1000, gamma=0.99, epsilon=0.1):
        policy = self.policy_class.uniform_random_policy()
        q = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        returns = {}
        for state in range(self.env.state_space_dim):
            for action in range(self.env.action_space_dim):
                returns[(state, action)] = []
        for _ in tqdm(range(max_episode_num), desc="Episode"):
            buffer = rollout(self.env, policy, max_episode_length=max_episode_length)
            G = 0
            while buffer:
                state, action, reward = buffer.pop()
                G = reward + gamma*G
                if not any(state == s and action == a for s, a, _ in buffer):
                    returns[(state, action)].append(G)
                    q[state, action] = np.mean(returns[(state, action)])
                    policy = self.policy_class.epsilon_greedy_policy(q, epsilon)
        return q, policy

    def off_policy_mc_control(self, max_episode_num=100, max_episode_length=1000, gamma=0.99, epsilon=0.1):
        uniform_policy = self.policy_class.uniform_random_policy()
        q = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        c = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        returns = {}
        for state in range(self.env.state_space_dim):
            for action in range(self.env.action_space_dim):
                returns[(state, action)] = []
        for _ in tqdm(range(max_episode_num), desc="Episode"):
            buffer = rollout(self.env, uniform_policy, max_episode_length=max_episode_length)
            G = 0
            W = 1
            while buffer:
                state, action, reward = buffer.pop()
                G = reward + gamma*G
                c[state, action] += W
                q[state, action] += (W/c[state, action])*(G - q[state, action])
                policy = self.policy_class.greedy_policy(q)
                if action == policy[state, 0]:
                    W = W * (1/uniform_policy[state, action])
                else:
                    break
        return q, policy

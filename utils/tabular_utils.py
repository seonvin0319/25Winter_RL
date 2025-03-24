import numpy as np
from collections import deque

class TabularPolicy:
    def __init__(self, env):
        self.env = env
        self.iter_num = 0

    def uniform_random_policy(self):
        return np.ones((self.env.state_space_dim, self.env.action_space_dim)) / self.env.action_space_dim

    def greedy_policy(self, q):
        policy = np.zeros((self.env.state_space_dim, self.env.action_space_dim))
        policy[np.arange(self.env.state_space_dim), np.argmax(q, axis = 1)] = 1
        return policy

    def epsilon_greedy_policy(self, q, epsilon, epsilon_decay = 0.99):
        epsilon *= epsilon_decay ** (self.iter_num // 100)
        policy = np.ones((self.env.state_space_dim, self.env.action_space_dim)) * epsilon / self.env.action_space_dim
        policy[np.arange(self.env.state_space_dim), np.argmax(q, axis = 1)] += 1 - epsilon
        self.iter_num += 1
        return policy

def rollout(env, policy, initial_state = None, initial_action = None, max_episode_length = 1000):
    buffer = deque(maxlen = max_episode_length)
    if initial_state is None:
        state = env.reset()
    else:
        state = env.reset(initial_state)
    if initial_action is None:
        action = np.random.choice(env.action_space, p=policy[state])
    else:
        action = initial_action
    for _ in range(max_episode_length):
        next_state, reward, done = env.step(action)
        buffer.append((state, action, reward))
        state = next_state
        action = np.random.choice(env.action_space, p=policy[state])
        if done:
            break
    return buffer

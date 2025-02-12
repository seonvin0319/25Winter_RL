import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class Env:
    def __init__(self, init_state = 0, p_head = 0.5, r_head = 1, r_tail = 0):
        # State: 0 = Head, 1 = Tail
        # Action: 0 = Flip, 1 = Pass
        # Reward: 1 if Head, 0 if Tail
        # Probability of Head: p
        # Reward for Head: r[0], Reward for Tail: r[1]

        self.state_space = np.array([0, 1])
        self.action_space = np.array([0, 1])

        self.state_dim = len(self.state_space[0].shape)
        self.state_space_dim = len(self.state_space)
        self.action_dim = len(self.action_space[0].shape)
        self.action_space_dim = len(self.action_space)
        self.init_state = init_state

        self.p_head = p_head
        self.p_tail = 1 - p_head
        r = np.array([r_head, r_tail])
        self.transition_prob = np.array([[[self.p_head, 1-self.p_head], [1, 0]], [[1-self.p_head, self.p_head], [0, 1]]]) # P(s'|s,a)
        self.reward = np.tile(r, (self.state_space_dim, self.state_space_dim))  # R(s,s')

    def step(self, action):
        p_sa = self.transition_prob[self.state, action]
        next_state = np.random.choice(self.state_space, p=p_sa)
        reward = self.reward[self.state, next_state]
        self.state = next_state
        return next_state, reward
    
    def reset(self, state = None):
        if state is None:
            self.state = self.init_state
        else:
            self.state = state
        return self.state

    def true_v(self, policy, gamma = 0.99):
        p1 = policy[0, 0]
        p2 = policy[1, 0]
        den = 2 - (2-p1-p2) * gamma
        v1 = 1/(1-gamma)*(1-p1/den)
        v2 = 1/(1-gamma)*p2/den
        return np.array([v1, v2])
    
    def state_to_str(self, state):
        return 'Head' if state == 0 else 'Tail'
    
    def action_to_str(self, action):
        return 'Flip' if action == 0 else 'Pass'

if __name__ == '__main__':
    p_head = 0.5 # Unbiased coin
    r_head = 1
    r_tail = 0
    gamma = 0.99
    env = Env(p_head = p_head, r_head = r_head, r_tail = r_tail)
    uniform_policy = np.ones((env.state_space_dim, env.action_space_dim)) / env.action_space_dim
    true_v = env.true_v(uniform_policy, gamma = gamma)
    print(f'V(Head) = {true_v[0]:.2f}, V(Tail) = {true_v[1]:.2f}')

    init_state = 0
    env.reset(init_state)
    state = env.state
    total_iter = 500
    total_steps = 1000
    returns = np.zeros(total_iter)
    for iter in tqdm(range(total_iter)):
        for step in range(total_steps):
            # print('-'*50)
            # print(f'Step {step}:')
            # print(f'Current state: {env.state_to_str(state)}')
            sample_action = np.random.choice(env.action_space, p=uniform_policy[state])
            # print(f'Action: {env.action_to_str(sample_action)}')
            next_state, reward = env.step(sample_action)
            # print(f'Next state: {env.state_to_str(next_state)}, Reward: {reward}')
            returns[iter] += reward * gamma**step
            state = next_state
    mean_returns = np.mean(returns)
    print(f'Expected Returns for {env.state_to_str(init_state)}: {mean_returns:.2f}')
    # Plot the histogram of returns
    plt.figure()
    plt.hist(returns, bins=20, density=True)
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.axvline(true_v[init_state], color='red', label='True Value', linewidth=2)
    plt.axvline(mean_returns, color='green', label='Estimated Value', linewidth=2)
    plt.title(f'Histogram of Returns for {env.state_to_str(init_state)}')
    plt.legend()
    plt.show()

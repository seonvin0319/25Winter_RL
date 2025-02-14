import numpy as np

class Env:
    def __init__(self, init_state = None, p_head = 0.5, r_head = 1, r_tail = 0):
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
        if init_state is None:
            self.init_state = np.random.choice(self.state_space)
        else:
            self.init_state = init_state
        self.done = False

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
        return next_state, reward, False
    
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
    
    def true_q(self, v):
        q = np.zeros((self.state_space_dim, self.action_space_dim))
        q[0, 0] = self.p_head * v[0] + (1 - self.p_head) * v[1]
        q[0, 1] = v[0]
        q[1, 0] = q[0, 0]
        q[1, 1] = v[1]
        return q
    
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
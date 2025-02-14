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
        self.r_head = r_head
        self.r_tail = r_tail
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
        """
        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape (state_space_dim, action_space_dim)
        gamma : float
            Discount factor
        """
        p_th = policy[0, 0]
        p_tt = policy[1, 0]
        tr_h = self.p_head
        g = gamma
        r_h = self.r_head
        r_t = self.r_tail
        Den = g**2*p_th*tr_h - g**2*p_th - g**2*p_tt*tr_h + g**2 - g*p_th*tr_h + g*p_th + g*p_tt*tr_h - 2*g + 1
        num_vh = -g*p_th*r_h*tr_h + g*p_th*r_h + g*p_tt*r_h*tr_h - g*r_h + p_th*r_h*tr_h - p_th*r_h - p_th*r_t*tr_h + p_th*r_t + r_h
        num_vt = -g*p_th*r_t*tr_h + g*p_th*r_t + g*p_tt*r_t*tr_h - g*r_t + p_tt*r_h*tr_h - p_tt*r_t*tr_h + r_t
        v_h = num_vh / Den
        v_t = num_vt / Den
        return np.array([v_h, v_t])
    
    def true_q(self, v, gamma = 0.99):
        """
        Parameters
        ----------
        v : numpy.ndarray
            Value function of shape (state_space_dim,)
        """
        q = np.zeros((self.state_space_dim, self.action_space_dim))
        for s in self.state_space:
            for a in self.action_space:
                p_sa = self.transition_prob[s, a]
                for next_s in self.state_space:
                    q[s, a] += p_sa[next_s] * (self.reward[s, next_s] + gamma * v[next_s])
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
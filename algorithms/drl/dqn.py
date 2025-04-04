import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 128):
        super(MLP, self).__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.mlp = nn.Sequential(
            self.fc1,
            self.activation,
            self.fc2
        )

    def forward(self, x):
        return self.mlp(x)

class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states),
                np.array(actions),
                np.array(rewards),
                np.stack(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim = 128, buffer_capacity = 10000, lr = 1e-3, gamma = 0.99, eps = 1.0, eps_min = 0.01, eps_decay = 0.995, update_frequency = 100, target_net_hard_update = True, device= None):
        super(DQN, self).__init__()

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_target = MLP(state_dim, action_dim, hidden_dim).to(self.device)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=1e-3)
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.update_frequency = update_frequency
        self.target_net_hard_update = target_net_hard_update
        self.update_counter = 0
        self.update_target_network(tau = 1.0)

    def _loss_info(self):
        """CSV 파일의 헤더 정보 반환"""
        return ['loss']

    def sample_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_net(state)
                return q_values.argmax().item()

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def update_target_network(self, tau = 0.005):
        if self.target_net_hard_update:
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(param.data)
        else:
            for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, batch_size):
        self.q_net.train()
        if len(self.replay_buffer) < batch_size:
            return [0.0]

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.q_target(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decay_epsilon()
        self.update_counter += 1
        if self.target_net_hard_update:
            if self.update_counter % self.update_frequency == 0:
                self.update_target_network()
        else:
            self.update_target_network()

        return [loss.item()]

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))

    def eval(self):
        self.q_net.eval()
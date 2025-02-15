import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from colorama import Fore, Style

class PG(nn.Module):
    def __init__(self, env, hidden_dim=32, embedding_dim=16, layer_type='softmax', distance='L2', lr=0.001, gamma=0.99, n_layers=2):
        super(PG, self).__init__()

        # Env
        self.env = env
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        # Store hyperparameters
        self.lr = lr
        self.gamma = gamma

        # Network architecture
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        layers = []
        in_dim = embedding_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        # Harmonic layers
        self.layer_type = layer_type
        self.distance = distance

        if layer_type == 'softmax':
            self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        else:
            self.weights = nn.Parameter(torch.randn(output_dim, hidden_dim))

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x_embedded = self.embedding(x.argmax(dim=1))
        features = self.mlp(x_embedded)

        if self.layer_type == 'softmax':
            logits = self.output_layer(features)
            return torch.softmax(logits, dim=1)
        else:
            p = 1 if self.distance == 'L1' else 2
            dist = torch.cdist(features.unsqueeze(0), self.weights.unsqueeze(0), p=p).squeeze(0)
            inv_dist = 1 / (dist**2 + 1e-8)
            return inv_dist / inv_dist.sum(dim=1, keepdim=True)

    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = self.forward(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
        return action

    def store_outcome(self, log_prob, reward):
        self.saved_log_probs.append(log_prob)
        self.rewards.append(reward)

    def learn(self, total_timesteps=10000, callback=None):
        timestep = 0
        try:
            while timestep < total_timesteps:
                state = self.env.reset()
                episode_rewards = []
                self.saved_log_probs = []
                done = False
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action_probs = self.forward(state_tensor)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    next_state, reward, done, info = self.env.step(action.item())
                    self.store_outcome(log_prob, reward)
                    episode_rewards.append(reward)
                    state = next_state
                    timestep += 1

                    if callback is not None:
                        callback._on_step(reward)

                    if timestep >= total_timesteps:
                        break

                self._update_policy()
        except StopIteration:
            #print(Fore.YELLOW + "Training stopped early due to early stopping condition." + Style.RESET_ALL)
            print('')


    def _update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import matplotlib.pyplot as plt

class CustomA2C(nn.Module):
    def __init__(self, env, hidden_dim=64, gamma=0.99, n_steps=5, lr=0.001, prob_layer='softmax'):
        """
        Initialize the A2C agent.
        :param env: The Gym environment.
        :param hidden_dim: Number of units in hidden layers.
        :param gamma: Discount factor.
        :param n_steps: Number of steps to collect per update.
        :param lr: Learning rate.
        """
        super(CustomA2C, self).__init__()
        self.env = env
        self.gamma = gamma
        self.n_steps = n_steps

        # Determine input dimension from the environment's observation space.
        if isinstance(env.observation_space, spaces.Discrete):
            self.input_dim = env.observation_space.n
        else:
            self.input_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = env.action_space.n

        # Build a simple shared network (feature extractor)
        self.policy_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head: outputs logits for each action.
        self.actor = nn.Linear(hidden_dim, self.action_dim)
        # Critic head: outputs a single value.
        self.critic = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        """
        Forward pass that outputs both logits and state value.
        :param x: Input tensor of shape [batch, input_dim].
        """
        features = self.policy_net(x)
        logits = self.actor(features)
        value = self.critic(features)

        if self.prob_layer == 'softmax':
            probs = 
        elif self.prob_layer == 'harmax':
            probs = 

        return probs, value

    def preprocess(self, obs):
        """
        Preprocess an observation.
        For a Discrete observation, convert the integer state to a one-hot vector.
        For Box observations, convert the array to a FloatTensor.
        """
        if isinstance(self.env.observation_space, spaces.Discrete):
            one_hot = np.zeros(self.input_dim, dtype=np.float32)
            one_hot[obs] = 1.0
            return torch.tensor(one_hot).unsqueeze(0)  # Shape: [1, input_dim]
        else:
            return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    def predict(self, obs, deterministic=True):
        """
        Predict an action given an observation.
        :param obs: The observation from the environment.
        :param deterministic: If True, select the action with highest probability.
        :return: Selected action.
        """
        obs_tensor = self.preprocess(obs)  # Shape: [1, input_dim]
        logits, _ = self.forward(obs_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()
        return action.cpu().numpy()[0]

    def learn(self, total_timesteps):
        """
        Train the agent using an n-step update.
        :param total_timesteps: Total number of timesteps for training.
        :return: List of episode returns.
        """
        obs = self.env.reset()
        ep_rewards = []
        episode_returns = []
        timestep = 0
        rewards_all = []

        while timestep < total_timesteps:
            states, actions, rewards, dones, values = [], [], [], [], []

            # Collect rollout
            for _ in range(self.n_steps):
                obs_tensor = self.preprocess(obs)
                logits, value = self.forward(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()

                next_obs, reward, done, _ = self.env.step(action.item())
                rewards_all.append(reward)

                states.append(obs_tensor)
                actions.append(action)
                rewards.append(torch.tensor([reward], dtype=torch.float32))
                dones.append(torch.tensor([done], dtype=torch.float32))
                values.append(value)

                ep_rewards.append(reward)
                obs = next_obs
                timestep += 1

                if done:
                    episode_returns.append(np.sum(ep_rewards))
                    ep_rewards = []
                    obs = self.env.reset()
                    break

            # Bootstrap from the last state if not done
            if not done:
                next_obs_tensor = self.preprocess(obs)
                _, next_value = self.forward(next_obs_tensor)
                next_value = next_value.detach()
            else:
                next_value = torch.tensor([[0.0]])

            # Compute n-step returns (backwards)
            returns = []
            R = next_value
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + self.gamma * R * (1 - d)
                returns.insert(0, R)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            actions = torch.cat(actions)

            advantage = returns - values

            # Compute policy loss and value loss
            logits, _ = self.forward(torch.cat(states))
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return rewards_all

# Example usage:
if __name__ == "__main__":
    # Create environment (for example, FrozenLake from Gym)
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name="4x4")

    # Instantiate custom A2C agent
    agent = CustomA2C(env, hidden_dim=64, gamma=0.99, n_steps=5, lr=0.001)

    # Train for a given number of timesteps
    total_timesteps = 5000
    rewards_all = agent.learn(total_timesteps)
    #print("Episode returns:", returns)
    plt.plot(rewards_all)
    plt.show()

    # Evaluate the agent
    test_rewards = []
    for _ in range(100):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)
    print("Average test return:", np.mean(test_rewards))


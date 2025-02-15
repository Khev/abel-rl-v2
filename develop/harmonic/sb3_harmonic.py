import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from gymnasium.envs.toy_text import FrozenLakeEnv


class HarmonicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16, p=2):
        super(HarmonicLayer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.weights = nn.Parameter(torch.randn(output_dim, embedding_dim))
        self.p = p  # Distance metric: p=2 for L2, p=1 for L1

    def forward(self, x):
        # x is expected to be a one-hot vector; we take argmax to get indices
        x_embedded = self.embedding(x.argmax(dim=1))
        dist = torch.cdist(
            x_embedded.unsqueeze(0),
            self.weights.unsqueeze(0),
            p=self.p
        ).squeeze(0)
        inv_dist = 1 / (dist**2 + 1e-8)
        return inv_dist / inv_dist.sum(dim=1, keepdim=True)


class HarmonicFeaturesExtractor(BaseFeaturesExtractor):
    """
    Robust feature extractor using HarmonicLayer for A2C.
    Supports both Discrete and Box observation spaces.
    Requires 'action_space' in features_extractor_kwargs.
    The extractor's output dimension is set to action_space.n (number of actions),
    which is what the harmonic layer outputs.
    """
    def __init__(self, observation_space, action_space, embedding_dim=16):
        # Determine input_dim based on the observation space type:
        if isinstance(observation_space, spaces.Discrete):
            input_dim = observation_space.n
        elif isinstance(observation_space, spaces.Box):
            input_dim = int(np.prod(observation_space.shape))
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")

        output_dim = action_space.n  # number of actions (e.g., 4 for FrozenLake)

        # IMPORTANT: Set features_dim to output_dim, because the harmonic layer outputs a vector of length output_dim.
        super(HarmonicFeaturesExtractor, self).__init__(observation_space, features_dim=output_dim)
        self.observation_space = observation_space  # Save for later use

        self.harmonic_layer = HarmonicLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            embedding_dim=embedding_dim
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        For Discrete observations, assumes one-hot encoded input.
        For Box observations, flattens the input.
        """
        if isinstance(self.observation_space, spaces.Discrete):
            return self.harmonic_layer(observations)
        elif isinstance(self.observation_space, spaces.Box):
            flat_obs = observations.view(observations.size(0), -1)
            return self.harmonic_layer(flat_obs)



class TrainingProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        episode_rewards = self.locals['rewards']
        if len(episode_rewards) > 0:
            self.rewards.append(np.mean(episode_rewards))
        return True


def evaluate(agent, env, episodes=100):
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


# Env
def frozen_lake_env():
    return FrozenLakeEnv(render_mode=None, is_slippery=False, map_name="4x4")
env = DummyVecEnv([lambda: frozen_lake_env()])

# Agents
agent = A2C( "MlpPolicy", env)

breakpoint()

agent_harmonic = A2C(
    policy="MlpPolicy",
    env=env, 
    policy_kwargs={
        "features_extractor_class": HarmonicFeaturesExtractor,
        "features_extractor_kwargs": {
            "embedding_dim": 16,
            "action_space": env.action_space  # This is required!
        }
    }
)

# Train
Ntrain = 1000
callback_softmax = TrainingProgressCallback()
callback_harmonic = TrainingProgressCallback()

print("\nTraining A2C (Softmax) ...")
agent.learn(total_timesteps=Ntrain, callback=callback_softmax)
print("\nTraining A2C (Harmonic) ...")
agent_harmonic.learn(total_timesteps=Ntrain, callback=callback_harmonic)

# Evalauge
softmax_mean, softmax_std = evaluate(agent, env)
harmonic_mean, harmonic_std = evaluate(agent_harmonic, env)
print(f"A2C (Softmax): Mean Return: {softmax_mean:.2f}, Std: {softmax_std:.2f}")
print(f"A2C (Harmonic): Mean Return: {harmonic_mean:.2f}, Std: {harmonic_std:.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(callback_softmax.rewards, label="A2C (Softmax)", alpha=0.7)
plt.plot(callback_harmonic.rewards, label="A2C (Harmonic)", alpha=0.7)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Learning Curves: A2C Softmax vs Harmonic")
plt.legend()
plt.grid(True)
plt.show()

# Save modle
agent.save("a2c_softmax")
agent_harmonic.save("a2c_harmonic")
print("✅ Models saved!")

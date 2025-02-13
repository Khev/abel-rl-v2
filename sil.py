import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import random
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def log_with_timestamp(message):
    """Print a message with a timestamp."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

# ------------------------- Smoothed Reward Callback ------------------------- #
class SmoothedRewardCallback(BaseCallback):
    """Track rewards and success rates during training."""
    def __init__(self, total_timesteps, window_fraction=0.1, verbose=1):
        super().__init__(verbose)
        self.rewards = []
        self.successes = []
        self.window_size = int(total_timesteps * window_fraction)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0] if "rewards" in self.locals else None
        done = self.locals["dones"][0] if "dones" in self.locals else None

        if reward is not None:
            self.rewards.append(reward)
        if done is not None:
            self.successes.append(1 if reward == 1 else 0)

        return True

# ------------------------- SIL Buffer ------------------------- #
class SILBuffer:
    """Buffer for Self-Imitation Learning."""
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add_experience(self, state, action, reward, value):
        """Add single experience if return > value estimate."""
        if reward > value and len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward))

    def sample(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

# --------------------- SIL Update Callback --------------------- #
class SILCallback(BaseCallback):
    """Apply SIL updates every n_steps (same as A2C updates)."""
    def __init__(self, model, sil_buffer, optimizer, batch_size=64, sil_updates=4, n_steps=5):
        super().__init__()
        self.model = model
        self.sil_buffer = sil_buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.sil_updates = sil_updates
        self.n_steps = n_steps
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        # Add experiences to SIL buffer every n_steps (like A2C updates)
        if "rewards" in self.locals and "actions" in self.locals and "obs" in self.locals:
            reward = self.locals["rewards"][0]
            action = self.locals["actions"][0]
            state = self.locals["obs"][0]
            value = self.model.policy.predict_values(torch.tensor(state).float().unsqueeze(0)).item()
            self.sil_buffer.add_experience(state, action, reward, value)

        # Perform SIL updates every n_steps
        if self.step_count % self.n_steps == 0:
            for _ in range(self.sil_updates):
                samples = self.sil_buffer.sample(self.batch_size)
                if samples is None:
                    continue

                states, actions, returns = zip(*samples)
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                returns = torch.tensor(returns, dtype=torch.float32)

                policy, value = self.model.policy.forward(states)
                log_probs = torch.log_softmax(policy, dim=1)
                action_log_probs = log_probs[range(self.batch_size), actions]

                advantages = (returns - value.squeeze()).clamp(min=0)
                policy_loss = -(advantages * action_log_probs).mean()
                value_loss = 0.5 * (advantages.pow(2)).mean()

                loss = policy_loss + 0.01 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return True


# ------------------------- Train and Plot ------------------------- #
def moving_average(data, window_size=100):
    """Compute a moving average using pandas."""
    if len(data) < window_size:
        return data
    return pd.Series(data).rolling(window=window_size).mean().dropna().tolist()

def success_rate(data, window_size=100):
    """Compute the moving average of success rate."""
    if len(data) < window_size:
        return data
    return pd.Series(data).rolling(window=window_size).mean().dropna().tolist()

# ------------------------- Main Execution ------------------------- #
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # Ensures compatibility with macOS

    # Parameters
    Ntrain = 5*10**4  # total training timesteps
    buffer_size = 10**5

    # Create environment
    num_envs = 5
    vec_env = SubprocVecEnv([lambda: gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)] * num_envs)

    # ------------------------- Baseline: A2C ------------------------- #
    baseline_callback = SmoothedRewardCallback(total_timesteps=Ntrain)
    model_a2c = A2C("MlpPolicy", vec_env, verbose=0)
    log_with_timestamp('Training baseline A2C')
    model_a2c.learn(total_timesteps=Ntrain, callback=baseline_callback)

    # ------------------------- A2C + SIL ------------------------- #
    sil_callback = SmoothedRewardCallback(total_timesteps=Ntrain)
    sil_buffer = SILBuffer(capacity=buffer_size)
    sil_optimizer = torch.optim.Adam(model_a2c.policy.parameters(), lr=1e-4)
    sil_update_callback = SILCallback(model_a2c, sil_buffer, sil_optimizer)

    model_a2c_sil = A2C("MlpPolicy", vec_env, verbose=0)
    log_with_timestamp('Training A2C + SIL')
    model_a2c_sil.learn(total_timesteps=Ntrain, callback=[sil_callback, sil_update_callback])

    # ------------------------- Plot 1x2: Rewards and Success Rate ------------------------- #
    log_with_timestamp('Plotting rewards')
    baseline_reward_smoothed = moving_average(baseline_callback.rewards, window_size=int(0.1 * Ntrain))
    sil_reward_smoothed = moving_average(sil_callback.rewards, window_size=int(0.1 * Ntrain))

    baseline_success_smoothed = success_rate(baseline_callback.successes, window_size=int(0.1 * Ntrain))
    sil_success_smoothed = success_rate(sil_callback.successes, window_size=int(0.1 * Ntrain))

    plt.figure(figsize=(12, 5))

    # Plot 1: Smoothed Rewards
    plt.subplot(1, 2, 1)
    plt.plot(baseline_reward_smoothed, label="A2C (Reward)", alpha=0.7, color="blue")
    plt.plot(sil_reward_smoothed, label="A2C + SIL (Reward)", alpha=0.7, color="red")
    plt.xlabel("Timesteps")
    plt.ylabel("Smoothed Reward")
    plt.title("Smoothed Reward Comparison")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Plot 2: Success Rates
    plt.subplot(1, 2, 2)
    plt.plot(baseline_success_smoothed, label="A2C (Success Rate)", alpha=0.7, color="blue")
    plt.plot(sil_success_smoothed, label="A2C + SIL (Success Rate)", alpha=0.7, color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Comparison")
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

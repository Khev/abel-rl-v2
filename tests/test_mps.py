import time
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Function to check available devices
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple M1/M2 GPU (MPS)")
        return "mps"
    else:
        print("Using CPU")
        return "cpu"

# Create Pendulum environment (better suited for SAC)
env = gym.make("Pendulum-v1")
env = DummyVecEnv([lambda: env])

# Training parameters
N_TRAIN_STEPS = 10**5  # More steps to allow SAC to learn
policy = "MlpPolicy"

# Define models dictionary
models = {}

# Train SAC on CPU and MPS
for device_type in ["cpu", "mps"]:
    if device_type == "mps" and not torch.backends.mps.is_available():
        continue  # Skip if MPS is not available

    print(f"\nTraining on {device_type.upper()}...\n")
    start_time = time.time()

    # Train SAC model
    model = SAC(policy, env, device=device_type, verbose=0, batch_size=1024, policy_kwargs={'net_arch': [512, 512, 512]})
    model.learn(total_timesteps=N_TRAIN_STEPS)

    end_time = time.time()
    models[device_type] = {"model": model, "time": end_time - start_time}

    print(f"\nTraining on {device_type.upper()} completed in {models[device_type]['time']:.2f} seconds\n")

# Compare results
if "mps" in models and "cpu" in models:
    speedup = models["cpu"]["time"] / models["mps"]["time"]
    print(f"\nSpeedup with MPS: {speedup:.2f}x faster than CPU")
else:
    print("\nMPS training not available, only CPU results shown.")

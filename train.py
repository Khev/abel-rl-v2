# train.py
import boto3
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

bucket = 'abel-rl'
s3 = boto3.client('s3')

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=0)

# Evaluate before training
def evaluate(model, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        done = False
        obs = env.reset()[0]
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

acc_before = evaluate(model)
print(f"🎯 Initial reward: {acc_before:.2f}")

# Train
model.learn(total_timesteps=1000)

acc_after = evaluate(model)
print(f"✅ Final reward: {acc_after:.2f}")

# Save results locally
with open('cartpole_results.txt', 'w') as f:
    f.write(f"Initial reward: {acc_before:.2f}\n")
    f.write(f"Final reward: {acc_after:.2f}\n")

# Upload to S3
s3.upload_file('data/cartpole_results.txt', bucket, 'cartpole_results.txt')
print(f"📤 Uploaded results to s3://{bucket}/data/cartpole_results.txt")

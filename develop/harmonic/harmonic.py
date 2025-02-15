import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Toy Environment (2-state, 2-action) ---
class ToyEnv:
    def __init__(self, max_steps=20):
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        self.state = 0
        self.current_step = 0
        return self.state

    def step(self, action):
        reward = 1 if (self.state == 0 and action == 1) or (self.state == 1 and action == 1) else 0
        self.state = 1 - self.state  # Alternate between states
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.state, reward, done, {}

# --- Harmonic Layer (HarMax) ---
class HarmonicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n=2):
        super(HarmonicLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim))
        self.n = n

    def forward(self, x):
        dist = torch.cdist(x, self.weights, p=2)  # L2 distance
        probs = 1 / (dist**self.n + 1e-8)
        return probs / probs.sum(dim=1, keepdim=True)

# --- Softmax Policy ---
class SoftmaxPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxPolicy, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        logits = self.fc(x)
        return torch.softmax(logits, dim=1)

# --- REINFORCE Training Function ---
def train(policy, env, optimizer, episodes=10000, baseline_name=""):
    reinforce_losses = []
    cross_entropy_losses = []

    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        while True:
            state_tensor = torch.tensor([[state]], dtype=torch.float32)
            action_probs = policy(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)

            if done:
                break
            state = next_state

        # Compute REINFORCE loss
        returns = sum(rewards)
        reinforce_loss = -sum([log_prob * returns for log_prob in log_probs])


        optimizer.zero_grad()
        reinforce_loss.backward()
        optimizer.step()

        reinforce_losses.append(reinforce_loss.item())

    # Print REINFORCE loss every 100 episodes
        if episode % 100 == 0:
            print(f"{baseline_name} Episode {episode}, Loss: {reinforce_loss.item()}")


    return reinforce_losses, _

# --- Smoothing Function for Plotting ---
def smooth_curve(data, smoothing_factor=0.9):
    smoothed = []
    avg = data[0]
    for value in data:
        avg = smoothing_factor * avg + (1 - smoothing_factor) * value
        smoothed.append(avg)
    return smoothed

# --- Main Execution ---

num_eps = 50000
env = ToyEnv()

# Harmonic Policy
harmonic_policy = HarmonicLayer(input_dim=1, output_dim=2, n=2)
harmonic_optimizer = optim.Adam(harmonic_policy.parameters(), lr=0.01)
harmonic_reinforce_losses, harmonic_cross_entropy_losses = train(
    harmonic_policy, env, harmonic_optimizer, baseline_name="Harmonic", episodes=num_eps
)
print('\n \n ')

# Softmax Policy
softmax_policy = SoftmaxPolicy(input_dim=1, output_dim=2)
softmax_optimizer = optim.Adam(softmax_policy.parameters(), lr=0.01)
softmax_reinforce_losses, softmax_cross_entropy_losses = train(
    softmax_policy, env, softmax_optimizer, baseline_name="Softmax", episodes=num_eps
)

# --- Print Action Probabilities for Both States ---
print("\n--- Final Action Probabilities ---")

# Harmonic Policy
for state in range(2):
    state_tensor = torch.tensor([[state]], dtype=torch.float32)
    action_probs = harmonic_policy(state_tensor).detach().numpy()
    print(f"Harmonic Policy - State {state}: Action Probabilities: {action_probs}")

# Softmax Policy
for state in range(2):
    state_tensor = torch.tensor([[state]], dtype=torch.float32)
    action_probs = softmax_policy(state_tensor).detach().numpy()
    print(f"Softmax Policy - State {state}: Action Probabilities: {action_probs}")


# --- Plotting ---
plt.figure(figsize=(10, 12))

# REINFORCE Loss Plot
plt.subplot(2, 1, 1)
plt.plot(smooth_curve(harmonic_reinforce_losses), label='Harmonic REINFORCE', alpha=0.7)
plt.plot(smooth_curve(softmax_reinforce_losses), label='Softmax REINFORCE', alpha=0.7)
plt.xlabel('Episodes')
plt.ylabel('REINFORCE Loss')
plt.title('REINFORCE Loss Comparison')
plt.legend()
plt.grid(True)


# Learned Weights Comparison
plt.subplot(2, 1, 2)
plt.plot(softmax_policy.fc.weight.data.numpy().flatten(), 'o-', label='Softmax Weights')
plt.plot(harmonic_policy.weights.data.numpy().flatten(), 'x-', label='Harmonic Weights')
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.title('Learned Weights Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os

# ============================
# --- Improved GridWorld Environment ---
# ============================
class ToyEnv2:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacles=None, max_steps=20):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()
        self.max_steps = max_steps
        self.current_step = 0
        self.state = self.start  # Initial position of the agent
        
        # Observation and Action dimensions
        self.obs_dim = grid_size[0] * grid_size[1]  # One-hot encoding for each grid cell
        self.action_dim = 4  # Up, Down, Left, Right
        
        # Rewards
        self.goal_reward = 10
        self.step_reward = -0.01
        self.obstacle_penalty = -1
        self.timeout_penalty = 0
        self.illegal_penalty = -0.2  # Penalty for hitting boundaries

    def get_state_index(self, state):
        """Convert (x, y) position to flattened state index."""
        return state[0] * self.grid_size[1] + state[1]

    def to_one_hot(self, index):
        """Return one-hot encoded state vector."""
        state = np.zeros(self.obs_dim, dtype=np.float32)
        state[index] = 1.0
        return state

    def reset(self):
        """Reset the environment to the start state."""
        self.state = self.start
        self.current_step = 0
        return self.to_one_hot(self.get_state_index(self.state))

    def step(self, action):
        """Take a step based on the action and return (state, reward, done, info)."""
        if action not in range(self.action_dim):
            raise ValueError(f"Invalid action: {action}. Action must be in [0, 1, 2, 3].")

        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

        dx, dy = moves[action]
        new_state = (self.state[0] + dx, self.state[1] + dy)

        # Check for illegal moves (boundary hit)
        if not (0 <= new_state[0] < self.grid_size[0] and 0 <= new_state[1] < self.grid_size[1]):
            reward = self.illegal_penalty
            done = False
            info = {"info": "Illegal Action (Boundary Hit)"}
            return self.to_one_hot(self.get_state_index(self.state)), reward, done, info

        # Handle obstacle collision
        if new_state in self.obstacles:
            reward = self.obstacle_penalty
            done = False
            info = {"info": "Hit Obstacle"}
        elif new_state == self.goal:
            reward = self.goal_reward
            done = True
            info = {"info": "Goal Reached"}
        elif self.current_step >= self.max_steps - 1:
            reward = self.timeout_penalty
            done = True
            info = {"info": "Max Steps Reached"}
        else:
            reward = self.step_reward
            done = False
            info = {"info": "Step Taken"}

        self.state = new_state
        self.current_step += 1

        return self.to_one_hot(self.get_state_index(new_state)), reward, done, info


# ============================
# --- Harmonic Layer (HarMax) ---
# ============================
class HarmonicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16, n=2):
        super(HarmonicLayer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.weights = nn.Parameter(torch.randn(output_dim, embedding_dim))
        self.n = n

    def forward(self, x):
        x_embedded = self.embedding(x.argmax(dim=1))  # Add embedding layer
        dist = torch.cdist(x_embedded.unsqueeze(0), self.weights.unsqueeze(0), p=2).squeeze(0)
        inv_dist = 1 / (dist**self.n + 1e-8)
        probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
        return probs


# ============================
# --- Harmonic Layer with L1 Distance ---
# ============================
class HarmonicLayerL1(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16, n=1):
        super(HarmonicLayerL1, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.weights = nn.Parameter(torch.randn(output_dim, embedding_dim))
        self.n = n

    def forward(self, x):
        x_embedded = self.embedding(x.argmax(dim=1))
        dist = torch.cdist(x_embedded.unsqueeze(0), self.weights.unsqueeze(0), p=1).squeeze(0)  # L1 distance
        inv_dist = 1 / (dist**self.n + 1e-8)
        probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
        return probs


# ============================
# --- Softmax Policy ---
# ============================
class SoftmaxPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16):
        super(SoftmaxPolicy, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim, bias=False)

    def forward(self, x):
        x_embedded = self.embedding(x.argmax(dim=1))
        logits = self.fc(x_embedded)
        return torch.softmax(logits, dim=1)


# ============================
# --- REINFORCE Training Function ---
# ============================
def train(policy, env, optimizer, episodes=10000, baseline_name=""):
    reinforce_losses = []

    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = policy(state_tensor)

            if len(action_probs.shape) == 1:
                action_probs = action_probs.unsqueeze(0)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            next_state, reward, done, _ = env.step(action.item())

            rewards.append(reward)
            if done:
                break
            state = next_state

        returns = sum(rewards)
        reinforce_loss = -sum([log_prob * returns for log_prob in log_probs])

        optimizer.zero_grad()
        reinforce_loss.backward()
        optimizer.step()

        reinforce_losses.append(reinforce_loss.item())

        if episode % 100 == 0:
            print(f"{baseline_name} Episode {episode}, Loss: {reinforce_loss.item():.2f}, Return: {returns:.2f}")

    return reinforce_losses


# ============================
# --- Agent Visualization Function ---
# ============================
def plot_agents_side_by_side(harmonic_policy, softmax_policy, env):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    policies = [("Harmonic Policy", harmonic_policy, axs[0]),
                ("Softmax Policy", softmax_policy, axs[1])]

    for title, policy, ax in policies:
        state = env.reset()
        trajectory = [env.state]

        for _ in range(env.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
            next_state, _, done, _ = env.step(action)
            trajectory.append(env.state)
            if done:
                break
            state = next_state

        grid = np.zeros(env.grid_size)
        for obs in env.obstacles:
            grid[obs] = -1
        grid[env.goal] = 2

        trajectory = np.array(trajectory)

        ax.imshow(grid, cmap="gray", origin="upper")
        ax.plot(trajectory[:, 1], trajectory[:, 0], marker="o", color="orange", label="Trajectory")
        ax.scatter(env.start[1], env.start[0], color="green", s=100, label="Start")
        ax.scatter(env.goal[1], env.goal[0], color="red", s=100, label="Goal")
        ax.set_title(title)
        ax.grid(True, color='white', linestyle='--', linewidth=0.7)
        ax.invert_yaxis()
        ax.legend()

    plt.suptitle("Comparison of Harmonic and Softmax Policy Trajectories")
    plt.show()


# ============================
# --- Plot Learning Curves ---
# ============================

def smooth_curve(data, window_length):
    """Apply a moving average to smooth the curve."""
    if window_length < 2:
        return data  # No smoothing for very small datasets
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_length:] - cumsum[:-window_length]) / window_length

def plot_learning_curves(results, episodes):
    plt.figure(figsize=(10, 6))

    window_length = int(0.1 * episodes)  # Smoothing window (10% of total episodes)

    for label, losses in results.items():
        # Apply smoothing
        losses = smooth_curve(losses, window_length)
        plt.plot(losses, label=f"{label} (Smoothed)")
    
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Learning Curves: Harmonic (L1/L2) vs Softmax (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()



# ============================
# --- Save Models ---
# ============================
def save_model(model, filename):
    os.makedirs('models', exist_ok=True)  # Create 'models' directory if it doesn't exist
    save_path = os.path.join('models', filename)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

# ============================
# --- Main Script ---
# ============================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compare Harmonic and Softmax policies on ToyEnv2.")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    args = parser.parse_args()

    num_eps = args.episodes

    # Create the Harder ToyEnv2 environment
    env = ToyEnv2(
        grid_size=(10, 10),
        start=(0, 0),
        goal=(9, 9),
        obstacles={
            (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
            (7, 4), (7, 5), (7, 6), (7, 7),
            (8, 7)
        },
        max_steps=50
    )

    # Create the Harder ToyEnv2 environment
    # env = ToyEnv2(
    #     grid_size=(5, 5),
    #     start=(0, 0),
    #     goal=(4, 4),
    #     obstacles={(2, 2)},  # Single obstacle in the center
    #     max_steps=20
    # )

    input_dim = env.obs_dim
    output_dim = env.action_dim

    results = {}
    lr = 0.01
    embedding_dim = 2

    # # --- Harmonic L2 Policy ---
    harmonic_l2_policy = HarmonicLayer(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim)
    harmonic_l2_optimizer = optim.Adam(harmonic_l2_policy.parameters(), lr=lr)
    harmonic_l2_losses = train(harmonic_l2_policy, env, harmonic_l2_optimizer, episodes=num_eps, baseline_name="Harmonic (L2)")
    save_model(harmonic_l2_policy, "harmonic_policy_l2.pth")
    results["Harmonic (L2)"] = harmonic_l2_losses

    # # --- Harmonic L1 Policy ---
    harmonic_l1_policy = HarmonicLayerL1(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim)
    harmonic_l1_optimizer = optim.Adam(harmonic_l1_policy.parameters(), lr=lr)
    harmonic_l1_losses = train(harmonic_l1_policy, env, harmonic_l1_optimizer, episodes=num_eps, baseline_name="Harmonic (L1)")
    save_model(harmonic_l1_policy, "harmonic_policy_l1.pth")
    results["Harmonic (L1)"] = harmonic_l1_losses

    # # --- Softmax Policy ---
    softmax_policy = SoftmaxPolicy(input_dim=input_dim, output_dim=output_dim,embedding_dim=embedding_dim)
    softmax_optimizer = optim.Adam(softmax_policy.parameters(), lr=lr)
    softmax_losses = train(softmax_policy, env, softmax_optimizer, episodes=num_eps, baseline_name="Softmax")
    save_model(softmax_policy, "softmax_policy.pth")
    results["Softmax"] = softmax_losses

    # # ============================
    # # --- Plot Learning Curves ---
    # # ============================
    plot_learning_curves(results, num_eps)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import multiprocessing
from tqdm import tqdm
from scipy.stats import ttest_ind

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
        self.state = self.start

        self.obs_dim = grid_size[0] * grid_size[1]
        self.action_dim = 4

        self.goal_reward = 10
        self.step_reward = -0.01
        self.obstacle_penalty = -1
        self.timeout_penalty = 0
        self.illegal_penalty = -0.2

    def get_state_index(self, state):
        return state[0] * self.grid_size[1] + state[1]

    def to_one_hot(self, index):
        state = np.zeros(self.obs_dim, dtype=np.float32)
        state[index] = 1.0
        return state

    def reset(self):
        self.state = self.start
        self.current_step = 0
        return self.to_one_hot(self.get_state_index(self.state))

    def step(self, action):
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        dx, dy = moves[action]
        new_state = (self.state[0] + dx, self.state[1] + dy)

        if not (0 <= new_state[0] < self.grid_size[0] and 0 <= new_state[1] < self.grid_size[1]):
            reward = self.illegal_penalty
            done = False
            return self.to_one_hot(self.get_state_index(self.state)), reward, done, {"info": "Illegal Action"}

        if new_state in self.obstacles:
            reward = self.obstacle_penalty
            done = False
        elif new_state == self.goal:
            reward = self.goal_reward
            done = True
        elif self.current_step >= self.max_steps - 1:
            reward = self.timeout_penalty
            done = True
        else:
            reward = self.step_reward
            done = False

        self.state = new_state
        self.current_step += 1
        return self.to_one_hot(self.get_state_index(new_state)), reward, done, {}


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
def train(policy, env, optimizer, episodes=10000):
    returns = []

    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = policy(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)

            if done:
                break
            state = next_state

        total_return = sum(rewards)
        returns.append(total_return)

        reinforce_loss = -sum([log_prob * total_return for log_prob in log_probs])
        optimizer.zero_grad()
        reinforce_loss.backward()
        optimizer.step()

    return np.array(returns)


# ============================
# --- Single Run Function (Parallelized) ---
# ============================
def single_run(args):
    """Single training run with provided parameters."""
    policy_class, env_config, optimizer_class, lr, episodes = args
    env = ToyEnv2(**env_config)
    policy = policy_class(input_dim=env.obs_dim, output_dim=env.action_dim)
    optimizer = optimizer_class(policy.parameters(), lr=lr)
    returns = train(policy, env, optimizer, episodes)
    return returns


# ============================
# --- Multi-Run Training (Parallel) ---
# ============================
def multi_run_train(policy_class, env_config, optimizer_class, lr, episodes, n_runs, num_workers, baseline_name):
    """Run multiple independent trainings in parallel (over runs)."""
    args_list = [(policy_class, env_config, optimizer_class, lr, episodes) for _ in range(n_runs)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(single_run, args_list),
            total=n_runs,
            desc=f"{baseline_name} Training"
        ))

    return np.array(results)


# ============================
# --- Combined Training ---
# ============================
def run_training(env_config, episodes, n_runs, lr, num_workers):
    """Run training for Harmonic and Softmax policies in parallel over runs."""
    harmonic_returns = multi_run_train(
        HarmonicLayer, env_config, torch.optim.Adam, lr, episodes, n_runs, num_workers, "Harmonic"
    )
    
    softmax_returns = multi_run_train(
        SoftmaxPolicy, env_config, torch.optim.Adam, lr, episodes, n_runs, num_workers, "Softmax"
    )
    
    return harmonic_returns, softmax_returns


# ============================
# --- Learning Curve Plot ---
# ============================
def plot_learning_curves(harmonic_returns, softmax_returns, episodes):
    plt.figure(figsize=(10, 6))

    # --- Harmonic Results (Min, Mean, Max) ---
    harmonic_mean = harmonic_returns.mean(axis=0)
    harmonic_min = harmonic_returns.min(axis=0)
    harmonic_max = harmonic_returns.max(axis=0)

    plt.plot(harmonic_mean, label='Harmonic (Mean)', color='blue')
    plt.fill_between(
        range(episodes),
        harmonic_min,
        harmonic_max,
        color='blue',
        alpha=0.2,
        label='Harmonic (Min/Max)'
    )

    # --- Softmax Results (Min, Mean, Max) ---
    softmax_mean = softmax_returns.mean(axis=0)
    softmax_min = softmax_returns.min(axis=0)
    softmax_max = softmax_returns.max(axis=0)

    plt.plot(softmax_mean, label='Softmax (Mean)', color='orange')
    plt.fill_between(
        range(episodes),
        softmax_min,
        softmax_max,
        color='orange',
        alpha=0.2,
        label='Softmax (Min/Max)'
    )

    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('Learning Curves: Harmonic vs Softmax (Min/Mean/Max)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()



# ============================
# --- Statistical Analysis (Last 10% Averaged) ---
# ============================
def statistical_test(harmonic_returns, softmax_returns):
    # Determine the last 10% of episodes
    num_episodes = harmonic_returns.shape[1]
    last_10_percent = max(1, int(0.1 * num_episodes))

    # Average over the last 10% of episodes
    harmonic_final = harmonic_returns[:, -last_10_percent:].mean(axis=1)
    softmax_final = softmax_returns[:, -last_10_percent:].mean(axis=1)

    # Perform t-test
    t_stat, p_value = ttest_ind(harmonic_final, softmax_final)

    # Print results
    print("\n📊 Statistical Test on Last 10% of Episode Returns (Averaged):")
    print(f" - Harmonic Mean: {harmonic_final.mean():.2f} ± {harmonic_final.std():.2f}")
    print(f" - Softmax Mean: {softmax_final.mean():.2f} ± {softmax_final.std():.2f}")
    print(f" - t-statistic: {t_stat:.2f}")
    print(f" - p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("✅ Statistically Significant Difference (p < 0.05)")
    else:
        print("⚠️ No Significant Difference (p >= 0.05)")



# ============================
# --- Main Script ---
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Harmonic and Softmax policies on ToyEnv2.")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes per run")
    parser.add_argument("--runs", type=int, default=8, help="Number of independent runs for averaging")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers (CPU cores)")
    args = parser.parse_args()

    # Required for multiprocessing on MacOS
    multiprocessing.set_start_method('spawn', force=True)

    # Create Environment Configuration
    env_config = {
        "grid_size": (10, 10),
        "start": (0, 0),
        "goal": (9, 9),
        "obstacles": {
            (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
            (7, 4), (7, 5), (7, 6), (7, 7),
            (8, 7)
        },
        "max_steps": 50
    }

    # Run Training
    harmonic_returns, softmax_returns = run_training(
        env_config, args.episodes, args.runs, args.lr, args.num_workers
    )

    # Plot Learning Curves
    plot_learning_curves(harmonic_returns, softmax_returns, args.episodes)

    # Perform Statistical Test
    statistical_test(harmonic_returns, softmax_returns)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os
import random  # For randomizing start/goal positions
import copy
from datetime import datetime  # For timestamp logging

# --- Simple timestamp logger ---
def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ============================
# --- Improved GridWorld Environment ---
# ============================
class ToyEnv2:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacles=None, max_steps=20,
                 randomize=False, possible_starts=None, possible_goals=None, mode="train"):
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
        
        # For train/test split: if randomize is True, then start/goal will be sampled.
        self.randomize = randomize
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        self.mode = mode  # "train" or "test"

    def get_state_index(self, state):
        """Convert (x, y) position to flattened state index."""
        return state[0] * self.grid_size[1] + state[1]

    def to_one_hot(self, index):
        """Return one-hot encoded state vector."""
        state = np.zeros(self.obs_dim, dtype=np.float32)
        state[index] = 1.0
        return state

    def reset(self):
        """Reset the environment to the (possibly randomized) start state."""
        if self.randomize:
            if self.possible_starts is not None:
                self.start = random.choice(self.possible_starts)
            if self.possible_goals is not None:
                self.goal = random.choice(self.possible_goals)
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

        if not (0 <= new_state[0] < self.grid_size[0] and 0 <= new_state[1] < self.grid_size[1]):
            reward = self.illegal_penalty
            done = False
            info = {}
            self.current_step += 1
            if self.current_step > self.max_steps:
                done = True
            return self.to_one_hot(self.get_state_index(self.state)), reward, done, info

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
        if self.current_step > self.max_steps:
            done = True

        return self.to_one_hot(self.get_state_index(new_state)), reward, done, info


# ============================
# --- Deep Harmonic Layer (HarMax) ---
# ============================
class HarmonicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16, hidden_dim=32, n=2):
        super(HarmonicLayer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # A deeper representation using an MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Learnable weights in the same hidden space
        self.weights = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.n = n

    def forward(self, x):
        x_embedded = self.embedding(x.argmax(dim=1))
        features = self.mlp(x_embedded)
        dist = torch.cdist(features.unsqueeze(0), self.weights.unsqueeze(0), p=2).squeeze(0)
        inv_dist = 1 / (dist**self.n + 1e-8)
        probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
        return probs


# ============================
# --- Deep Harmonic Layer with L1 Distance ---
# ============================
class HarmonicLayerL1(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16, hidden_dim=32, n=1):
        super(HarmonicLayerL1, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.weights = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.n = n

    def forward(self, x):
        x_embedded = self.embedding(x.argmax(dim=1))
        features = self.mlp(x_embedded)
        dist = torch.cdist(features.unsqueeze(0), self.weights.unsqueeze(0), p=1).squeeze(0)
        inv_dist = 1 / (dist**self.n + 1e-8)
        probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
        return probs


# ============================
# --- Deep Softmax Policy ---
# ============================
class SoftmaxPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16, hidden_dim=32):
        super(SoftmaxPolicy, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x_embedded = self.embedding(x.argmax(dim=1))
        features = self.mlp(x_embedded)
        logits = self.fc(features)
        return torch.softmax(logits, dim=1)


# ============================
# --- REINFORCE Training Function ---
# ============================
def train(policy, env, optimizer, episodes=10000, baseline_name="", log_interval=100, test_env=None, test_eval_episodes=10):
    reinforce_losses = []
    success_count = 0

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
            next_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            if done:
                if info.get("info") == "Goal Reached":
                    success_count += 1
                break
            state = next_state

        returns = sum(rewards)
        reinforce_loss = -sum([lp * returns for lp in log_probs])
        optimizer.zero_grad()
        reinforce_loss.backward()
        optimizer.step()
        reinforce_losses.append(reinforce_loss.item())

        if episode % log_interval == 0 and episode > 0:
            train_accuracy = (success_count / log_interval) * 100
            if test_env is not None:
                _, test_success_rate = evaluate_policy(policy, test_env, episodes=test_eval_episodes)
                test_accuracy = test_success_rate * 100
            else:
                test_accuracy = None
            if test_accuracy is not None:
                log(f"{baseline_name} Episode {episode}: Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")
            else:
                log(f"{baseline_name} Episode {episode}: Train Acc: {train_accuracy:.2f}%")
            success_count = 0

    return reinforce_losses


# ============================
# --- Train Ensemble Function ---
# ============================
def train_ensemble(policy, env, optimizer_class, optimizer_args, episodes=10000, baseline_name="", n_ensemble=1, log_interval=100, test_env=None, test_eval_episodes=10):
    ensemble_results = []
    acc_data = []  # To record accuracy data for each ensemble member
    
    for i in range(n_ensemble):
        log(f"Training ensemble member {i+1}/{n_ensemble} for {baseline_name}")
        model = copy.deepcopy(policy)
        optimizer = optimizer_class(model.parameters(), **optimizer_args)
        losses = []
        returns_list = []
        success_count = 0
        member_acc = {'episodes': [], 'train_acc': [], 'test_acc': []}
        
        # Initialize baseline variables for variance reduction:
        baseline = 0.0
        baseline_decay = 0.99
        
        for episode in range(episodes):
            state = env.reset()
            log_probs = []
            rewards = []
            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs = model(state_tensor)
                if action_probs.dim() == 1:
                    action_probs = action_probs.unsqueeze(0)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                next_state, reward, done, info = env.step(action.item())
                rewards.append(reward)
                if done:
                    if info.get("info") == "Goal Reached":
                        success_count += 1
                    break
                state = next_state

            returns = sum(rewards)
            # Update the baseline using an exponential moving average
            baseline = baseline_decay * baseline + (1 - baseline_decay) * returns
            adjusted_return = returns - baseline  # variance reduction

            returns_list.append(returns)
            reinforce_loss = -sum([lp * adjusted_return for lp in log_probs])
            optimizer.zero_grad()
            reinforce_loss.backward()
            optimizer.step()
            losses.append(reinforce_loss.item())

            if episode % log_interval == 0 and episode > 0:
                train_accuracy = (success_count / log_interval) * 100
                if test_env is not None:
                    _, test_success_rate = evaluate_policy(model, test_env, episodes=test_eval_episodes)
                    test_accuracy = test_success_rate * 100
                else:
                    test_accuracy = None
                member_acc['episodes'].append(episode)
                member_acc['train_acc'].append(train_accuracy)
                member_acc['test_acc'].append(test_accuracy if test_accuracy is not None else 0)
                log(f"{baseline_name} {i+1} Episode {episode}: Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")
                success_count = 0

        ensemble_results.append({
            'model': model,
            'losses': losses,
            'returns': returns_list,
            'acc_data': member_acc
        })
        acc_data.append(member_acc)
    
    # Compute mean, min, and max accuracy across ensemble members at each log interval:
    common_episodes = ensemble_results[0]['acc_data']['episodes']
    train_acc_array = np.array([member['acc_data']['train_acc'] for member in ensemble_results])
    test_acc_array  = np.array([member['acc_data']['test_acc'] for member in ensemble_results])
    
    acc_summary = {
        'episodes': common_episodes,
        'train_acc_mean': np.mean(train_acc_array, axis=0),
        'train_acc_min': np.min(train_acc_array, axis=0),
        'train_acc_max': np.max(train_acc_array, axis=0),
        'test_acc_mean': np.mean(test_acc_array, axis=0),
        'test_acc_min': np.min(test_acc_array, axis=0),
        'test_acc_max': np.max(test_acc_array, axis=0)
    }
    
    best_model = None
    best_avg_return = -float('inf')
    best_losses = None
    last_portion = episodes // 10  # Last 10% of episodes
    
    for result in ensemble_results:
        avg_return = np.mean(result['returns'][-last_portion:])
        log(f"{baseline_name} Ensemble member average return (last 10%): {avg_return:.2f}")
        if avg_return > best_avg_return:
            best_avg_return = avg_return
            best_model = result['model']
            best_losses = result['losses']

    log(f"Best model for {baseline_name} achieved an average return of {best_avg_return:.2f} over the last {last_portion} episodes.")
    return best_model, best_losses, acc_summary


# ============================
# --- Policy Evaluation Function ---
# ============================
def evaluate_policy(policy, env, episodes=100):
    total_return = 0
    successes = 0
    for _ in range(episodes):
        state = env.reset()
        episode_return = 0
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = policy(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            state = next_state
            if done:
                if info.get("info") == "Goal Reached":
                    successes += 1
                break
        total_return += episode_return
    avg_return = total_return / episodes
    success_rate = successes / episodes
    return avg_return, success_rate


# ============================
# --- Plot Accuracy Curves Function ---
# ============================
def plot_accuracy_curves(accuracy_data):
    """
    Expects accuracy_data to be a dict with keys for each policy.
    Each value is a dict with:
      'episodes', 'train_acc_mean', 'train_acc_min', 'train_acc_max',
      'test_acc_mean', 'test_acc_min', 'test_acc_max'
    Plots a 3x1 plot (one panel per policy).
    """
    policies = list(accuracy_data.keys())
    num_policies = len(policies)
    fig, axs = plt.subplots(num_policies, 1, figsize=(8, 4*num_policies), sharex=True)
    if num_policies == 1:
        axs = [axs]
    for ax, policy in zip(axs, policies):
        data = accuracy_data[policy]
        episodes = data['episodes']
        ax.plot(episodes, data['train_acc_mean'], label='Train Accuracy', color='blue')
        ax.fill_between(episodes, data['train_acc_min'], data['train_acc_max'], color='blue', alpha=0.3)
        ax.plot(episodes, data['test_acc_mean'], label='Test Accuracy', color='red')
        ax.fill_between(episodes, data['test_acc_min'], data['test_acc_max'], color='red', alpha=0.3)
        ax.set_title(policy)
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
    axs[-1].set_xlabel("Episodes")
    plt.tight_layout()
    plt.show()


# ============================
# --- Plot Learning Curves Function ---
# ============================
def smooth_curve(data, window_length):
    if window_length < 2:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_length:] - cumsum[:-window_length]) / window_length

def plot_learning_curves(results, episodes):
    plt.figure(figsize=(10, 6))
    window_length = int(0.1 * episodes)
    for label, losses in results.items():
        losses = smooth_curve(losses, window_length)
        plt.plot(losses, label=f"{label} (Smoothed)")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================
# --- Save Models ---
# ============================
def save_model(model, filename):
    os.makedirs('models', exist_ok=True)
    save_path = os.path.join('models', filename)
    torch.save(model.state_dict(), save_path)
    log(f"✅ Model saved to {save_path}")


# ============================
# --- Main Script ---
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble training of Harmonic and Softmax policies on ToyEnv2 with train/test split.")
    parser.add_argument("--episodes", type=int, default=10**5, help="Number of training episodes")
    parser.add_argument("--n_ensemble", type=int, default=10, help="Number of ensemble members per policy")
    args = parser.parse_args()
    num_eps = args.episodes
    n_ensemble = args.n_ensemble

    lr = 0.01
    embedding_dim = 16
    eval_episodes = 100

    # Define separate distributions with overlap and additional distinct elements:
    train_possible_starts = [(0, 0), (0, 1), (1, 0), (1, 1),
                             (4, 0), (4, 1), (3, 2)]
    train_possible_goals  = [(3, 3), (3, 4), (4, 3), (4, 4),
                             (4, 2), (0, 4), (2, 4)]

    test_possible_starts  = [(0, 0), (0, 1), (1, 1), (1, 2),
                             (3, 4), (4, 4), (2, 3)]
    test_possible_goals   = [(3, 3), (4, 3), (4, 4), (4, 2),
                             (0, 0), (0, 1), (1, 0)]

    # Create training environment on a 5x5 grid.
    train_env = ToyEnv2(
        grid_size=(5, 5),
        start=(0, 0),
        goal=(4, 4),
        obstacles={(2, 2)},
        max_steps=20,
        randomize=True,
        possible_starts=train_possible_starts,
        possible_goals=train_possible_goals,
        mode="train"
    )

    # Create testing environment on a 5x5 grid.
    test_env = ToyEnv2(
        grid_size=(5, 5),
        start=(0, 0),
        goal=(4, 4),
        obstacles={(2, 2)},
        max_steps=20,
        randomize=True,
        possible_starts=test_possible_starts,
        possible_goals=test_possible_goals,
        mode="test"
    )

    input_dim = train_env.obs_dim
    output_dim = train_env.action_dim
    optimizer_args = {"lr": lr}

    policies_to_train = [
        ("Harmonic (L2)", HarmonicLayer, "best_harmonic_policy_l2.pth"),
        ("Softmax", SoftmaxPolicy, "best_softmax_policy.pth")
        # ("Harmonic (L1)", HarmonicLayerL1, "best_harmonic_policy_l1.pth")
    ]

    best_models = {}
    accuracy_results = {}  # To store accuracy curves per policy
    ensemble_losses = {}   # To store loss curves per policy

    for label, PolicyClass, filename in policies_to_train:
        log(f"=== Ensemble Training for {label} Policy ===")
        # For deep models, we pass a hidden_dim as well:
        if label.startswith("Softmax"):
            policy_instance = PolicyClass(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim, hidden_dim=32)
        else:
            policy_instance = PolicyClass(input_dim=input_dim, output_dim=output_dim, embedding_dim=embedding_dim, hidden_dim=32)
        best_model, losses, acc_data = train_ensemble(policy_instance, train_env, optim.Adam, optimizer_args,
                                                      episodes=num_eps, baseline_name=label, n_ensemble=n_ensemble,
                                                      log_interval=100, test_env=test_env, test_eval_episodes=20)
        save_model(best_model, filename)
        best_models[label] = best_model
        ensemble_losses[label] = losses
        accuracy_results[label] = acc_data

    log("Evaluating Best Ensemble Models")
    for label, model in best_models.items():
        avg_return, success_rate = evaluate_policy(model, test_env, episodes=eval_episodes)
        log(f"Evaluation {label}: Test Accuracy = {success_rate*100:.1f}% (Avg Return: {avg_return:.2f})")

    # Plot the accuracy curves in a 3x1 plot.
    plot_accuracy_curves(accuracy_results)

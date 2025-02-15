import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool, get_context

from envs.env_gridworld import gridWorld
from agents.pg import PG
from colorama import Fore, Style


def print_parameters(params):
    print(Fore.CYAN + "----------------")
    print(Fore.CYAN + "Parameters")
    print(Fore.CYAN + "----------------" + Style.RESET_ALL)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("\n")


class RewardCallback:
    def __init__(self, env_train, env_eval, model, log_interval=100, eval_interval=500, save_dir=".", verbose=1, early_stopping=True):
        self.env_train = env_train
        self.env_eval = env_eval
        self.model = model
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.episode_rewards = []
        self.train_acc = []
        self.test_acc = []
        self.current_episode = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _on_step(self, reward):
        self.episode_rewards.append(reward)
        self.current_episode += 1
        if self.current_episode % self.log_interval == 0:
            train_acc = self._compute_accuracy(self.env_train)
            self.train_acc.append(train_acc)
            test_acc = self._compute_accuracy(self.env_eval)
            self.test_acc.append(test_acc)
            print(Fore.WHITE + f"[{datetime.now().strftime('%H:%M:%S')}] Episode {self.current_episode}: acc_train, acc_test: {train_acc:.2f}%, {test_acc:.2f}%" + Style.RESET_ALL)

            if self.early_stopping and test_acc == 100.0:
                print(Fore.GREEN + f"Early stopping triggered at episode {self.current_episode} with test accuracy of 100%!" + Style.RESET_ALL)
                raise StopIteration  # Stop the training loop

    def _compute_accuracy(self, env):
        successes = 0
        total = 50
        for _ in range(total):
            state = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = self.model.predict(state)
                state, reward, done, info = env.step(action)
                if info.get('info') == "Goal Reached":
                    successes += 1
        return (successes / total) * 100

    def get_results(self):
        return self.episode_rewards, self.train_acc, self.test_acc


def pad_to_length(arr, length):
    """Pad array with last value to the target length."""
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'edge')
    return arr


def plot_ensemble_curves(ensemble_train, ensemble_test, save_dir, log_interval):
    max_len = max(max(len(t), len(v)) for t, v in zip(ensemble_train, ensemble_test))
    ensemble_train = np.array([pad_to_length(arr, max_len) for arr in ensemble_train])
    ensemble_test = np.array([pad_to_length(arr, max_len) for arr in ensemble_test])

    mean_train = np.mean(ensemble_train, axis=0)
    min_train = np.min(ensemble_train, axis=0)
    max_train = np.max(ensemble_train, axis=0)
    mean_test = np.mean(ensemble_test, axis=0)
    min_test = np.min(ensemble_test, axis=0)
    max_test = np.max(ensemble_test, axis=0)

    steps = np.arange(len(mean_train)) * log_interval  # X-axis as Nstep

    plt.figure(figsize=(10, 6))
    plt.fill_between(steps, min_train, max_train, color='blue', alpha=0.2, label='Train Acc Range')
    plt.fill_between(steps, min_test, max_test, color='red', alpha=0.2, label='Test Acc Range')
    plt.plot(steps, mean_train, label='Mean Train Accuracy', color='blue')
    plt.plot(steps, mean_test, label='Mean Test Accuracy', color='red')
    plt.xlabel('Nstep')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy with PG Agent')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'ensemble_accuracy_curves.png'))
    plt.show()


def run_training(args, seed=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    env_train, env_test = gridWorld(), gridWorld()
    model = PG(env_train, layer_type=args.layer_type, distance=args.distance_norm, n_layers=args.n_layers)
    callback = RewardCallback(env_train, env_test, model, log_interval=args.log_interval, eval_interval=args.log_interval)
    try:
        model.learn(total_timesteps=args.Ntrain, callback=callback)
    except StopIteration:
        pass
    model.save(os.path.join(args.save_dir, f"{args.agent}_{args.layer_type}_model_{seed}.pth"))
    _, train_acc, test_acc = callback.get_results()
    return train_acc, test_acc


def main(args):
    print("\nStarting Training with the following configuration:")
    print_parameters(vars(args))

    if args.parallel:
        ctx = get_context('spawn')
        with ctx.Pool(args.num_workers) as pool:
            results = pool.starmap(run_training, [(args, i) for i in range(args.ensemble_size)])
    else:
        results = [run_training(args, i) for i in range(args.ensemble_size)]

    ensemble_train = [res[0] for res in results]
    ensemble_test = [res[1] for res in results]
    plot_ensemble_curves(ensemble_train, ensemble_test, args.save_dir, args.log_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PG agent on gridWorld with ensemble and multiprocessing.")
    parser.add_argument('--Ntrain', type=int, default=10**4, help='Number of training timesteps per agent.')
    parser.add_argument('--agent', type=str, default='PG', help='Agent type.')
    parser.add_argument('--layer_type', type=str, choices=['softmax', 'harmonic'], default='softmax', help='Layer type.')
    parser.add_argument('--distance_norm', type=str, choices=['L1', 'L2'], default='L2', help='Distance norm for harmonic layer.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the PG agent.')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of ensemble agents.')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results.')
    parser.add_argument('--parallel', action='store_true', help='Flag to use parallel training.')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of parallel workers.')
    parser.add_argument('--log_interval', type=int, default=None, help='Logging interval.')

    args = parser.parse_args()
    if args.log_interval is None:
        args.log_interval = int(0.1 * args.Ntrain)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)

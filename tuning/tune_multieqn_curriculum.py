#!/usr/bin/env python
import sys
import os
import json
import argparse
import pickle
import time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
import itertools
import pandas as pd
import multiprocessing as mp
from itertools import product

# Append parent directory to path for local module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import environment and utility modules.
from envs.multi_eqn_curriculum import multiEqn
from utils.utils_train import get_intrinsic_reward, get_device, CustomGNNPolicy, get_agent
from utils.custom_functions import operation_names
from utils.utils_general import print_parameters, print_header

# Import SB3-related modules.
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

device = get_device()

# === Helper Functions ===

def print_eval_results(test_results, label=""):
    """Print evaluation results in a table format."""
    print(f"{label} Equations")
    df = pd.DataFrame([{'Eqn': eqn, 'Win%': f"{winpct:.1f}%"} for eqn, winpct in test_results.items()])
    print(df.to_string(index=False))
    print()

def print_results_dict_as_df(title, d):
    """Print results dictionary as a DataFrame."""
    print(f"\n{title}")
    rows = [{'Eqn': eqn, 'TSolve': tsolve} for eqn, tsolve in d.items()]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

def get_action_mask(env):
    return env.action_mask

def evaluate_agent(agent, env, equation_list, n_eval_episodes=10):
    """
    Evaluate the agent on each equation in equation_list.
    Returns a dict mapping each equation to its success percentage.
    """
    results = {}
    for eqn in equation_list:
        eqn_successes = 0
        for ep in range(n_eval_episodes):
            obs = env.reset()
            env.env_method('set_equation', eqn)
            done = [False]
            while not done[0]:
                action, _ = agent.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                if info[0].get('is_solved', False):
                    eqn_successes += 1
                    break
        results[eqn] = (eqn_successes / n_eval_episodes) * 100.0
    return results

# === Callbacks ===

class TrainingLogger(BaseCallback):
    """Callback for logging reward statistics, evaluations, and checkpointing."""
    def __init__(self, log_interval=1000, eval_interval=10000, save_dir=".", verbose=1, eval_env=None):
        super(TrainingLogger, self).__init__(verbose)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.eval_env = eval_env
        self.rewards_ext = []
        self.T_solve = None
        self.T_converge = None
        self.early_stopping = False

        self.results_train = {}
        self.results_test = {}
        self.logged_steps = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_accuracy_one_shot = []
        self.test_accuracy_one_shot = []
        self.max_test_acc_one_shot = 0.0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _on_training_start(self):
        eqns_attr = self.training_env.get_attr('train_eqns')
        if eqns_attr:
            train_eqns = eqns_attr[0]
            self.results_train = {eqn: None for eqn in train_eqns}

        test_eqns_attr = self.training_env.get_attr('test_eqns')
        if test_eqns_attr:
            test_eqns = test_eqns_attr[0]
            self.results_test = {eqn: None for eqn in test_eqns}

    def _on_step(self) -> bool:
        reward_ext = self.locals["rewards"][0]
        self.rewards_ext.append(reward_ext)
        info = self.locals["infos"][0]

        if info.get('is_solved', False):
            main_eqn = info['main_eqn']
            if self.results_train.get(main_eqn) is None:
                self.results_train[main_eqn] = self.num_timesteps
                #print(Fore.YELLOW + f'\nSolved {main_eqn} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)

        #if self.n_calls % self.log_interval == 0:
            #mean_reward_ext = np.mean(self.rewards_ext[-self.log_interval:])
            #print(f"Step {self.num_timesteps}: mean_reward = {mean_reward_ext:.2f}")
        return True

# === Main Training Function ===

def main(args):
    """Main function for training the RL agent."""
    t1 = time.time()
    #print('\n')
    params = vars(args)
    # print_parameters(params)

    def make_env():
        env = multiEqn(normalize_rewards=args.normalize_rewards, 
                       state_rep=args.state_rep, 
                       level=args.level, 
                       generalization=args.generalization)
        if args.agent_type in ["ppo-mask", "ppo-cnn", "ppo-gnn", "ppo-gnn1"]:
            env = ActionMasker(env, get_action_mask)
        return env

    env_fns = [lambda: make_env() for _ in range(args.n_envs)]
    env = DummyVecEnv(env_fns)

    sb3_kwargs = {
        "ent_coef": args.ent_coef,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "vf_coef": args.vf_coef,
        "clip_range": args.clip_range,
        "max_grad_norm": args.max_grad_norm,
    }
    net_arch = [args.hidden_dim] * args.n_layers
    policy_kwargs = {"net_arch": net_arch}
    sb3_kwargs["policy_kwargs"] = policy_kwargs

    agent = get_agent(args.agent_type, env, **sb3_kwargs)

    callback = TrainingLogger(log_interval=args.log_interval, save_dir=args.save_dir, eval_env=env)

    #print_header(f"Starting training for {args.Ntrain} timesteps", color='cyan')
    agent.learn(total_timesteps=args.Ntrain, callback=callback)
    agent.save(os.path.join(args.save_dir, f"{args.agent_type}_trained_model"))
    #print(f"Model saved to {args.save_dir}")

    train_results = evaluate_agent(agent, env, env.get_attr('train_eqns')[0], n_eval_episodes=10)
    test_results = evaluate_agent(agent, env, env.get_attr('test_eqns')[0], n_eval_episodes=10)
    #print_eval_results(train_results, label='Train')

    t2 = time.time()
    hours, rem = divmod(t2 - t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    return train_results, test_results, callback.max_test_acc_one_shot

# === Parallel Hyperparameter Tuning ===

# === Parallel Hyperparameter Tuning ===

# Global variable to store best trial so far
best_trial_so_far = None

def run_trial(args, lr, ent, n_layers, hidden_dim, trial_idx):
    """Run a single hyperparameter tuning trial."""
    global best_trial_so_far

    args.learning_rate = lr
    args.ent_coef = ent
    args.n_layers = n_layers
    args.hidden_dim = hidden_dim

    print(Fore.CYAN + f"\n[Trial {trial_idx}] Running with: "
          f"lr={lr}, ent_coef={ent}, n_layers={n_layers}, hidden_dim={hidden_dim}" + Style.RESET_ALL)

    results_train, results_test, max_test_acc_one_shot = main(args)

    train_acc = sum(1 for tsolve in results_train.values() if tsolve is not None) / len(results_train)
    tsolve_vals = [tsolve for tsolve in results_train.values() if tsolve is not None]
    avg_tsolve = np.mean(tsolve_vals) if tsolve_vals else None

    result = {
        'trial': trial_idx,
        'learning_rate': lr,
        'ent_coef': ent,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'train_acc': train_acc,
        'avg_tsolve': avg_tsolve,
        'max_test_acc': max_test_acc_one_shot
    }

    trial_save_path = os.path.join(args.save_dir, f"trial_{trial_idx}.json")
    with open(trial_save_path, 'w') as f:
        json.dump(result, f, indent=4)

    # Check if this is the best trial so far
    if best_trial_so_far is None or result['train_acc'] > best_trial_so_far['train_acc']:
        best_trial_so_far = result
        print(Fore.GREEN + f"\n🌟 New Best Trial Found! (Trial {trial_idx})" + Style.RESET_ALL)

    # Print the best trial so far in clear magenta
    print(Fore.MAGENTA + "\n💎 Best Trial So Far:" + Style.RESET_ALL)
    for key, value in best_trial_so_far.items():
        print(f"  {key}: {value}")
    print("\n" + "-"*60)

    return result

def parallel_tuning(args, learning_rates, ent_coeffs, net_arch_options):
    """Run hyperparameter tuning in parallel."""
    start_time = time.time()

    hyperparameter_grid = list(product(learning_rates, ent_coeffs, net_arch_options))

    with mp.Pool(processes=args.num_workers) as pool:
        results = pool.starmap(
            run_trial,
            [(args, lr, ent, n_layers, hidden_dim, idx + 1)
             for idx, (lr, ent, (n_layers, hidden_dim)) in enumerate(hyperparameter_grid)]
        )

    df_results = pd.DataFrame(results).sort_values(by='train_acc', ascending=False)
    csv_save_path = os.path.join(args.save_dir, "tuning_results.csv")
    df_results.to_csv(csv_save_path, index=False)
    print(Fore.GREEN + f"\n✅ Tuning completed! Results saved to {csv_save_path}" + Style.RESET_ALL)

    end_time = time.time()
    print(f"\n⏱ Total tuning time: {end_time - start_time:.2f} seconds")

    best_trial = df_results.iloc[0]
    print(Fore.CYAN + "\n🎯 Final Best Trial Overall:" + Style.RESET_ALL)
    print(best_trial)


# === Main Script Entry Point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-gnn', help='Agent type')
    parser.add_argument('--state_rep', type=str, default='graph_integer_2d', help='State representation/encoding')
    parser.add_argument('--Ntrain', type=int, default=10**6, help='Number of training steps')
    parser.add_argument('--intrinsic_reward', type=str, default='None', choices=['ICM', 'E3B', 'RIDE', 'None'],
                        help='Type of intrinsic reward')
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"),
                        default=True, help="Normalize rewards (True/False)")
    parser.add_argument('--log_interval', type=int, default=1000, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='data/tuning/curriculum', help='Directory to save results')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of environments')

    # Agent parameters
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy coefficient for PPO')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per rollout')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function loss coefficient')
    parser.add_argument('--clip_range', type=float, default=0.2, help='Clip range for PPO')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')

    # Network parameters
    parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')

    # Generalization parameters
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--generalization', type=str, default='random')

    # Parallelization
    parser.add_argument('--parallel', type=lambda v: v.lower() in ("yes", "true", "t", "1"),
                        default=False, help="Run hyperparameter tuning in parallel")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for parallel tuning')

    args = parser.parse_args()
    args.log_interval = args.Ntrain
    args.save_dir = os.path.join(args.save_dir, args.agent_type)

    # Define hyperparameter grids
    learning_rates = [3e-4]
    ent_coeffs = [0, 0.01, 0.1]
    net_arch_options = [(2, 64), (3, 256)]

    if args.parallel:
        print("\n🚀 Running hyperparameter tuning in parallel")
        parallel_tuning(args, learning_rates, ent_coeffs, net_arch_options)
    else:
        print("\n🔬 Running hyperparameter tuning sequentially")
        for idx, (lr, ent, (n_layers, hidden_dim)) in enumerate(
                product(learning_rates, ent_coeffs, net_arch_options), start=1):
            run_trial(args, lr, ent, n_layers, hidden_dim, idx)

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
        # Initialize training and test equation dictionaries if available.
        eqns_attr = self.training_env.get_attr('train_eqns')
        if eqns_attr:
            train_eqns = eqns_attr[0]
            self.results_train = {eqn: None for eqn in train_eqns}
        else:
            print("Warning: no 'train_eqns' found in env; results_train will remain empty.")

        test_eqns_attr = self.training_env.get_attr('test_eqns')
        if test_eqns_attr:
            test_eqns = test_eqns_attr[0]
            self.results_test = {eqn: None for eqn in test_eqns}
        else:
            print("No 'test_eqns' found in env. results_test will remain empty.")

        if self.eval_env:
            print("\nInitial evaluation (t=0)...")
            print('Solve counts')
            all_solve_counts = self.eval_env.get_attr("solve_counts")[0]
            for eqn, solve_count in all_solve_counts.items():
                print(f"{eqn}: {solve_count}")
            print('\n')
            self.logged_steps.append(0)

    def _on_step(self) -> bool:
        reward_ext = self.locals["rewards"][0]
        self.rewards_ext.append(reward_ext)
        info = self.locals["infos"][0]

        if info.get('is_solved', False):
            main_eqn, lhs, rhs = info['main_eqn'], info['lhs'], info['rhs']
            self.T_solve = self.num_timesteps
            if self.results_train.get(main_eqn) is None:
                self.results_train[main_eqn] = self.num_timesteps
                print(Fore.YELLOW + f'\nSolved {main_eqn} = 0 ==> {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)
            if self.early_stopping:
                print(Fore.YELLOW + f"{main_eqn}: Early stopping triggered at step {self.num_timesteps}!" + Style.RESET_ALL)
                return False  

        if self.n_calls % self.log_interval == 0:
            mean_reward_ext = np.mean(self.rewards_ext[-self.log_interval:])
            min_reward_ext = np.min(self.rewards_ext[-self.log_interval:])
            max_reward_ext = np.max(self.rewards_ext[-self.log_interval:])
            main_eqn = info.get('main_eqn', 'N/A')
            print(f"{main_eqn}: Step {self.num_timesteps}: "
                  f"(min, mean, max)_reward_external = ({min_reward_ext:.2f}, {mean_reward_ext:.2f}, {max_reward_ext:.2f})")
        return True

    def _on_training_end(self) -> None:
        print("\nFinal Training Completed. Plotting Learning Curves...")
        # Here you could plot and/or save learning curves.
        # For example, saving the logged steps and accuracies.
        # plt.savefig(os.path.join(self.save_dir, "learning_curve.png"))
        # with open(os.path.join(self.save_dir, "learning_progress.pkl"), "wb") as f:
        #     pickle.dump({
        #         "steps": self.logged_steps,
        #         "train_success": self.train_accuracy,
        #         "train_success_one_shot": self.train_accuracy_one_shot,
        #     }, f)
        # print(f"Saved learning progress to {self.save_dir}")

class IntrinsicReward(BaseCallback):
    """Callback for logging intrinsic rewards."""
    def __init__(self, irs, verbose=0, log_interval=100):
        super(IntrinsicReward, self).__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.rewards_internal = []
        self.log_interval = log_interval
        self.last_computed_intrinsic_rewards = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        if self.last_computed_intrinsic_rewards is not None:
            intrinsic_reward = self.last_computed_intrinsic_rewards[-1]
            self.rewards_internal.append(intrinsic_reward)
        if self.n_calls % self.log_interval == 0 and self.rewards_internal:
            mean_intrinsic = np.mean(self.rewards_internal[-self.log_interval:])
            min_intrinsic = np.min(self.rewards_internal[-self.log_interval:])
            max_intrinsic = np.max(self.rewards_internal[-self.log_interval:])
            main_eqn = self.locals["infos"][0].get('main_eqn', 'N/A')
            print(f"{main_eqn}: Step {self.num_timesteps}: "
                  f"(min, mean, max)_reward_internal = ({min_intrinsic:.3f}, {mean_intrinsic:.3f}, {max_intrinsic:.3f})\n")
        return True

    def _on_rollout_end(self) -> None:
        obs = th.as_tensor(self.buffer.observations).float()
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"]).float()
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions,
                         rewards=rewards, terminateds=dones,
                         truncateds=dones, next_observations=new_obs),
            sync=True
        ).cpu().numpy()
        self.last_computed_intrinsic_rewards = intrinsic_rewards
        self.buffer.advantages += intrinsic_rewards
        self.buffer.returns += intrinsic_rewards

# === Main Training Function ===

def main(args):
    t1 = time.time()
    print('\n')
    params = vars(args)
    print_parameters(params)

    # Create the environment factories.
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

    # Setup agent parameters and policy network architecture.
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

    # Setup callbacks.
    callback = TrainingLogger(log_interval=args.log_interval, save_dir=args.save_dir, eval_env=env)
    irs = get_intrinsic_reward(args.intrinsic_reward, env)
    if irs:
        callback_intrinsic_reward = IntrinsicReward(irs, log_interval=args.log_interval)
        callback = [callback, callback_intrinsic_reward]

    print_header(f"Starting training for {args.Ntrain} timesteps", color='cyan')
    agent.learn(total_timesteps=args.Ntrain, callback=callback)
    agent.save(os.path.join(args.save_dir, f"{args.agent_type}_trained_model"))
    print(f"Model saved to {args.save_dir}")

    print_header('Evaluating trained agent', color='cyan')
    train_results = evaluate_agent(agent, env, env.get_attr('train_eqns')[0], n_eval_episodes=10)
    test_results = evaluate_agent(agent, env, env.get_attr('test_eqns')[0], n_eval_episodes=10)
    print_eval_results(train_results, label='Train')
    
    if isinstance(callback, list):
        callback = callback[0]
    results_train, results_test = callback.results_train, callback.results_test

    train_save_path = os.path.join(args.save_dir, "Tsolve.json")
    train_dict_str_keys = {str(k): v for k, v in results_train.items()}
    with open(train_save_path, "w") as f:
        json.dump(train_dict_str_keys, f, indent=4)
    print(f"Saved train results to {train_save_path}")

    solve_counts = env.get_attr("solve_counts")[0]
    with open(os.path.join(args.save_dir, "solve_counts.json"), "w") as f:
        json.dump(solve_counts, f, indent=4)
    print("Saved solve counts.")

    sample_counts = env.get_attr("sample_counts")[0]
    with open(os.path.join(args.save_dir, "sample_counts.json"), "w") as f:
        json.dump(sample_counts, f, indent=4)
    print("Saved sample counts.")

    print_header("Final results", color='cyan')
    print_results_dict_as_df("Training Results", results_train)

    t2 = time.time()
    hours, rem = divmod(t2 - t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    return results_train, results_test, callback.max_test_acc_one_shot

# === Main Script Entry Point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-mask', help='Agent type')
    parser.add_argument('--state_rep', type=str, default='integer_1d', help='State representation/encoding')
    parser.add_argument('--Ntrain', type=int, default=10**6, help='Number of training steps')
    parser.add_argument('--intrinsic_reward', type=str, default='ICM', choices=['ICM', 'E3B', 'RIDE', 'None'],
                        help='Type of intrinsic reward')
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"),
                        default=True, help="Normalize rewards (True/False)")
    parser.add_argument('--log_interval', type=int, default=None, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='data/tuning/curriculum', help='Directory to save the results')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--n_envs', type=int, default=1, help='Number of envs to run in parallel')

    # Agent parameters
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy coefficient for PPO or MaskablePPO')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the RL agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per rollout in PPO')
    parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size for gradient update')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per batch')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function loss coefficient')
    parser.add_argument('--clip_range', type=float, default=0.2, help='Clip range for PPO')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Gradient clipping norm')

    # New network parameters
    parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden layers in the policy network')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden units per layer')

    # Generalization parameters
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--generalization', type=str, default='random', choices=['lexical', 'structural', 'shallow', 'deep', 'random'])

    args = parser.parse_args()

    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = int(args.Ntrain)
    
    # Set save_dir based on hyperparameters.
    args.save_dir = os.path.join(args.save_dir, f"{args.generalization}/level{args.level}/{args.agent_type}/n_layer{args.n_layers}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Check for invalid (agent, state_rep) pairs.
    if args.state_rep in ['graph_integer_1d', 'graph_integer_2d'] and args.agent_type not in ['ppo-gnn', 'ppo-gnn1']:
        raise ValueError(f"❌ ERROR: '{args.agent_type}' requires 'graph_integer_1d' or 'graph_integer_2d' state_rep, but got '{args.state_rep}'.")

    # Save arguments.
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Define your hyperparameter grids:
    learning_rates = [1e-3, 3e-4, 1e-4]        # Learning rates to try
    ent_coeffs = [0, 0.01, 0.1, 0.2]          # Entropy coefficients
    net_arch_options = [(2, 64), (3,256), (4,1024)]  # Each tuple is (n_layers, hidden_dim)


    # Generate the grid using itertools.product:
    hyperparameter_grid = list(itertools.product(learning_rates, ent_coeffs, net_arch_options))

    tuning_data = []  # To store results for each configuration
    best_train_acc = -np.inf
    best_trial = None
    trial_idx = 0

    for lr, ent, (n_layers, hidden_dim) in hyperparameter_grid:
        trial_idx += 1
        # Update the arguments (assuming args is already defined)
        args.learning_rate = lr
        args.ent_coef = ent
        args.n_layers = n_layers
        args.hidden_dim = hidden_dim

        print(f"\n[Trial {trial_idx}] Running experiment with: "
            f"learning_rate = {lr}, ent_coef = {ent}, n_layers = {n_layers}, hidden_dim = {hidden_dim}")

        # Run the training experiment. (main() returns results_train, results_test, max_test_acc_one_shot)
        results_train, results_test, max_test_acc_one_shot = main(args)

        # Compute training accuracy: count fraction of equations with a non-None Tsolve.
        train_acc = sum(1 for eqn, tsolve in results_train.items() if tsolve is not None) / len(results_train)
        
        # Compute average Tsolve (ignoring unsolved equations)
        tsolve_vals = [tsolve for tsolve in results_train.values() if tsolve is not None]
        avg_tsolve = np.mean(tsolve_vals) if tsolve_vals else None

        # Record this trial’s results in a dictionary.
        current_result = {
            'trial': trial_idx,
            'learning_rate': lr,
            'ent_coef': ent,
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'train_acc': train_acc,
            'avg_tsolve': avg_tsolve,
            'max_test_acc': max_test_acc_one_shot
        }
        tuning_data.append(current_result)
        
        # Print checkpoint and track best trial.
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_trial = current_result
            print(Fore.GREEN + f"New best trial: Trial {trial_idx} with train_acc = {train_acc*100:.2f}%" + Style.RESET_ALL)
        else:
            print(f"Trial {trial_idx} completed with train_acc = {train_acc*100:.2f}%")

        print(Fore.MAGENTA + "\nBest trial so far:" + Style.RESET_ALL)
        print(Fore.MAGENTA + str(best_trial) + Style.RESET_ALL)

    # Convert results to a Pandas DataFrame and sort by training accuracy.
    df_results = pd.DataFrame(tuning_data)
    df_sorted = df_results.sort_values(by='train_acc', ascending=False)

    print("\nTuning results (sorted by train_acc):")
    print(df_sorted)

    # Save the tuning results CSV to a subdirectory under args.save_dir.
    csv_save_path = os.path.join(args.save_dir, "tuning_results.csv")
    df_sorted.to_csv(csv_save_path, index=False)
    print(f"Tuning results saved to {csv_save_path}")



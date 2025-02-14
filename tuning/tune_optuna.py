#!/usr/bin/env python
import sys
import os
import json
import argparse
import time
import numpy as np
import torch as th
import pandas as pd
import optuna
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.visualization import plot_optimization_history, plot_param_importances
from colorama import Fore, Style
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker

# Append parent directory to path for local module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.multi_eqn_curriculum import multiEqn
from utils.utils_train import get_agent

best_trial_so_far = None

def pretty_print_top_trials(trials, top_n=5):
    """Display the top trials with ranking."""
    print(Fore.MAGENTA + f"\n🏆 Top {top_n} Trials by Train Accuracy:" + Style.RESET_ALL)
    for rank, t in enumerate(trials[:top_n], start=1):
        print(f"{Fore.BLUE}  🥇 Rank {rank}:{Style.RESET_ALL} Trial #{t.number}")
        print(f"    - Accuracy: {Fore.GREEN}{t.value:.4f}{Style.RESET_ALL}")
        print(f"    - Params:")
        for p, v in t.params.items():
            print(f"      • {Fore.YELLOW}{p}{Style.RESET_ALL}: {v}")
        print("-" * 40)

def make_env(args):
    """Create environment with action masking if needed."""
    env = multiEqn(
        normalize_rewards=args.normalize_rewards,
        state_rep=args.state_rep,
        level=args.level,
        generalization=args.generalization
    )
    if args.agent_type in ["ppo-mask", "ppo-cnn", "ppo-gnn", "ppo-gnn1"]:
        env = ActionMasker(env, lambda e: e.action_mask)
    return env

def train_and_evaluate(args, lr, ent_coef, n_layers, hidden_dim):
    """Train the agent and evaluate it over multiple repeats."""
    train_accs, avg_tsolve_list, max_test_accs = [], [], []

    for repeat_idx in range(1, args.n_repeats + 1):
        seed = int(time.time()) % 2**32
        np.random.seed(seed)
        th.manual_seed(seed)

        env = DummyVecEnv([lambda: make_env(args) for _ in range(args.n_envs)])
        policy_kwargs = {"net_arch": [hidden_dim] * n_layers}
        sb3_kwargs = {
            "ent_coef": ent_coef,
            "learning_rate": lr,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "vf_coef": args.vf_coef,
            "clip_range": args.clip_range,
            "max_grad_norm": args.max_grad_norm,
            "policy_kwargs": policy_kwargs
        }

        agent = get_agent(args.agent_type, env, **sb3_kwargs)
        agent.learn(total_timesteps=args.Ntrain)

        train_results = evaluate_agent(agent, env, env.get_attr('train_eqns')[0])
        test_results = evaluate_agent(agent, env, env.get_attr('test_eqns')[0])

        train_acc = sum(1 for tsolve in train_results.values() if tsolve != 0.0) / len(train_results)
        tsolve_vals = [tsolve for tsolve in train_results.values() if tsolve is not None]
        avg_tsolve = np.mean(tsolve_vals) if tsolve_vals else None

        train_accs.append(train_acc)
        avg_tsolve_list.append(avg_tsolve)
        max_test_accs.append(max(test_results.values(), default=0))

    return {
        "train_acc": np.mean(train_accs),
        "train_acc_std": np.std(train_accs),
        "avg_tsolve_mean": np.mean(avg_tsolve_list),
        "avg_tsolve_std": np.std(avg_tsolve_list),
        "max_test_acc_mean": np.mean(max_test_accs),
        "max_test_acc_std": np.std(max_test_accs)
    }

def evaluate_agent(agent, env, equation_list, n_eval_episodes=10):
    """Evaluate agent on a list of equations."""
    results = {}
    for eqn in equation_list:
        eqn_successes = 0
        for _ in range(n_eval_episodes):
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


# === Checkpointing and Live Result Saving ===
def save_trial_to_csv(trial, args):
    """Save trial results to a CSV file immediately after completion."""
    results_path = os.path.join(args.save_dir, "optuna_live_results.csv")
    trial_dict = {
        "trial_number": trial.number,
        "train_acc_mean": trial.user_attrs["train_acc_mean"],
        "train_acc_std": trial.user_attrs["train_acc_std"],
        "avg_tsolve_mean": trial.user_attrs["avg_tsolve_mean"],
        "avg_tsolve_std": trial.user_attrs["avg_tsolve_std"],
        "max_test_acc_mean": trial.user_attrs["max_test_acc_mean"],
        "max_test_acc_std": trial.user_attrs["max_test_acc_std"],
    }
    trial_dict.update(trial.params)

    results_df = pd.DataFrame([trial_dict])
    if not os.path.exists(results_path):
        results_df.to_csv(results_path, index=False)
    else:
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    print(f"💾 Trial {trial.number} saved to {results_path}")


class SaveTrialCallback:
    """Optuna callback to save results after each trial."""
    def __init__(self, args):
        self.args = args

    def __call__(self, study, trial):
        save_trial_to_csv(trial, self.args)


def save_study_results(study, args):
    """Save final study results."""
    results_path = os.path.join(args.save_dir, "optuna_results.csv")
    study.trials_dataframe().to_csv(results_path, index=False)
    print(f"✅ Study results saved to {results_path}")

    best_params_path = os.path.join(args.save_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"🎯 Best parameters saved to {best_params_path}")


# === Visualization ===
def save_optuna_plots(study, args):
    """Save Optuna plots for optimization history and parameter importance."""
    fig1 = plot_optimization_history(study)
    fig1.write_image(os.path.join(args.save_dir, "optuna_optimization_history.png"))

    evaluator = MeanDecreaseImpurityImportanceEvaluator()
    fig2 = plot_param_importances(study, evaluator=evaluator)
    fig2.write_image(os.path.join(args.save_dir, "optuna_param_importance.png"))

    print("✅ Optuna plots saved.")


def objective(trial, args):
    """Objective function for Optuna hyperparameter tuning."""
    global best_trial_so_far

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.2)
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])

    results = train_and_evaluate(args, lr, ent_coef, n_layers, hidden_dim)

    trial.set_user_attr("train_acc_mean", results["train_acc"])
    trial.set_user_attr("train_acc_std", results["train_acc_std"])
    trial.set_user_attr("avg_tsolve_mean", results["avg_tsolve_mean"])
    trial.set_user_attr("avg_tsolve_std", results["avg_tsolve_std"])
    trial.set_user_attr("max_test_acc_mean", results["max_test_acc_mean"])
    trial.set_user_attr("max_test_acc_std", results["max_test_acc_std"])

    # Check and log if this is the best trial
    if best_trial_so_far is None or results["train_acc"] > best_trial_so_far["train_acc"]:
        best_trial_so_far = results
        print(Fore.GREEN + "\n🌟 New Best Trial Found!" + Style.RESET_ALL)
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        print("-" * 60)

    return results["train_acc"]


def main(args):
    """Main entry point for Optuna tuning with checkpointing."""
    checkpoint_path = os.path.join(args.save_dir, "optuna_checkpoint.db")

    if os.path.exists(checkpoint_path):
        print(f"📂 Resuming from checkpoint: {checkpoint_path}")
        study = optuna.load_study(
            storage=f"sqlite:///{checkpoint_path}",
            study_name="multi_eqn_curriculum"
        )
    else:
        print("🆕 Starting a new Optuna study...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
            storage=f"sqlite:///{checkpoint_path}",
            study_name="multi_eqn_curriculum"
        )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        callbacks=[SaveTrialCallback(args)],
        show_progress_bar=True
    )

    save_study_results(study, args)
    save_optuna_plots(study, args)

    # Show top trials sorted by train accuracy
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    pretty_print_top_trials(sorted_trials, top_n=5)


# === Main Script Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-gnn')
    parser.add_argument('--state_rep', type=str, default='graph_integer_2d')
    parser.add_argument('--Ntrain', type=int, default=10**2, help='Number of training steps')
    parser.add_argument('--intrinsic_reward', type=str, default='None')
    parser.add_argument('--normalize_rewards', type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=True)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='data/tuning/curriculum/optuna')
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--level', type=int, default=7)
    parser.add_argument('--generalization', type=str, default='random')

    # PPO Parameters
    parser.add_argument('--n_steps', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)

    # Tuning Parameters
    parser.add_argument('--n_trials', type=int, default=2, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=6, help='Number of parallel workers')
    parser.add_argument('--n_repeats', type=int, default=2, help='Number of repeats per trial')

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.agent_type)
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)

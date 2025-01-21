"""
Script to compare RL performance (T_solve) between:
1. PPO with state_rep='integer_1d' and intrinsic_reward=None
2. PPO-CNN with state_rep='integer_2d' and intrinsic_reward=None

Runs multiple trials per equation, computes mean/std of Tsolve, and prints results in a formatted table.
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_single_eqn import main


# --------------------------------------
# Training function
# --------------------------------------
def train_equation(params):
    """Runs the training process for a given equation with multiple trials."""
    main_eqn, args = params
    print(f"\n🚀 Starting training for: {main_eqn} (State Rep: {args.state_rep}, Agent: {args.agent_type})")

    Tsolve_trials = []
    for trial in range(args.Ntrial):
        print(f"   🎲 Trial {trial + 1}/{args.Ntrial} for {main_eqn}...")
        args.main_eqn = main_eqn
        Tsolve, _ = main(args)  # Run training
        if Tsolve is not None:
            Tsolve_trials.append(int(Tsolve))  # 🔹 Convert np.int64 → Python int

    # Handle case where all trials return None
    if not Tsolve_trials:
        results = {"mean": None, "std": None, "trials": []}
    else:
        results = {
            "mean": float(np.mean(Tsolve_trials)),  # 🔹 Convert to float
            "std": float(np.std(Tsolve_trials)),  # 🔹 Convert to float
            "trials": [int(t) for t in Tsolve_trials]
        }

    return main_eqn, results


# --------------------------------------
# Execution wrapper
# --------------------------------------
def run_experiments(eqns, args):
    """Runs training sequentially for multiple equations."""
    results = {}
    for eqn in eqns:
        eqn, result = train_equation((eqn, args))
        results[eqn] = result
    return results


# --------------------------------------
# Main function
# --------------------------------------
def main_comparison():
    parser = argparse.ArgumentParser(description="Compare PPO and PPO-CNN on different equations.")
    parser.add_argument("--Ntrain", type=int, default=10**5, help="Number of training steps")
    parser.add_argument("--Ntrial", type=int, default=10, help="Number of trials per equation")
    parser.add_argument("--normalized_rewards", type=int, default=1, help="Number of trials per equation")
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=True, help="Normalize rewards (True/False)")
    parser.add_argument("--log_interval", type=int, default=None, help="Log interval")
    parser.add_argument("--save_dir", type=str, default="data/misc/", help="Directory to save logs")
   

    args = parser.parse_args()

    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**5)

    # --------------------------------------
    # Define equations to test
    # --------------------------------------
    MAIN_EQNS = [
        "a*x", "x + b", "a*x + b", "a/x + b", "c*(a*x + b) + d",
        "sqrt(a*x+b) - c",  "(a*x**2+b)**2 + c", "d/(a*x + b) + c", "e*(a*x + b) + (c*x + d)",
        "(a*x + b)/(c*x + d) + e"
    ][:7]

    print("\n🧪 Running experiments for multiple equations...")

    # Configurations to compare
    configs = {
        "PPO": {
            "agent_type": "ppo",
            "state_rep": "integer_1d",
            "intrinsic_reward": "None"
        },
        "PPO-CNN": {
            "agent_type": "ppo-cnn",
            "state_rep": "integer_2d",
            "intrinsic_reward": "None"
        }
    }

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n🔬 Running experiments for {config_name} (State Rep: {config['state_rep']})...\n")

        args.agent_type = config["agent_type"]
        args.state_rep = config["state_rep"]
        args.intrinsic_reward = config["intrinsic_reward"]

        # Run training
        results = run_experiments(MAIN_EQNS, args)
        all_results[config_name] = results

    # --------------------------------------
    # Format and display results
    # --------------------------------------
    table_data = []
    for eqn in MAIN_EQNS:
        ppo_mean, ppo_std = all_results["PPO"][eqn]["mean"], all_results["PPO"][eqn]["std"]
        ppo_cnn_mean, ppo_cnn_std = all_results["PPO-CNN"][eqn]["mean"], all_results["PPO-CNN"][eqn]["std"]

        table_data.append([
            eqn,
            f"{ppo_mean:.1f} ± {ppo_std:.1f}" if ppo_mean is not None else "N/A",
            f"{ppo_cnn_mean:.1f} ± {ppo_cnn_std:.1f}" if ppo_cnn_mean is not None else "N/A"
        ])

    df_results = pd.DataFrame(table_data, columns=["Equation", "PPO", "PPO-CNN"])

    print("\n📊 Training Results:")
    print(df_results.to_string(index=False))  # Pretty print table

    # Save results
    os.makedirs("data/comparison", exist_ok=True)
    df_results.to_csv("data/comparison/results_comparison.csv", index=False)
    print("\n📂 Results saved to: data/comparison/results_comparison.csv")

    print("\n✅ Experiment comparison completed!")


if __name__ == "__main__":
    main_comparison()


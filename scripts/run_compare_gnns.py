#!/usr/bin/env python3
"""
Script to compare RL performance (T_solve) between different configurations.

Runs multiple trials per equation, computes mean/std of Tsolve and win%, and prints results in a formatted table.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_single_eqn import main


# --------------------------------------
# Training function
# --------------------------------------
def train_equation(params):
    """Runs the training process for a given equation with multiple trials."""
    main_eqn, args = params
    print(f"\n🚀 Starting training for: {main_eqn} (State Rep: {args.state_rep}, Agent: {args.agent_type})", flush=True)

    Tsolve_trials = []
    success_count = 0

    for trial in range(args.Ntrial):
        print(f"   🎲 Trial {trial + 1}/{args.Ntrial} for {main_eqn}...", flush=True)
        args.main_eqn = main_eqn
        Tsolve, _ = main(args)  # Run training
        if Tsolve is not None:
            Tsolve_trials.append(int(Tsolve))  # Convert np.int64 → Python int
            success_count += 1  # Count successful trials

    # Handle case where all trials return None
    if not Tsolve_trials:
        results = {"mean": None, "std": None, "win%": 0.0, "trials": []}
    else:
        results = {
            "mean": float(np.mean(Tsolve_trials)),  # Convert to float
            "std": float(np.std(Tsolve_trials)),      # Convert to float
            "win%": (success_count / args.Ntrial) * 100,  # Calculate success rate
            "trials": [int(t) for t in Tsolve_trials]
        }

    return main_eqn, results


# --------------------------------------
# Execution wrapper (Parallel or Sequential)
# --------------------------------------
def run_experiments(eqns, args):
    """Runs training for multiple equations, using multiprocessing if requested."""
    if args.parallel:
        print(f"\n⚡ Running experiments in parallel with {args.num_workers} workers...\n", flush=True)
        with mp.Pool(args.num_workers) as pool:
            async_result = pool.map_async(train_equation, [(eqn, args) for eqn in eqns])
            try:
                # Set a timeout (in seconds) to catch any hanging worker; adjust as needed.
                results_list = async_result.get(timeout=86400)
            except mp.TimeoutError:
                print("❌ A multiprocessing worker timed out.", flush=True)
                pool.terminate()
                pool.join()
                raise
            results = dict(results_list)
    else:
        print("\n🛠 Running experiments sequentially...\n", flush=True)
        results = {}
        for eqn in eqns:
            eqn, result = train_equation((eqn, args))
            results[eqn] = result
    return results


# --------------------------------------
# Main function
# --------------------------------------
def main_comparison():
    parser = argparse.ArgumentParser(description="Compare PPO-GNN and PPO-GNN1 on different equations.")
    parser.add_argument("--Ntrain", type=int, default=3 * 10**6, help="Number of training steps")
    parser.add_argument("--Ntrial", type=int, default=5, help="Number of trials per equation")
    parser.add_argument("--normalized_rewards", type=int, default=1, help="Normalized rewards flag")
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=True, help="Normalize rewards (True/False)")
    parser.add_argument("--log_interval", type=int, default=None, help="Log interval")
    parser.add_argument("--save_dir", type=str, default="data/misc/", help="Directory to save logs")
    parser.add_argument("--parallel", type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=False, help="Run experiments in parallel")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of parallel workers")

    args = parser.parse_args()

    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**5)

    # --------------------------------------
    # Define equations to test (using only the final two equations)
    # --------------------------------------
    MAIN_EQNS = [
        "a*x", "x + b", "a*x + b", "a/x + b", "c*(a*x + b) + d",
        "d/(a*x + b) + c", "e*(a*x + b) + (c*x + d)",
        "(a*x + b)/(c*x + d) + e"
    ][-3:]

    print("\n🧪 Running experiments for multiple equations...", flush=True)

    # --------------------------------------
    # Configurations to compare (final two only)
    # --------------------------------------
    configs = {
        "PPO-GNN": {
            "agent_type": "ppo-gnn",
            "state_rep": "graph_integer_2d",
            "intrinsic_reward": "None"
        },
        "PPO-GNN1": {
            "agent_type": "ppo-gnn1",
            "state_rep": "graph_integer_2d",
            "intrinsic_reward": "None"
        }
    }

    all_results = {}

    # Run experiments for each configuration
    for config_name, config in configs.items():
        print(f"\n🔬 Running experiments for {config_name} (State Rep: {config['state_rep']})...\n", flush=True)
        args.agent_type = config["agent_type"]
        args.state_rep = config["state_rep"]
        args.intrinsic_reward = config["intrinsic_reward"]

        # Run training (parallel or sequential)
        results = run_experiments(MAIN_EQNS, args)
        all_results[config_name] = results

    # --------------------------------------
    # Format and display results (automatically fill table from config keys)
    # --------------------------------------
    # Create header from config keys
    header = ["Equation"] + list(configs.keys())
    table_data = []

    for eqn in MAIN_EQNS:
        row = [eqn]
        for config_name in configs.keys():
            result = all_results[config_name][eqn]
            if result["mean"] is not None:
                cell = f"{result['mean']:.1f} ± {result['std']:.1f}, {result['win%']:.1f}%"
            else:
                cell = "N/A"
            row.append(cell)
        table_data.append(row)

    df_results = pd.DataFrame(table_data, columns=header)

    print("\n📊 Training Results:", flush=True)
    print(df_results.to_string(index=False), flush=True)  # Pretty print table

    # Save results
    os.makedirs("data/comparison", exist_ok=True)
    df_results.to_csv("data/comparison/gnn_results_comparison.csv", index=False)
    print("\n📂 Results saved to: data/comparison/gnn_results_comparison.csv", flush=True)

    print("\n✅ Experiment comparison completed!", flush=True)


if __name__ == "__main__":
    main_comparison()

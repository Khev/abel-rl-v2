"""
Script to compare RL performance (T_solve) between:
1. PPO with state_rep='integer_1d'
2. PPO-CNN with state_rep='integer_2d'
3. PPO-GNN with state_rep='graph_integer_2d'

Runs multiple trials per equation, computes mean/std of Tsolve and win%, and prints results in a formatted table.
"""

import sys
import os
import argparse
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
    success_count = 0

    for trial in range(args.Ntrial):
        print(f"   🎲 Trial {trial + 1}/{args.Ntrial} for {main_eqn}...")
        args.main_eqn = main_eqn
        Tsolve, _ = main(args)  # Run training
        if Tsolve is not None:
            Tsolve_trials.append(int(Tsolve))  # 🔹 Convert np.int64 → Python int
            success_count += 1  # Count successful trials

    # Handle case where all trials return None
    if not Tsolve_trials:
        results = {"mean": None, "std": None, "win%": 0.0, "trials": []}
    else:
        results = {
            "mean": float(np.mean(Tsolve_trials)),  # 🔹 Convert to float
            "std": float(np.std(Tsolve_trials)),  # 🔹 Convert to float
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
        print(f"\n⚡ Running experiments in parallel with {args.num_workers} workers...\n")
        with mp.Pool(args.num_workers) as pool:
            results = dict(pool.map(train_equation, [(eqn, args) for eqn in eqns]))
    else:
        print("\n🛠 Running experiments sequentially...\n")
        results = {}
        for eqn in eqns:
            eqn, result = train_equation((eqn, args))
            results[eqn] = result
    return results


# --------------------------------------
# Main function
# --------------------------------------
def main_comparison():
    parser = argparse.ArgumentParser(description="Compare PPO, PPO-CNN, and PPO-GNN on different equations.")
    parser.add_argument("--Ntrain", type=int, default=10**5, help="Number of training steps")
    parser.add_argument("--Ntrial", type=int, default=5, help="Number of trials per equation")
    parser.add_argument("--normalized_rewards", type=int, default=1, help="Number of trials per equation")
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=True, help="Normalize rewards (True/False)")
    parser.add_argument("--log_interval", type=int, default=None, help="Log interval")
    parser.add_argument("--save_dir", type=str, default="data/misc/", help="Directory to save logs")
    parser.add_argument("--parallel", type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=False, help="Run experiments in parallel")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of parallel workers")

    args = parser.parse_args()

    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**5)

    # --------------------------------------
    # Define equations to test
    # --------------------------------------
    MAIN_EQNS = [
        "a*x", "x + b", "a*x + b", "a/x + b", "c*(a*x + b) + d",
        "sqrt(a*x+b) - c", "(a*x**2+b)**2 + c", "d/(a*x + b) + c", "e*(a*x + b) + (c*x + d)",
        "(a*x + b)/(c*x + d) + e"
    ][:6]

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
        },
        "PPO-GNN": {
            "agent_type": "ppo-gnn",
            "state_rep": "graph_integer_2d",
            "intrinsic_reward": "None"
        }
    }

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n🔬 Running experiments for {config_name} (State Rep: {config['state_rep']})...\n")

        args.agent_type = config["agent_type"]
        args.state_rep = config["state_rep"]
        args.intrinsic_reward = config["intrinsic_reward"]

        # Run training (parallel or sequential)
        results = run_experiments(MAIN_EQNS, args)
        all_results[config_name] = results

    # --------------------------------------
    # Format and display results
    # --------------------------------------
    table_data = []
    for eqn in MAIN_EQNS:
        ppo_result = all_results["PPO"][eqn]
        ppo_cnn_result = all_results["PPO-CNN"][eqn]
        ppo_gnn_result = all_results["PPO-GNN"][eqn]

        table_data.append([
            eqn,
            f"{ppo_result['mean']:.1f} ± {ppo_result['std']:.1f}, {ppo_result['win%']:.1f}%" if ppo_result["mean"] is not None else "N/A",
            f"{ppo_cnn_result['mean']:.1f} ± {ppo_cnn_result['std']:.1f}, {ppo_cnn_result['win%']:.1f}%" if ppo_cnn_result["mean"] is not None else "N/A",
            f"{ppo_gnn_result['mean']:.1f} ± {ppo_gnn_result['std']:.1f}, {ppo_gnn_result['win%']:.1f}%" if ppo_gnn_result["mean"] is not None else "N/A"
        ])

    df_results = pd.DataFrame(
        table_data,
        columns=["Equation", "PPO (Mean ± Std, Win%)", "PPO-CNN (Mean ± Std, Win%)", "PPO-GNN (Mean ± Std, Win%)"]
    )

    print("\n📊 Training Results:")
    print(df_results.to_string(index=False))  # Pretty print table

    # Save results
    os.makedirs("data/comparison", exist_ok=True)
    df_results.to_csv("data/comparison/results_comparison.csv", index=False)
    print("\n📂 Results saved to: data/comparison/results_comparison.csv")

    print("\n✅ Experiment comparison completed!")


if __name__ == "__main__":
    main_comparison()

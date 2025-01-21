"""
Script to run reinforcement learning experiments for solving equations.
Runs multiple trials per equation, computes min/mean/max Tsolve.
Supports parallel and sequential execution.
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import multiprocessing as mp

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_single_eqn import main


# --------------------------------------
# Training function
# --------------------------------------
def train_equation(params):
    """Runs the training process for a given equation with multiple trials."""
    main_eqn, args = params
    print(f"\n🚀 Starting training for: {main_eqn}")

    Tsolve_trials = []
    for trial in range(args.Ntrial):
        print(f"   🎲 Trial {trial + 1}/{args.Ntrial} for {main_eqn}...")
        args.main_eqn = main_eqn  # Set the equation for the current trial
        Tsolve, _ = main(args)  # Run training
        if Tsolve is not None:
            Tsolve_trials.append(int(Tsolve))  # 🔹 Convert np.int64 → Python int

    # Handle case where all trials return None
    if not Tsolve_trials:
        results = {
            "Equation": main_eqn,
            "Min Tsolve": None,
            "Mean Tsolve": None,
            "Max Tsolve": None,
            "Trials": []
        }
    else:
        results = {
            "Equation": main_eqn,
            "Min Tsolve": int(np.min(Tsolve_trials)),  # 🔹 Convert to int
            "Mean Tsolve": float(np.mean(Tsolve_trials)),  # 🔹 Convert to float
            "Max Tsolve": int(np.max(Tsolve_trials)),  # 🔹 Convert to int
            "Trials": [int(t) for t in Tsolve_trials]  # 🔹 Convert all elements to int
        }

    return main_eqn, results


# --------------------------------------
# Parallel execution wrapper
# --------------------------------------
def run_parallel(eqns, args, num_workers=2):
    """Runs training in parallel for multiple equations."""
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(train_equation, [(eq, args) for eq in eqns])
    return dict(results)


# --------------------------------------
# Sequential execution wrapper
# --------------------------------------
def run_sequential(eqns, args):
    """Runs training sequentially for multiple equations."""
    results = {}
    for eqn in eqns:
        eqn, result = train_equation((eqn, args))
        results[eqn] = result
    return results


# --------------------------------------
# Main function
# --------------------------------------
def main_runner():
    parser = argparse.ArgumentParser(description="Run RL experiments on different equations.")
    parser.add_argument("--Ntrain", type=int, default=3*10**6, help="Number of training steps")
    parser.add_argument("--Ntrial", type=int, default=5, help="Number of trials per equation")
    parser.add_argument("--agent_type", type=str, default="ppo-mask", choices=["a2c", "ppo", "dqn"], help="RL agent type")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=True, help="Normalize rewards (True/False)")
    parser.add_argument("--log_interval", type=int, default=None, help="Log interval")
    parser.add_argument("--save_dir", type=str, default="data/misc/", help="Directory to save logs")
    parser.add_argument("--intrinsic_reward", type=str, default="ICM", choices=["ICM", "E3B", "RIDE", "None"])
    parser.add_argument("--state_rep", type=str, default="integer_1d")

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
    ]

    print("\n🧪 Running experiments for multiple equations...")

    # Run experiments in parallel or sequentially
    if args.parallel:
        print(f"🌍 Running in PARALLEL mode with {args.num_workers} workers...")
        results = run_parallel(MAIN_EQNS, args, num_workers=args.num_workers)
    else:
        print("🛠 Running in SEQUENTIAL mode...")
        results = run_sequential(MAIN_EQNS, args)

    # --------------------------------------
    # Format and save results
    # --------------------------------------
    df_results = pd.DataFrame.from_dict(results, orient="index")

    print("\n📊 Training Results:")
    print(df_results.to_string(index=False))  # Pretty print table

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f"results_{args.intrinsic_reward}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n📂 Results saved to CSV: {csv_path}")

    print("\n✅ All experiments completed!")
    print("📊 Results saved in:", args.save_dir)


if __name__ == "__main__":
    main_runner()

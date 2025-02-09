import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import multiprocessing
from training.train_multi_eqn import main

# Function to run a single instance of main() with args
def run_training(level):
    print(f"Starting worker for level {level}...")

    # Manually construct arguments
    args = argparse.Namespace(
        agent_type='ppo-gnn1',
        state_rep='graph_integer_2d',
        Ntrain=10**7,
        intrinsic_reward='None',
        normalize_rewards=True,
        log_interval=None,
        save_dir=f'data/generalize/level{level}',
        verbose=0,
        level=level,
        generalization='lexical'
    )

    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**4)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results for level {level} saved to {args.save_dir}")

    # Run training
    train_results, test_results, max_test_acc_one_shot = main(args)
    train_acc = np.mean(list(train_results.values())) if train_results else 0.0
    test_acc = np.mean(list(test_results.values())) if test_results else 0.0

    return level, train_acc, test_acc, max_test_acc_one_shot 

if __name__ == "__main__":
    # Set parameters
    levels = range(7)  # Iterate over levels 0 to 7
    parallel = True  # Whether to run in parallel

    if parallel:
        num_workers = 2
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(run_training, levels)
    else:
        results = [run_training(level) for level in levels]

    # Print final results
    print("\n==== Summary of max_test_acc_one_shot across all trials ====")
    for level, train_acc, test_acc, max_test_acc_one_shot in results:
        print(f"Level {level}: train, test, test_max_one_shot = {train_acc:.2f}, {test_acc:.2f}, {max_test_acc_one_shot:.2f}")

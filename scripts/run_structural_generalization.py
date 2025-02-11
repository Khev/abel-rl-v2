import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import multiprocessing
import numpy as np
from training.train_multi_eqn import main

# Function to run a single instance of main() with args
def run_training(level):
    print(f"Starting training for level {level}...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-gnn',
                        choices=['dqn', 'a2c', 'ppo', 'ppo-mask', 'ppo-cnn', 'ppo-gnn', 'ppo-gnn1'])
    parser.add_argument('--state_rep', type=str, default='graph_integer_2d', help='State representation/encoding')
    parser.add_argument('--Ntrain', type=int, default=10**7, help='Number of training steps')
    parser.add_argument('--intrinsic_reward', type=str, default='None',
                        choices=['ICM', 'E3B', 'RIDE', 'None'], help='Type of intrinsic reward')
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"),
                        default=True, help="Normalize rewards (True/False)")
    parser.add_argument('--log_interval', type=int, default=None, help='Log interval')
    parser.add_argument('--save_dir', type=str, default=f'data/generalize/structural/run_level{level}', help='Directory to save results')
    parser.add_argument('--verbose', type=int, default=0)

    # Generalization parameters
    parser.add_argument('--level', type=int, default=level)
    parser.add_argument('--generalization', type=str, default='structural')

    args = parser.parse_args()

    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**4)

    # Set save directory for each level
    args.save_dir = os.path.join(args.save_dir, f"{args.generalization}/level{level}")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Training for level {level} results saved to {args.save_dir}")

    # Run training
    train_results, test_results, max_test_acc_one_shot = main(args)
    train_acc = np.mean(list(train_results.values())) if train_results else 0.0
    test_acc = np.mean(list(test_results.values())) if test_results else 0.0

    return level, train_acc, test_acc, max_test_acc_one_shot


if __name__ == "__main__":

    # List of levels to run
    levels_to_run = [4, 5, 6, 7]  # Define the levels you want to iterate over
    parallel = True  # Set to True to run in parallel

    if parallel:
        with multiprocessing.Pool(len(levels_to_run)) as pool:
            results = pool.map(run_training, levels_to_run)
    else:
        results = [run_training(level) for level in levels_to_run]

    # Print final results
    print("\n==== Summary of max_test_acc_one_shot across all levels ====")
    for level, train_acc, test_acc, max_test_acc_one_shot in results:
        print(f"Level {level}: train, test, test_max_one_shot = {train_acc:.2f}, {test_acc:.2f}, {max_test_acc_one_shot:.2f}")

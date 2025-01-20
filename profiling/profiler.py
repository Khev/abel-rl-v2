import cProfile
import pstats
import io
from training.train_single_eqn import main  

def profile_main():
    """Wraps the main function of train.py under cProfile."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-mask', choices=['dqn', 'a2c', 'ppo', 'ppo-mask'])
    parser.add_argument('--main_eqn', type=str, default='c*(a*x+b)+d')
    parser.add_argument('--Ntrain', type=int, default=10**4)
    parser.add_argument('--log_interval', type=int, default=10**4)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--intrinsic_reward', type=str, default='ICM', choices=['ICM', 'E3B', 'RIDE', 'None'])
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), default=True)

    args = parser.parse_args()
    
    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**4)
    
    # Set save_dir
    if args.save_dir is None:
        args.save_dir = f'data/{args.main_eqn}' if args.main_eqn is not None else f'data/general_eqn'
    
    main(args)  # Run training

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    profile_main()

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'  # Sort by cumulative execution time
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)  # Print top 30 slowest functions

    # Save profiling output to a file
    with open("profiling_results.txt", "w") as f:
        f.write(s.getvalue())

    print("\nProfiling complete! Check profiling_results.txt for details.")


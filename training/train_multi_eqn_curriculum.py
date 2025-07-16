import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, json, pickle, time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt

from envs.multi_eqn_curriculum import multiEqn

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


from utils.utils_train import get_intrinsic_reward, get_device, CustomGNNPolicy, get_agent
from utils.custom_functions import operation_names
from utils.utils_general import print_parameters, print_header
from colorama import Fore, Style

device = get_device()

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO, A2C


def print_eval_results(test_results, label=""):
    """
    test_results is a dict: { eqn_string : success_rate_float, ... }
    """
    print(f"{label} Equations")
    # Convert dict to a DataFrame with columns = ['Eqn', 'Win%']
    df = pd.DataFrame(
        [{'Eqn': eqn, 'Win%': f"{winpct:.1f}%"} for eqn, winpct in test_results.items()]
    )
    print(df.to_string(index=False))
    print()

def print_results_dict_as_df(title, d):
    """
    d is presumably { eqn_str: Tsolve_step or None }.
    We'll convert to a DataFrame with columns ['Eqn', 'TSolve'].
    """
    print(f"\n{title}")
    rows = []
    for eqn, tsolve in d.items():
        rows.append({'Eqn': eqn, 'TSolve': tsolve})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

# For PPO-mask
def get_action_mask(env):
    return env.action_mask

def evaluate_agent(agent, env, equation_list, n_eval_episodes=1):
    """
    Evaluate agent on each eqn in equation_list for n_eval_episodes per eqn.
    Returns a dict eqn -> success percentage.
    """
    results = {}

    for eqn in equation_list:
        eqn_successes = 0
        for ep in range(n_eval_episodes):
            # Ensure we reset correctly in DummyVecEnv
            #obs, _ = env.get_attr('reset')[0]() if hasattr(env, 'get_attr') else env.reset()
            obs = env.reset()
            env.env_method('set_equation', eqn)
            #breakpoint()
            done = [False]  # Ensure it's a list/array for vectorized envs
            while not done[0]:  # Extract first element
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if info[0].get('is_solved', False):
                    eqn_successes += 1
                    break
        success_rate = eqn_successes / n_eval_episodes * 100.0
        results[eqn] = success_rate

    return results


class TrainingLogger(BaseCallback):
    """
    Callback for logging reward statistics, evaluating the agent, and saving checkpoints at regular intervals.
    """

    def __init__(self, log_interval=1000, eval_interval=10000, save_dir=".", verbose=1, gen=None, eval_env=None):
        super(TrainingLogger, self).__init__(verbose)
        self.log_interval = log_interval
        self.eval_interval = log_interval
        self.save_dir = save_dir
        self.eval_env = eval_env  # Separate evaluation environment
        self.rewards_ext = []  # External rewards
        self.T_solve = None
        self.T_converge = None
        self.early_stopping = False
        self.gen = gen

        # We'll defer initialization of results_train until training actually starts:
        self.results_train = {}
        self.results_test = {}

        # Learning states
        self.logged_steps = []
        self.train_accuracy = []
        self.test_accuracy = []

        self.train_accuracy_one_shot = []
        self.test_accuracy_one_shot = []

        self.max_test_acc_one_shot = 0.0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _on_training_start(self):
        """
        Called once the training starts, when self.training_env is available.
        """
        # 'train_eqns' presumably is an attribute in your env that lists all training eqns
        # Because we are inside a DummyVecEnv, we can do get_attr('train_eqns')[0].
        train_eqns = getattr(self.training_env.envs[0], 'train_eqns', None)
        self.train_eqns = train_eqns
        if train_eqns is not None:
            self.results_train = {eqn: None for eqn in train_eqns}
        else:
            print("Warning: no 'train_eqns' found in env; results_train will remain empty.")

        test_eqns = getattr(self.training_env.envs[0], 'test_eqns', None)
        if test_eqns is not None:
            self.results_test = {eqn: None for eqn in test_eqns}
        else:
            print("No 'test_eqns' found in env. results_test will remain empty.")
        self.test_eqns = test_eqns

        if self.eval_env:
            print("\nInitial evaluation (t=0)...")
            train_results = evaluate_agent(self.model, self.eval_env, train_eqns)
            print_eval_results(train_results, label="Train")

            test_results = evaluate_agent(self.model, self.eval_env, test_eqns)
            print_eval_results(test_results, label="Test")

            self.logged_steps.append(0)  # Log step 0

            train_acc = np.mean(list(train_results.values()))
            test_acc = np.mean(list(test_results.values()))
            self.train_accuracy.append(np.mean(list(train_results.values())))
            self.test_accuracy.append(np.mean(list(test_results.values())))

            train_acc_one_shot = np.mean([100 if i > 0 else 0 for i in train_results.values()])
            test_acc_one_shot = np.mean([100 if i>0 else 0 for i in test_results.values()])
            self.train_accuracy_one_shot.append(train_acc_one_shot)
            self.test_accuracy_one_shot.append(test_acc_one_shot)
            self.max_test_acc_one_shot = max(self.max_test_acc_one_shot, test_acc_one_shot)

    
            print(Fore.GREEN + f'train: acc, acc_one_shot = {train_acc:.2f}, {train_acc_one_shot:.2f}' + Style.RESET_ALL)
            print(Fore.GREEN + f'test: acc, acc_one_shot = {test_acc:.2f}, {test_acc_one_shot:.2f}' + Style.RESET_ALL)
            print(Fore.RED + f'max test acc one shot = {self.max_test_acc_one_shot:.2f}', Style.RESET_ALL)


    def _on_step(self) -> bool:
        """
        This function is called at each step of training.
        """
        # Get latest external reward from the environment
        reward_ext = self.locals["rewards"][0]
        self.rewards_ext.append(reward_ext)
        info = self.locals["infos"][0]

        if info['is_solved']:
            main_eqn, lhs, rhs = info['main_eqn'], info['lhs'], info['rhs']
            #print(Fore.YELLOW + f'\nSolved {main_eqn} = 0 ==> {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)
            self.T_solve = self.num_timesteps

            if main_eqn not in self.results_train:
                self.results_train[main_eqn] = None

            if self.results_train[main_eqn] == None:
                self.results_train[main_eqn] = self.num_timesteps
                print(Fore.YELLOW + f'\nSolved {main_eqn} = 0 ==> {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)
                #print_results_dict_as_df("Training Results", self.results_train)

            # Stop training early if enabled
            if self.early_stopping:
                print(Fore.YELLOW + f"{main_eqn}: Early stopping for triggered at step {self.num_timesteps}!" + Style.RESET_ALL)
                return False  

        # Logging reward statistics
        if self.n_calls % self.log_interval == 0:
            mean_reward_ext = np.mean(self.rewards_ext[-self.log_interval:])
            min_reward_ext = np.min(self.rewards_ext[-self.log_interval:])
            max_reward_ext = np.max(self.rewards_ext[-self.log_interval:])

            main_eqn = info['main_eqn']
            print(f"{main_eqn}: Step {self.num_timesteps}: "
                  f"(min, mean, max)_reward_external = ({min_reward_ext:.2f}, {mean_reward_ext:.2f}, {max_reward_ext:.2f})")

        #Evaluation at intervals
        if self.eval_env and self.n_calls % self.eval_interval == 0:

            if self.gen == 'random':
                print('\n--------------------')
                print("Eqn: Sample, solve counts")
                print('--------------------')
                solve_counts  = self.eval_env.get_attr("solve_counts")[0]
                sample_counts = self.eval_env.get_attr("sample_counts")[0]
                for eqn in sorted(set(solve_counts) | set(sample_counts)):
                    s_cnt  = sample_counts.get(eqn, 0)
                    sol_cnt = solve_counts.get(eqn, 0)
                    print(f"{eqn}: {s_cnt}, {sol_cnt}")
                print()
            else:

                print("\nRunning evaluation...")
                #train_results = evaluate_agent(self.model, self.eval_env, self.eval_env.get_attr('train_eqns')[0], n_eval_episodes=10)
                train_results = evaluate_agent(self.model, self.eval_env, self.train_eqns)

                print_eval_results(train_results, label='Train')

                #test_results = evaluate_agent(self.model, self.eval_env, self.eval_env.get_attr('test_eqns')[0], n_eval_episodes=10)
                test_results = evaluate_agent(self.model, self.eval_env, self.test_eqns)
                print_eval_results(test_results, label='Test')

                self.logged_steps.append(self.num_timesteps)

                train_acc = np.mean(list(train_results.values()))
                test_acc = np.mean(list(test_results.values()))
                self.train_accuracy.append(np.mean(list(train_results.values())))
                self.test_accuracy.append(np.mean(list(test_results.values())))

                train_acc_one_shot = np.mean([100 if i > 0 else 0 for i in train_results.values()])
                test_acc_one_shot = np.mean([100 if i>0 else 0 for i in test_results.values()])
                self.train_accuracy_one_shot.append(train_acc_one_shot)
                self.test_accuracy_one_shot.append(test_acc_one_shot)

                self.max_test_acc_one_shot = max(self.max_test_acc_one_shot, test_acc_one_shot)
                print(Fore.GREEN + f'train: acc, acc_one_shot = {train_acc:.2f}, {train_acc_one_shot:.2f}' + Style.RESET_ALL)
                print(Fore.GREEN + f'test: acc, acc_one_shot = {test_acc:.2f}, {test_acc_one_shot:.2f}' + Style.RESET_ALL)
                print(Fore.RED + f'max test acc one shot = {self.max_test_acc_one_shot:.2f}', Style.RESET_ALL)

                # Early stopping
                if test_acc == 100.0 and train_acc == 100.0:
                    print(Fore.YELLOW + f"'train_acc = test_acc = 100. Early stopping step {self.num_timesteps}!" + Style.RESET_ALL)
                    return False

        return True  # Continue training


    def _on_training_end(self) -> None:
        """
        Called when training finishes. Plots train vs test accuracy curves.
        """
        print("\nFinal Training Completed. Plotting Learning Curves...")


class IntrinsicReward(BaseCallback):
    """
    A more efficient callback for logging intrinsic rewards in RL training.
    """

    def __init__(self, irs, verbose=0, log_interval=100):
        super(IntrinsicReward, self).__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.rewards_internal = []  # Store intrinsic rewards for logging
        self.log_interval = log_interval
        self.last_computed_intrinsic_rewards = None  # Store for logging

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        Instead of computing at each step, log previously computed intrinsic rewards.
        """
        if self.last_computed_intrinsic_rewards is not None:
            # Get last intrinsic reward from the rollout buffer
            intrinsic_reward = self.last_computed_intrinsic_rewards[-1]
            self.rewards_internal.append(intrinsic_reward)

        # ✅ Print intrinsic reward stats every `log_interval` steps
        if self.n_calls % self.log_interval == 0 and self.rewards_internal:
            mean_intrinsic = np.mean(self.rewards_internal[-self.log_interval:])
            min_intrinsic = np.min(self.rewards_internal[-self.log_interval:])
            max_intrinsic = np.max(self.rewards_internal[-self.log_interval:])
            main_eqn = self.locals["infos"][0]['main_eqn']
            print(f"{main_eqn}: Step {self.num_timesteps}: "
                  f"(min, mean, max)_reward_internal = ({min_intrinsic:.3f}, {mean_intrinsic:.3f}, {max_intrinsic:.3f})\n")

        return True

    def _on_rollout_end(self) -> None:
        """
        Efficiently compute intrinsic rewards once per rollout and store them.
        """
        obs = th.as_tensor(self.buffer.observations).float()
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"]).float()
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)

        # ✅ Compute **intrinsic rewards for the entire rollout** at once
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions,
                         rewards=rewards, terminateds=dones,
                         truncateds=dones, next_observations=new_obs),
            sync=True
        ).cpu().numpy()

        # ✅ Store them so `_on_step()` can access them
        self.last_computed_intrinsic_rewards = intrinsic_rewards

        # ✅ Add intrinsic rewards to the rollout buffer
        self.buffer.advantages += intrinsic_rewards
        self.buffer.returns += intrinsic_rewards



# -------------------------------------------------------------------------------------------------------
########################################### MAIN #########################################################
# -------------------------------------------------------------------------------------------------------

def main(args):

    t1 = time.time()

    # Print out args
    print('\n')
    params = vars(args)
    print_parameters(params)


    # # Make env
    # env = multiEqn(normalize_rewards=args.normalize_rewards, state_rep=args.state_rep, level=args.level, \
    #      generalization=args.generalization)
    # if args.agent_type in ["ppo-mask",'ppo-cnn','ppo-gnn','ppo-gnn1']:
    #     env = ActionMasker(env, get_action_mask)
    # env = DummyVecEnv([lambda: Monitor(env)])

    #Make env
    use_mem = True if args.use_memory == 'true' else False
    def make_env():
        env = multiEqn(normalize_rewards=args.normalize_rewards, state_rep=args.state_rep, \
             level=args.level, generalization=args.generalization, use_memory=use_mem)
        if args.agent_type in ["ppo-mask", "ppo-cnn", "ppo-gnn", "ppo-gnn1"]:
            env = ActionMasker(env, get_action_mask)
        return env

    env_fns = [lambda: make_env() for _ in range(args.n_envs)]
    #env = SubprocVecEnv(env_fns)
    env = DummyVecEnv(env_fns)


    # Make agent
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
    if args.agent_type == 'dqn':
        sb3_kwargs = {
                "gamma": args.gamma,
            }
    agent = get_agent(args.agent_type,env,**sb3_kwargs) 

    # Callback
    callback = TrainingLogger(log_interval=args.log_interval, save_dir=args.save_dir, eval_env=env, gen=args.generalization)

    # Intrinsic reward
    irs = get_intrinsic_reward(args.intrinsic_reward, env)
    if irs:
        callback_intrinsic_reward = IntrinsicReward(irs, log_interval=args.log_interval)
        callback = [callback, callback_intrinsic_reward]

    # Training
    print_header(f"Starting training for {args.Ntrain} timesteps", color='cyan')
    agent.learn(total_timesteps=args.Ntrain, callback=callback)

    # Save model
    agent.save(os.path.join(args.save_dir, f"{args.agent_type}_{args.intrinsic_reward}_trained_model"))
    print(f"Model saved to {args.save_dir}")

    # Save memory from env (pickle the SolveMemory instance)
    mem_path = os.path.join(args.save_dir, f"{args.agent_type}_{args.intrinsic_reward}_memory.pkl")
    mem = env.get_attr('mem')[0]  # Extract from VecEnv (first env)
    with open(mem_path, 'wb') as f:
        pickle.dump(mem, f)
    print(f"Memory saved to {mem_path}")

    # Evaluation
    print_header('Evaluating trained agent', color='cyan')
    train_results = evaluate_agent(agent, env, env.get_attr('train_eqns')[0], n_eval_episodes=10)
    test_results = evaluate_agent(agent, env, env.get_attr('test_eqns')[0], n_eval_episodes=10)

    print_eval_results(train_results, label='Train')
    #print_eval_results(test_results, label='Test')

    # Extract info
    if type(callback) == list:
        callback = callback[0]
    results_train, results_test = callback.results_train, callback.results_test

    train_save_path = os.path.join(args.save_dir, "Tsolve.json")

    train_dict_str_keys = {str(k): v for k, v in results_train.items()}
    with open(train_save_path, "w") as f:
        json.dump(train_dict_str_keys, f, indent=4)
    print(f"Saved train results to {train_save_path}")

    # Save solve counts
    solve_counts = env.get_attr("solve_counts")[0]
    solve_counts_save_path = os.path.join(args.save_dir, "solve_counts.json")
    with open(solve_counts_save_path, "w") as f:
        json.dump(solve_counts, f, indent=4)
    print(f"Saved solve counts to {solve_counts_save_path}")

    # Save sample counts
    sample_counts = env.get_attr("sample_counts")[0]
    sample_counts_save_path = os.path.join(args.save_dir, "sample_counts.json")
    with open(sample_counts_save_path, "w") as f:
        json.dump(sample_counts, f, indent=4)
    print(f"Saved sample counts to {sample_counts_save_path}")

    # Print results
    print_header(f"Final results", color='cyan')
    print_results_dict_as_df("Training Results", results_train)

    # Run time
    t2 = time.time()
    elapsed_time = t2 - t1
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # extract off max
    max_test_acc_one_shot = callback.max_test_acc_one_shot

    return train_results, results_test, max_test_acc_one_shot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-mask')
    parser.add_argument('--state_rep', type=str, default='integer_1d', help='State representation/encoding')
    parser.add_argument('--Ntrain', type=int, default=10**4, help='Number of training steps')
    parser.add_argument('--intrinsic_reward', type=str, default='ICM', choices=['ICM', 'E3B', 'RIDE', 'None'], \
                         help='Type of intrinsic reward')
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), \
         default=True, help="Normalize rewards (True/False)")
    parser.add_argument('--log_interval', type=int, default=None, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='data/curriculum', help='Directory to save the results')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--n_envs', type=int, default=1, help='Number of envs to run in parallel')


    # Agent aprqameters
    parser.add_argument('--ent_coef', type=float, default=0.05, help='Entropy coefficient for PPO or MaskablePPO')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the RL agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (gamma) for the RL agent')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per rollout in PPO')
    parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size for each gradient update')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train on each batch')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function loss coefficient')
    parser.add_argument('--clip_range', type=float, default=0.2, help='Clip range for PPO')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Gradient clipping norm')

    # New network parameters
    parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden layers in the policy network')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units per layer')

    # Generalization parameters
    parser.add_argument('--level', type=int, default=8)
    parser.add_argument('--generalization', type=str, default='poesia', choices=['lexical', 'structural', 'shallow','deep', 'random', 'poesia', 'poesia-full'])
    parser.add_argument('--use_memory', type=str, default='true')

    args = parser.parse_args()
    
    # Set default log_interval if not provided
    if args.log_interval is None:
        #args.log_interval = min(int(0.1 * args.Ntrain), 10**4)   
        args.log_interval = int(0.1 * args.Ntrain)
    
    # Set save_dir
    args.save_dir = os.path.join(args.save_dir, f"{args.generalization}/level{args.level}/{args.agent_type}_{args.intrinsic_reward}/")
    os.makedirs(args.save_dir, exist_ok=True)

    # Check for invalid (agent, state_rep pairs)
    if args.state_rep in ['graph_integer_1d', 'graph_integer_2d'] and args.agent_type not in ['ppo-gnn', 'ppo-gnn1']:
        raise ValueError(
        f"❌ ERROR: 'ppo-gnn' requires 'graph_integer_1d' or 'graph_integer_2d' as state_rep, "
        f"but got '{args.state_rep}'.\n"
        f"👉 Fix this by using 'state_rep=graph_integer_2d' in your config."
    )

    args_dict = vars(args)  # Convert argparse Namespace to a dict
    args_save_path = os.path.join(args.save_dir, "args.json")
    with open(args_save_path, "w") as f:
        import json
        json.dump(args_dict, f, indent=4)

    import envs.multi_eqn_curriculum
    print(envs.multi_eqn_curriculum.__file__)
    
    results_train, results_test, max_test_acc_one_shot = main(args)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, json, pickle, time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt

from envs.multi_eqn import multiEqn

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

def evaluate_agent(agent, env, equation_list, n_eval_episodes=10):
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
                action, _ = agent.predict(obs, deterministic=False)
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

    def __init__(self, log_interval=1000, eval_interval=10000, save_dir=".", verbose=1, eval_env=None):
        super(TrainingLogger, self).__init__(verbose)
        self.log_interval = log_interval
        self.eval_interval = log_interval
        self.save_dir = save_dir
        self.eval_env = eval_env  # Separate evaluation environment
        self.rewards_ext = []  # External rewards
        self.T_solve = None
        self.T_converge = None
        self.early_stopping = False

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
        eqns_attr = self.training_env.get_attr('train_eqns')
        if eqns_attr:
            train_eqns = eqns_attr[0]
            self.results_train = {eqn: None for eqn in train_eqns}
        else:
            print("Warning: no 'train_eqns' found in env; results_train will remain empty.")

        # If you have test_eqns, you can do the same for results_test (though you might
        # want to populate it after your final evaluation, see below).
        test_eqns_attr = self.training_env.get_attr('test_eqns')
        if test_eqns_attr:
            test_eqns = test_eqns_attr[0]
            self.results_test = {eqn: None for eqn in test_eqns}
        else:
            print("No 'test_eqns' found in env. results_test will remain empty.")

        if self.eval_env:
            print("\nInitial evaluation (t=0)...")
            train_results = evaluate_agent(self.model, self.eval_env, train_eqns, n_eval_episodes=10)
            print_eval_results(train_results, label="Train")

            test_results = evaluate_agent(self.model, self.eval_env, test_eqns, n_eval_episodes=10)
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

            print("\nRunning evaluation...")
            train_results = evaluate_agent(self.model, self.eval_env, self.eval_env.get_attr('train_eqns')[0], n_eval_episodes=10)
            print_eval_results(train_results, label='Train')

            test_results = evaluate_agent(self.model, self.eval_env, self.eval_env.get_attr('test_eqns')[0], n_eval_episodes=10)
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


        # Save model checkpoint at intervals
        # if self.n_calls % self.eval_interval == 0:
        #     save_path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}.zip")
        #     self.model.save(save_path)
        #     print(f"Checkpoint saved: {save_path}")

        return True  # Continue training


    def _on_training_end(self) -> None:
        """
        Called when training finishes. Plots train vs test accuracy curves.
        """
        print("\nFinal Training Completed. Plotting Learning Curves...")

        plt.figure(figsize=(10, 6))
        plt.plot(self.logged_steps, self.train_accuracy, label="Train Accuracy", 
                marker='o', linestyle='-', color='b', markersize=6, linewidth=2)
        plt.plot(self.logged_steps, self.test_accuracy, label="Test Accuracy", 
                marker='s', linestyle='--', color='r', markersize=6, linewidth=2)
        plt.plot(self.logged_steps, self.train_accuracy_one_shot, label="Train Accuracy (1-shot)", 
                marker='D', linestyle='-.', color='g', markersize=6, linewidth=2)
        plt.plot(self.logged_steps, self.test_accuracy_one_shot, label="Test Accuracy (1-shot)", 
                marker='^', linestyle=':', color='m', markersize=6, linewidth=2)

        plt.xlabel("Training Steps")
        plt.ylabel("Success Rate (%)")
        plt.title("Train vs. Test Learning Progress")
        plt.legend()
        plt.grid()
        plt.ylim([0,105])
        plt.savefig(os.path.join(self.save_dir, "learning_curve.png"))
        #plt.show()  # Only if running interactively
        plt.close()


        # Save learning curve data
        save_path = os.path.join(self.save_dir, "learning_progress.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({
                "steps": self.logged_steps,
                "train_success": self.train_accuracy,
                "train_success_one_shot": self.train_accuracy_one_shot,
                "test_success": self.test_accuracy,
                "test_success_one_shot": self.test_accuracy_one_shot

            }, f)
        print(f"Saved learning progress to {save_path}")



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

    # Make env
    env = multiEqn(normalize_rewards=args.normalize_rewards, state_rep=args.state_rep, level=args.level, \
         generalization=args.generalization)
    if args.agent_type in ["ppo-mask",'ppo-cnn','ppo-gnn','ppo-gnn1']:
        env = ActionMasker(env, get_action_mask)
    env = DummyVecEnv([lambda: Monitor(env)])

    # Make agent
    agent = get_agent(args.agent_type, env)

    # Callback
    callback = TrainingLogger(log_interval=args.log_interval, save_dir=args.save_dir, eval_env=env)

    # Intrinsic reward
    irs = get_intrinsic_reward(args.intrinsic_reward, env)
    if irs:
        callback_intrinsic_reward = IntrinsicReward(irs, log_interval=args.log_interval)
        callback = [callback, callback_intrinsic_reward]

    # Training
    print_header(f"Starting training for {args.Ntrain} timesteps", color='cyan')
    agent.learn(total_timesteps=args.Ntrain, callback=callback)

    # Save model
    agent.save(os.path.join(args.save_dir, f"{args.agent_type}_trained_model"))
    print(f"Model saved to {args.save_dir}")

    # Evaluation
    print_header('Evaluating trained agent', color='cyan')
    train_results = evaluate_agent(agent, env, env.get_attr('train_eqns')[0], n_eval_episodes=10)
    test_results = evaluate_agent(agent, env, env.get_attr('test_eqns')[0], n_eval_episodes=10)

    print_eval_results(train_results, label='Train')
    print_eval_results(test_results, label='Test')

    # Extract info
    if type(callback) == list:
        callback = callback[0]
    results_train, results_test = callback.results_train, callback.results_test

    # Print results
    print_header(f"Final results", color='cyan')
    print_results_dict_as_df("Training Results", results_train)
    # print_results_dict_as_df("Testing Results", results_test)

    # Run time
    t2 = time.time()
    elapsed_time = t2 - t1
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # extract off max
    max_test_acc_one_shot = callback.max_test_acc_one_shot

    return train_results, test_results, max_test_acc_one_shot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-gnn1')
    parser.add_argument('--state_rep', type=str, default='graph_integer_2d', help='State representation/encoding')
    parser.add_argument('--Ntrain', type=int, default=10**3, help='Number of training steps')
    parser.add_argument('--intrinsic_reward', type=str, default='None', choices=['ICM', 'E3B', 'RIDE', 'None'], \
                         help='Type of intrinsic reward')
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), \
         default=True, help="Normalize rewards (True/False)")
    parser.add_argument('--log_interval', type=int, default=None, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='data/generalize', help='Directory to save the results')
    parser.add_argument('--verbose', type=int, default=0)

    # Generalization parameters
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--generalization', type=str, default='lexical',choices=['lexical', 'structural', 'shallow','deep'])

    args = parser.parse_args()
    
    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**4)   
        #args.log_interval = int(0.1 * args.Ntrain)
    
    # Set save_dir
    args.save_dir = os.path.join(args.save_dir, f"{args.generalization}/level{args.level}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Check for invalid (agent, state_rep pairs)
    if args.state_rep in ['graph_integer_1d', 'graph_integer_2d'] and args.agent_type not in ['ppo-gnn', 'ppo-gnn1']:
        raise ValueError(
        f"❌ ERROR: 'ppo-gnn' requires 'graph_integer_1d' or 'graph_integer_2d' as state_rep, "
        f"but got '{args.state_rep}'.\n"
        f"👉 Fix this by using 'state_rep=graph_integer_2d' in your config."
    )

    print(f"Results saved to {args.save_dir}")

        

    results_train, results_test, max_test_acc_one_shot = main(args)











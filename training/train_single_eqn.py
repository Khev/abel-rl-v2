import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch as th

from envs.single_eqn import singleEqn

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm


from utils.utils_train import get_agent, get_intrinsic_reward, get_device
from utils.custom_functions import operation_names
from utils.utils_general import print_parameters, print_header
from colorama import Fore, Style

device = get_device()

# For PPO-mask
def get_action_mask(env):
    return env.action_mask

def evaluate_trained_agent(agent, env, n_eval_episodes=1, deterministic=True):
    """Evaluates the trained agent on a set number of episodes."""
    episode_rewards = []
    for _ in range(n_eval_episodes):
        #obs, _ = env.reset()
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            next_state, reward, done, info = env.step(action)
            op, term = env.get_attr('actions')[0][int(action[0])]
            #print(f'{env.lhs} = {env.rhs} | reward = {reward:.3f} |  action = ({operation_names[op]}, {term})')
            print(f'{env.get_attr("lhs")[0]} = {env.get_attr("rhs")[0]} | reward = {float(reward):.3f} |  action = ({operation_names[op]}, {term})')
            total_reward += reward
        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    print(f"Evaluation over {n_eval_episodes} episodes: mean reward = {mean_reward:.2f}")
    return mean_reward


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
        self.early_stopping = True

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
            #print(Fore.GREEN + f'\nSolved {main_eqn} = 0 ==> {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)
            self.T_solve = self.num_timesteps

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

        # Evaluation at intervals
        # if self.eval_env and self.n_calls % self.eval_interval == 0:
        #     print("\nRunning evaluation...")
        #     mean_eval_reward = evaluate_trained_agent(self.model, self.eval_env, n_eval_episodes=1, deterministic=False)
        #     print(f"Evaluation Mean Reward: {mean_eval_reward:.2f}\n")

        # Save model checkpoint at intervals
        # if self.n_calls % self.eval_interval == 0:
        #     save_path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}.zip")
        #     self.model.save(save_path)
        #     print(f"Checkpoint saved: {save_path}")

        return True  # Continue training



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

    # Print out args
    print('\n')
    params = vars(args)
    print_parameters(params)

    # Make env
    env = singleEqn(main_eqn=args.main_eqn, normalize_rewards=args.normalize_rewards, state_rep=args.state_rep)
    if args.agent_type in ["ppo-mask",'ppo-cnn','ppo-gnn','ppo-gnn1']:
        env = ActionMasker(env, get_action_mask)
    env = DummyVecEnv([lambda: Monitor(env)])

    # Make agent
    agent = get_agent(args.agent_type, env)

    # Callback
    callback = TrainingLogger(log_interval=args.log_interval, eval_interval=10*args.log_interval, save_dir=args.save_dir, eval_env=env)

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
    #print_header('Evaluating trained agent', color='cyan')
    #evaluate_trained_agent(agent, env)

    # Extract info
    #T_solve, T_converge = callback.T_solve, callback.T_converge
    if type(callback) == list:
        T_solve, T_converge = callback[0].T_solve, None
    else:
        T_solve, T_converge = callback.T_solve, None


    return T_solve, T_converge


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='ppo-gnn',choices=['dqn','a2c','ppo','ppo-mask', 'ppo-cnn','ppo-gnn'])
    parser.add_argument('--main_eqn', type=str, default='a*x+b', help='Main equation to solv')
    parser.add_argument('--state_rep', type=str, default='graph_integer_2d', help='State representation/encoding')
    parser.add_argument('--Ntrain', type=int, default=10**3, help='Number of training steps')
    parser.add_argument('--intrinsic_reward', type=str, default='None', choices=['ICM', 'E3B', 'RIDE', 'None'], \
                         help='Type of intrinsic reward')
    parser.add_argument("--normalize_rewards", type=lambda v: v.lower() in ("yes", "true", "t", "1"), \
         default=True, help="Normalize rewards (True/False)")
    parser.add_argument('--log_interval', type=int, default=None, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='data/misc', help='Directory to save the results')
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    
    # Set default log_interval if not provided
    if args.log_interval is None:
        args.log_interval = min(int(0.1 * args.Ntrain), 10**5)
    
    # Set save_dir
    if args.save_dir is None:
        args.save_dir = f'data/{args.main_eqn}' if args.main_eqn is not None else f'data/general_eqn'
    os.makedirs(args.save_dir, exist_ok=True)

    # Check for invalid (agent, state_rep pairs)
    if args.state_rep in ['graph_integer_1d', 'graph_integer_2d'] and args.agent_type not in ['ppo-gnn', 'ppo-gnn1']:
        raise ValueError(
        f"❌ ERROR: 'ppo-gnn' requires 'graph_integer_1d' or 'graph_integer_2d' as state_rep, "
        f"but got '{args.state_rep}'.\n"
        f"👉 Fix this by using 'state_rep=graph_integer_2d' in your config."
    )
        

    T_solve, T_converge = main(args)











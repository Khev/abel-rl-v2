#!/usr/bin/env python3
"""
Train MaskablePPO on multiEqn with a simple timestamp‑logger and
printable CLI arguments.  Compares n_envs = 1 vs 8.
"""

import os, sys, argparse, datetime
import numpy as np
import pandas as pd
import torch, gymnasium as gym
from torch import as_tensor
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from envs.multi_eqn_curriculum import multiEqn
from envs.multi_eqn_develop import multiEqn
from utils.utils_env import mask_fn
from utils.utils_general import print_parameters, print_header

from rllte.xplore.reward import ICM

import warnings, re
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=re.escape("builtin type SwigPy"),   # narrows the filter
    module="importlib._bootstrap"
)

# ────────────────────────────────────────────────────────────────
# suppress noisy SymPy / SWIG deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=re.escape("builtin type SwigPy"),
    module="importlib._bootstrap",
)

# Filter out specific DeprecationWarnings related to SWIG types
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning
)


# ────────────────────────────────────────────────────────────────
# tiny helper: time‑stamped console prin
# ────────────────────────────────────────────────────────────────
def log(msg: str, color: str = None) -> None:
    """Log a time-stamped message to the console, optionally with color."""
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    if color == "cyan":
        msg = f"\033[36m{msg}\033[0m"
    print(f"[{stamp}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────
# evaluation helper
# ────────────────────────────────────────────────────────────────
def evaluate_agent(agent, env, equation_list, n_eval_episodes=1):
    results = {}
    for eqn in equation_list:
        successes = 0
        for _ in range(n_eval_episodes):
            obs = env.reset()
            env.env_method("set_equation", eqn)
            done = [False]
            while not done[0]:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                if info[0].get("is_solved", False):
                    successes += 1
                    break
        results[eqn] = successes / n_eval_episodes
    return results


# ────────────────────────────────────────────────────────────────
# callback
# ────────────────────────────────────────────────────────────────
class TrainingLogger(BaseCallback):
    def __init__(self, log_interval, eval_interval, save_dir, eval_env, algo_name, verbose=1):
        super().__init__(verbose)
        self.log_interval  = log_interval
        self.eval_interval = eval_interval
        self.save_dir      = save_dir
        self.eval_env      = eval_env
        self.algo_name     = algo_name

        # state
        self.eqns_solved = set()
        self.coverage    = []
        self.logged_steps = []
        self.train_acc   = []
        self.test_acc    = []

        os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------------
    def _on_training_start(self):
        self.train_eqns = self.training_env.get_attr("train_eqns")[0]
        self.test_eqns  = self.training_env.get_attr("test_eqns")[0]

    # ----------------------------------------------------------
    def _on_step(self):
        for info in self.locals["infos"]:
            if info.get("is_solved"):
                if info.get("main_eqn") not in self.eqns_solved:
                    main_eqn, lhs, rhs = info.get("main_eqn"), info.get("lhs"), info.get("rhs")
                    print(f"\033[33mSolved {main_eqn} ==> {lhs} = {rhs} at Nstep = {self.n_calls} \033[0m")
                self.eqns_solved.add(info.get("main_eqn"))

        if self.eval_env and self.n_calls % self.eval_interval == 0:
            train_res = evaluate_agent(self.model, self.eval_env, self.train_eqns)
            test_res  = evaluate_agent(self.model, self.eval_env, self.test_eqns)

            self.logged_steps.append(self.n_calls)
            self.coverage.append(len(self.eqns_solved) / len(self.train_eqns))
            self.train_acc.append(np.mean(list(train_res.values())))
            self.test_acc.append(np.mean(list(test_res.values())))

            log(f"[{self.algo_name}] step{self.n_calls:6d}| "
                f"cov {self.coverage[-1]:.2f} | "
                f"train {self.train_acc[-1]:.2f} | "
                f"test {self.test_acc[-1]:.2f}")
        return True

    # ----------------------------------------------------------
    def _on_training_end(self):
        curves = pd.DataFrame(
            dict(step=self.logged_steps,
                 coverage=self.coverage,
                 train_acc=self.train_acc,
                 test_acc=self.test_acc)
        )
        csv_path = os.path.join(self.save_dir, "learning_curves.csv")
        curves.to_csv(csv_path, index=False)
        log(f"Saved learning curves → {csv_path}")


class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = as_tensor(self.locals["actions"], device=device)
        rewards = as_tensor(self.locals["rewards"], device=device)
        dones = as_tensor(self.locals["dones"], device=device)
        next_observations = as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = as_tensor(self.locals["new_obs"])
        actions = as_tensor(self.buffer.actions)
        rewards = as_tensor(self.buffer.rewards)
        dones = as_tensor(self.buffer.episode_starts)
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #


# ────────────────────────────────────────────────────────────────
# env factory
# ────────────────────────────────────────────────────────────────
def make_env(use_curriculum, gen, rank, seed=0):
    def _init():
        env = multiEqn(generalization=gen,state_rep="integer_1d",normalize_rewards=True,use_curriculum=use_curriculum)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init


# ────────────────────────────────────────────────────────────────
# main training routine
# ────────────────────────────────────────────────────────────────
def main(args):
    # Print out args
    print('\n')
    params = vars(args)
    print_parameters(params)

    n_envs   = args.n_envs
    seed     = args.seed
    gen      = args.gen
    Ntrain   = args.Ntrain
    save_dir = args.save_dir
    algo     = f"ppo-nenvs-{n_envs}"
    use_curriculum = args.use_curriculum == 'True'
    net_arch = [args.hidden_dim] * args.n_layers

    vec_env  = SubprocVecEnv([make_env(use_curriculum, gen, i, seed) for i in range(n_envs)], start_method="spawn")
    eval_env = DummyVecEnv([make_env(use_curriculum, gen, 999, seed)])

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(net_arch=net_arch),
        n_steps=2048,
        batch_size=512,
        n_epochs=4,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        tensorboard_log=f".tensorboard/tensorboard_masked_n{n_envs}",
        seed=seed
    )

    # Poesia tiny... shud be more stable... shud try on poesia-med too and compare...
    # model = MaskablePPO(
    #     "MlpPolicy", vec_env,
    #     n_steps=2048,
    #     batch_size=1024,
    #     n_epochs=8,
    #     learning_rate=lambda f: 3e-4 * (1 - f),
    #     ent_coef=0.005,
    #     clip_range=0.1,
    #     target_kl=0.02,
    #     gae_lambda=0.98,
    #     max_grad_norm=0.5,
    #     tensorboard_log="./tb",
    #     seed=seed
    # )

    callback_logger = TrainingLogger(
        # log_interval=Ntrain // 1000 // n_envs,
        # eval_interval=Ntrain // 1000 // n_envs,
        log_interval= 10**4,
        eval_interval= 10**4,
        save_dir=save_dir,
        eval_env=eval_env,
        algo_name=algo
    )

    callback = [callback_logger]

    if args.use_curiosity == 'True':
        device = 'cpu'
        irs = ICM(vec_env, device=device)
        callback_ir = RLeXploreWithOnPolicyRL(irs)
        callback.append(callback_ir)

    log("Starting training …")
    model.learn(total_timesteps=Ntrain, callback=callback)
    log("Training finished.")

    vec_env.close()
    eval_env.close()

    callback = callback[0]
    cov  = callback.coverage[-1]   if callback.coverage else 0.0
    tr_a = callback.train_acc[-1] if callback.train_acc else 0.0
    te_a = callback.test_acc[-1]  if callback.test_acc  else 0.0
    return cov, tr_a, te_a


# ────────────────────────────────────────────────────────────────
# entry point
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--Ntrain",   type=int, default=10**7, help="total PPO timesteps")
    p.add_argument("--n_envs",   type=int, default=2,     help="number of parallel environments")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--gen",      type=str, default="abel-small")
    p.add_argument("--save_dir", type=str, default="data/vectorized_env/")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_layers",   type=int, default=2)
    p.add_argument("--use_curriculum",   type=str,  default='True')
    p.add_argument("--use_curiosity",   type=str,  default='True')
    args = p.parse_args()

    if args.gen == 'abel-small':
        args.hidden_dim = 1024

    # benchmark 1‑env vs 8‑env
    baseline = args.Ntrain
    results  = {}
    for n in [8]:
        run_args = argparse.Namespace(**vars(args))
        run_args.n_envs   = n
        run_args.Ntrain   = baseline * n          
        run_args.save_dir = f"{args.save_dir.rstrip('/')}/n_envs_{n}"
        cov, tr, te = main(run_args)
        results[n] = (cov, tr, te)

    # summary
    print("\n" + "="*46)
    print("  Parallel‑env benchmark (after training)")
    print("="*46)
    print(f"{'n_envs':>6} | {'coverage':>9} | {'train_acc':>10} | {'test_acc':>9}")
    print("-"*46)
    for n, (cov, tr, te) in results.items():
        print(f"{n:6d} | {cov:9.3f} | {tr:10.2f} | {te:9.2f}")
    print("="*46)

    # Make figure
    df1 = pd.read_csv('data/vectorized_env/n_envs_1/learning_curves.csv')
    df8 = pd.read_csv('data/vectorized_env/n_envs_8/learning_curves.csv')

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # Coverage plot
    axs[0].plot(df1['step'], df1['coverage'], label='n_envs=1', marker='o')
    axs[0].plot(df8['step'], df8['coverage'], label='n_envs=8', marker='o')
    axs[0].set_ylabel('Coverage')
    axs[0].legend()
    axs[0].grid(True)

    # Train accuracy plot
    axs[1].plot(df1['step'], df1['train_acc'], label='n_envs=1', marker='o')
    axs[1].plot(df8['step'], df8['train_acc'], label='n_envs=8', marker='o')
    axs[1].set_ylabel('Train Accuracy (%)')
    axs[1].legend()
    axs[1].grid(True)

    # Test accuracy plot
    axs[2].plot(df1['step'], df1['test_acc'], label='n_envs=1', marker='o')
    axs[2].plot(df8['step'], df8['test_acc'], label='n_envs=8', marker='o')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('Test Accuracy (%)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/learning_curves.png')
    plt.show()

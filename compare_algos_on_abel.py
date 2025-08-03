#!/usr/bin/env python3
"""
Train MaskablePPO on multiEqn with a simple timestamp-logger and
printable CLI arguments.  Now loops over:
  (algo='ppo',     use_curiosity='False'),
  (algo='ppo',     use_curiosity='True'),
  (algo='ppo-gnn', use_curiosity='True')
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
from utils.utils_train import get_agent

from rllte.xplore.reward import ICM

import warnings, re
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=re.escape("builtin type SwigPy"),
    module="importlib._bootstrap"
)
warnings.filterwarnings("ignore", message="builtin type SwigPyPacked has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type SwigPyObject has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type swigvarlink has no __module__ attribute", category=DeprecationWarning)


# ────────────────────────────────────────────────────────────────
# tiny helper: time-stamped console print
# ────────────────────────────────────────────────────────────────
def log(msg: str, color: str = None) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    if color == "cyan":
        msg = f"\033[36m{msg}\033[0m"
    print(f"[{stamp}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────
# evaluation helpers
# ────────────────────────────────────────────────────────────────
def evaluate_agent(agent, env, equation_list, n_eval_episodes=1):
    """Greedy evaluation: deterministic=True. Returns dict eqn -> success rate (0..1)."""
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


def eval_success_at_n(agent, env, equations, n_trials=10, max_steps=30):
    """
    Stochastic evaluation: sample actions (deterministic=False) N times.
    Count equation as solved if any trial succeeds.
    Returns:
      - results: dict eqn -> details
      - agg:     dict with 'success_rate' across equations
    """
    results = {}
    for eqn in equations:
        solved = 0
        steps_when_solved = []
        for _ in range(n_trials):
            obs = env.reset()
            env.env_method("set_equation", eqn)
            done = [False]
            steps = 0
            while not done[0] and steps < max_steps:
                action, _ = agent.predict(obs, deterministic=False)  # sample under masks
                obs, _, done, info = env.step(action)
                steps += 1
                if info[0].get("is_solved", False):
                    solved += 1
                    steps_when_solved.append(steps)
                    break
        succ = 1.0 if solved > 0 else 0.0
        results[eqn] = dict(
            success_at_N=succ,
            trials_solved=solved,
            n_trials=n_trials,
            mean_steps=float(np.mean(steps_when_solved)) if steps_when_solved else None,
            median_steps=float(np.median(steps_when_solved)) if steps_when_solved else None,
        )
    success_rate = float(np.mean([v["success_at_N"] for v in results.values()])) if results else 0.0
    return results, dict(success_rate=success_rate)


# ────────────────────────────────────────────────────────────────
# callbacks
# ────────────────────────────────────────────────────────────────
class TrainingLogger(BaseCallback):
    def __init__(self, log_interval, eval_interval, save_dir, eval_env, algo_name, verbose=1):
        super().__init__(verbose)
        self.log_interval  = log_interval
        self.eval_interval = eval_interval
        self.save_dir      = save_dir
        self.eval_env      = eval_env
        self.algo_name     = algo_name

        self.eqns_solved   = set()
        self.coverage      = []
        self.logged_steps  = []
        self.train_acc     = []
        self.test_acc      = []   # greedy
        self.test_acc_topn = []   # Success@10

        os.makedirs(save_dir, exist_ok=True)

    def _on_training_start(self):
        self.train_eqns = self.training_env.get_attr("train_eqns")[0]
        self.test_eqns  = self.training_env.get_attr("test_eqns")[0]

    def _on_step(self):
        for info in self.locals["infos"]:
            if info.get("is_solved"):
                if info.get("main_eqn") not in self.eqns_solved:
                    main_eqn, lhs, rhs = info.get("main_eqn"), info.get("lhs"), info.get("rhs")
                    print(f"\033[33mSolved {main_eqn} ==> {lhs} = {rhs} at Nstep = {self.n_calls} \033[0m")
                self.eqns_solved.add(info.get("main_eqn"))

        if self.eval_env and self.n_calls % self.eval_interval == 0:
            train_res      = evaluate_agent(self.model, self.eval_env, self.train_eqns)
            test_res_greedy = evaluate_agent(self.model, self.eval_env, self.test_eqns)
            _, agg_topn     = eval_success_at_n(self.model, self.eval_env, self.test_eqns, n_trials=10, max_steps=30)

            self.logged_steps.append(self.n_calls)
            self.coverage.append(len(self.eqns_solved) / len(self.train_eqns))
            self.train_acc.append(np.mean(list(train_res.values())))
            self.test_acc.append(np.mean(list(test_res_greedy.values())))
            self.test_acc_topn.append(agg_topn["success_rate"])

            log(f"[{self.algo_name}] t={self.n_calls:6d}| "
                f"cov {self.coverage[-1]:.2f} | "
                f"train {self.train_acc[-1]:.2f} | "
                f"test(greedy) {self.test_acc[-1]:.2f} | "
                f"test@10 {self.test_acc_topn[-1]:.2f}")
        return True

    def _on_training_end(self):
        curves = pd.DataFrame(
            dict(step=self.logged_steps,
                 coverage=self.coverage,
                 train_acc=self.train_acc,
                 test_acc=self.test_acc,
                 test_acc_topn=self.test_acc_topn)
        )
        csv_path = os.path.join(self.save_dir, "learning_curves.csv")
        curves.to_csv(csv_path, index=False)
        log(f"Saved learning curves → {csv_path}")


class RLeXploreWithOnPolicyRL(BaseCallback):
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = as_tensor(self.locals["actions"], device=device)
        rewards = as_tensor(self.locals["rewards"], device=device)
        dones = as_tensor(self.locals["dones"], device=device)
        next_observations = as_tensor(self.locals["new_obs"], device=device)
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        return True

    def _on_rollout_end(self) -> None:
        obs = as_tensor(self.buffer.observations)
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = as_tensor(self.locals["new_obs"])
        actions = as_tensor(self.buffer.actions)
        rewards = as_tensor(self.buffer.rewards)
        dones = as_tensor(self.buffer.episode_starts)
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions,
                         rewards=rewards, terminateds=dones,
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns    += intrinsic_rewards.cpu().numpy()


# ────────────────────────────────────────────────────────────────
# env factory
# ────────────────────────────────────────────────────────────────
def make_env(use_curriculum, gen, rank, algo, seed=0):
    def _init():
        if algo == 'ppo-gnn':
            state_rep_temp = 'graph_integer_2d'
        else:
            state_rep_temp = 'integer_1d'   # fixed (was accidentally 'use_curriculum')
        env = multiEqn(generalization=gen, state_rep=state_rep_temp,
                       normalize_rewards=True, use_curriculum=use_curriculum)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init


# ────────────────────────────────────────────────────────────────
# main training routine
# ────────────────────────────────────────────────────────────────
def main(args):
    print('\n')
    params = vars(args)
    print_parameters(params)

    n_envs   = args.n_envs
    seed     = args.seed
    gen      = args.gen
    Ntrain   = args.Ntrain
    save_dir = args.save_dir
    algo     = args.algo
    use_curriculum = args.use_curriculum == 'True'
    net_arch = [args.hidden_dim] * args.n_layers

    vec_env  = SubprocVecEnv([make_env(use_curriculum, gen, i, algo, seed) for i in range(n_envs)], start_method="spawn")
    eval_env = DummyVecEnv([make_env(use_curriculum, gen, 999, algo, seed)])

    if algo == 'ppo-gnn':
        model = get_agent('ppo-gnn', vec_env)
        # NOTE: curiosity now follows args.use_curiosity (no forced override)
    else:
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

    callback_logger = TrainingLogger(
        log_interval=10**4,
        eval_interval=10**4,
        save_dir=save_dir,
        eval_env=eval_env,
        algo_name=f"{algo}-nenvs-{n_envs}"
    )

    callbacks = [callback_logger]

    if args.use_curiosity == 'True':
        device = 'cpu'
        irs = ICM(vec_env, device=device)
        callbacks.append(RLeXploreWithOnPolicyRL(irs))

    log("Starting training …")
    model.learn(total_timesteps=Ntrain, callback=callbacks)
    log("Training finished.")

    vec_env.close()
    eval_env.close()

    cb = callback_logger
    cov  = cb.coverage[-1]       if cb.coverage else 0.0
    tr_a = cb.train_acc[-1]      if cb.train_acc else 0.0
    te_g = cb.test_acc[-1]       if cb.test_acc else 0.0
    te_n = cb.test_acc_topn[-1]  if cb.test_acc_topn else 0.0
    return cov, tr_a, te_g, te_n


# ────────────────────────────────────────────────────────────────
# entry point
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo",              type=str, default='ppo')
    p.add_argument("--Ntrain",            type=int, default=3*10**6, help="total PPO timesteps")
    p.add_argument("--n_envs",            type=int, default=8,     help="number of parallel environments")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--gen",               type=str, default="abel-small")
    p.add_argument("--save_dir",          type=str, default="data/vectorized_env/")
    p.add_argument("--hidden_dim",        type=int, default=512)
    p.add_argument("--n_layers",          type=int, default=2)
    p.add_argument("--use_curriculum",    type=str,  default='True')
    p.add_argument("--use_curiosity",     type=str,  default='True')
    args = p.parse_args()

    if args.gen == 'abel-small':
        args.hidden_dim = 1024

    # Scenarios to run
    scenarios = [
        ('ppo',     'False'),
        ('ppo',     'True'),
        ('ppo-gnn', 'False'),
    ]

    baseline = args.Ntrain
    results  = {}

    for algo_name, use_cur in scenarios:
        for n in [8]:  # keep as before; extend if you want {1,8}
            run_args = argparse.Namespace(**vars(args))
            run_args.algo = algo_name
            run_args.use_curiosity = use_cur
            run_args.n_envs = n
            run_args.Ntrain = baseline * n

            # put results in separate folders to avoid overwrite
            run_args.save_dir = (
                f"{args.save_dir.rstrip('/')}/algo_{algo_name}_cur_{use_cur}/n_envs_{n}"
            )

            cov, tr, te_g, te_n = main(run_args)
            results[(algo_name, use_cur, n)] = (cov, tr, te_g, te_n)

    # Summary
    print("\n" + "="*92)
    print("  Benchmark summary (after training)")
    print("="*92)
    print(f"{'algo':>8} | {'curiosity':>10} | {'n_envs':>6} | {'coverage':>9} | {'train_acc':>10} | {'test_greedy':>12} | {'test@10':>8}")
    print("-"*92)
    for (algo_name, use_cur, n), (cov, tr, te_g, te_n) in results.items():
        print(f"{algo_name:>8} | {use_cur:>10} | {n:6d} | {cov:9.3f} | {tr:10.2f} | {te_g:12.2f} | {te_n:8.2f}")
    print("="*92)

    # Also print the tuple you asked for, per scenario
    for (algo_name, use_cur, n), (cov, tr, te_g, te_n) in results.items():
        print(f"{algo_name}/{use_cur}/n={n}: (cov, acc_train, acc_test_greedy, acc_test_topN10) = ({cov:.3f}, {tr:.2f}, {te_g:.2f}, {te_n:.2f})")

    # ────────────────────────────────────────────────────────────────
    # Combined figure for the three scenarios
    # ────────────────────────────────────────────────────────────────
    def scenario_label(algo_name, use_cur):
        cur = "curiosity" if use_cur == "True" else "no-curiosity"
        return f"{algo_name} / {cur}"

    summary_fig_path = f"{args.save_dir.rstrip('/')}/summary_learning_curves.png"

    dfs = []  # list of (label, df)
    missing = []

    for algo_name, use_cur in scenarios:
        # We ran with n_envs=1 above; change if you extend to other values.
        csv_path = f"{args.save_dir.rstrip('/')}/algo_{algo_name}_cur_{use_cur}/n_envs_1/learning_curves.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append((scenario_label(algo_name, use_cur), df))
        else:
            missing.append(csv_path)

    if missing:
        log("Warning: missing learning_curves.csv for some scenarios:")
        for p in missing:
            log(f"  - {p}")

    if dfs:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Coverage
        for label, df in dfs:
            axs[0].plot(df['step'], df['coverage'], marker='o', linewidth=1.5, label=label)
        axs[0].set_ylabel('Coverage')
        axs[0].grid(True)
        axs[0].legend()

        # Train accuracy
        for label, df in dfs:
            axs[1].plot(df['step'], df['train_acc'], marker='o', linewidth=1.5, label=label)
        axs[1].set_ylabel('Train Accuracy')
        axs[1].grid(True)
        axs[1].legend()

        # Test accuracy: greedy (solid) + Success@10 (dashed) if present
        for label, df in dfs:
            axs[2].plot(df['step'], df['test_acc'], marker='o', linewidth=1.5, label=f"{label} (greedy)")
            if 'test_acc_topn' in df.columns:
                axs[2].plot(df['step'], df['test_acc_topn'], linestyle='--', linewidth=1.5, label=f"{label} (@10)")
        axs[2].set_xlabel('Steps')
        axs[2].set_ylabel('Test Accuracy')
        axs[2].grid(True)
        axs[2].legend(ncol=2)

        plt.tight_layout()
        plt.savefig(summary_fig_path, dpi=150)
        plt.show()
        log(f"Saved combined figure → {summary_fig_path}")
    else:
        log("No learning_curves.csv files found; skipping combined figure.")



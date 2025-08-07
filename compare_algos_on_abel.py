#!/usr/bin/env python3
"""
Benchmark MaskablePPO variants on multiEqn with timestamp logging.

Scenarios run:
  (algo='ppo',     use_curiosity='False')
  (algo='ppo',     use_curiosity='True')
  (algo='ppo-gnn', use_curiosity='False')

Key features:
- Multiple trials per scenario (different seeds): --trials N
- Keeps total experience *constant* across n_envs (do NOT multiply Ntrain by n_envs)
- Per-trial output folders to avoid overwrite
- Aggregated summary (mean ± std) across trials
- Combined figure showing mean curves with ±1σ bands for each scenario
- ALSO logs train Success@10 (stochastic) alongside test Success@10

Notes:
- Choose ONE multiEqn import below that exists in your repo.
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

# ⚠️ Keep only the one you actually have:
#from envs.multi_eqn_develop import multiEqn
from envs.multi_eqn_lookahead import multiEqn

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
    """Sampling evaluation: counts solved if any of N trials succeeds."""
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
                action, _ = agent.predict(obs, deterministic=False)
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
        # Note: we return 1.0/0.0 success_at_N per equation; aggregate outside
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
        self.train_acc_topn = []  # ADDED: train Success@10
        self.test_acc      = []   # greedy
        self.test_acc_topn = []   # Success@10

        self.stagnation_threshold = 2000000  # 3 million steps

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
            train_res        = evaluate_agent(self.model, self.eval_env, self.train_eqns)
            test_res_greedy  = evaluate_agent(self.model, self.eval_env, self.test_eqns)
            # ADDED: stochastic eval on TRAIN set
            _, agg_topn_train = eval_success_at_n(self.model, self.eval_env, self.train_eqns, n_trials=10, max_steps=30)
            # existing: stochastic eval on TEST set
            _, agg_topn       = eval_success_at_n(self.model, self.eval_env, self.test_eqns, n_trials=10, max_steps=30)

            self.logged_steps.append(self.n_calls)
            self.coverage.append(len(self.eqns_solved) / len(self.train_eqns))
            self.train_acc.append(np.mean(list(train_res.values())))
            self.test_acc.append(np.mean(list(test_res_greedy.values())))
            # ADDED: push train@10
            self.train_acc_topn.append(agg_topn_train["success_rate"])
            self.test_acc_topn.append(agg_topn["success_rate"])

            log(f"[{self.algo_name}] t={self.n_calls:6d}| "
                f"cov {self.coverage[-1]:.2f} | "
                f"train {self.train_acc[-1]:.2f} | "
                f"train@10 {self.train_acc_topn[-1]:.2f} | "
                f"test(greedy) {self.test_acc[-1]:.2f} | "
                f"test@10 {self.test_acc_topn[-1]:.2f}")

            # Early stopping check: count consecutive evals with no coverage change
            if len(self.coverage) > 1:
                no_change_count = 0
                for i in range(len(self.coverage)-2, -1, -1):
                    if self.coverage[i] == self.coverage[-1]:
                        no_change_count += 1
                    else:
                        break
                steps_stagnant = no_change_count * self.eval_interval
                if steps_stagnant >= self.stagnation_threshold:
                    log(f"Coverage stagnant for {steps_stagnant} steps ({self.coverage[-1]:.3f}); early stopping.")
                    return False  # Stop training

        return True

    def _on_training_end(self):
        curves = pd.DataFrame(
            dict(step=self.logged_steps,
                 coverage=self.coverage,
                 train_acc=self.train_acc,
                 train_acc_topn=self.train_acc_topn,  # ADDED
                 test_acc=self.test_acc,
                 test_acc_topn=self.test_acc_topn)
        )
        csv_path = os.path.join(self.save_dir, "learning_curves.csv")
        curves.to_csv(csv_path, index=False)
        log(f"Saved learning curves → {csv_path}")


class RLeXploreWithOnPolicyRL(BaseCallback):
    def __init__(self, irs, verbose=0):
        super().__init__(verbose)
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
def make_env(use_lookahead, use_memory, use_curriculum, curriculum_type, gen, rank, algo, seed=0):
    def _init():
        if algo == 'ppo-gnn':
            state_rep_temp = 'graph_integer_2d'
        else:
            state_rep_temp = 'integer_1d'
        env = multiEqn(generalization=gen, state_rep=state_rep_temp,
                       normalize_rewards=True, use_curriculum=use_curriculum, use_memory=use_memory,
                    use_lookahead=use_lookahead, curriculum_type=curriculum_type)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init


# ────────────────────────────────────────────────────────────────
# single training routine (one run)
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
    curriculum_type = args.curriculum_type
    use_memory = True if args.use_memory == 'True' else False
    use_lookahead = True if args.use_lookahead == 'True' else False
    net_arch = [args.hidden_dim] * args.n_layers

    vec_env  = SubprocVecEnv([make_env(use_lookahead, use_memory, use_curriculum, curriculum_type, gen, i, algo, seed) for i in range(n_envs)], start_method="spawn")
    eval_env = DummyVecEnv([make_env(use_lookahead, 'False', use_curriculum, curriculum_type, gen, 999, algo, seed)])

    if algo == 'ppo-gnn':
        model = get_agent('ppo-gnn', vec_env)
        # Curiosity controlled by args.use_curiosity (no forced override)
    else:
        model = MaskablePPO(
            "MlpPolicy", vec_env,
            policy_kwargs=dict(net_arch=net_arch),
            n_steps=2048,
            batch_size=1024,
            n_epochs=4,
            learning_rate=3e-4,
            ent_coef=0.01,
            gamma=0.99,
            tensorboard_log=f".tensorboard/tensorboard_masked_n{n_envs}",
            seed=seed
        )

    callback_logger = TrainingLogger(
        log_interval=10**5,
        eval_interval=10**5,
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
    cov  = cb.coverage[-1]         if cb.coverage else 0.0
    tr_a = cb.train_acc[-1]        if cb.train_acc else 0.0
    te_g = cb.test_acc[-1]         if cb.test_acc else 0.0
    te_n = cb.test_acc_topn[-1]    if cb.test_acc_topn else 0.0
    return cov, tr_a, te_g, te_n


# ────────────────────────────────────────────────────────────────
# utilities for aggregation & plotting
# ────────────────────────────────────────────────────────────────
def scenario_key(algo_name, use_cur):
    return f"algo_{algo_name}_cur_{use_cur}"

def scenario_label(algo_name, use_cur):
    cur = "curiosity" if use_cur == "True" else "no-curiosity"
    return f"{algo_name} / {cur}"

def load_and_aggregate_curves(root_dir_list):
    """
    Given a list of run dirs (one per trial) that each contain learning_curves.csv,
    return (mean_df, std_df) grouped by 'step'. Missing files are skipped.
    """
    dfs = []
    for rd in root_dir_list:
        csv = os.path.join(rd, "learning_curves.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df["trial_dir"] = rd
            dfs.append(df)
        else:
            log(f"Missing curves: {csv}")
    if not dfs:
        return None, None
    all_df = pd.concat(dfs, ignore_index=True)
    g = all_df.groupby("step", as_index=False)
    mean_df = g.mean(numeric_only=True)
    std_df  = g.std(numeric_only=True).fillna(0.0)
    return mean_df, std_df


# ────────────────────────────────────────────────────────────────
# entry point
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo",              type=str, default='ppo')
    p.add_argument("--Ntrain",            type=int, default=10**7, help="total PPO timesteps (do NOT multiply by n_envs)")
    p.add_argument("--n_envs",            type=int, default=4,     help="number of parallel environments")
    p.add_argument("--trials",            type=int, default=3,     help="number of trials per scenario (different seeds)")
    p.add_argument("--seed",              type=int, default=98112,    help="base seed; per-trial seed = base + 1000*trial_idx")
    p.add_argument("--gen",               type=str, default="abel-small")
    p.add_argument("--save_dir",          type=str, default="data/vectorized_env/")
    p.add_argument("--hidden_dim",        type=int, default=1024)
    p.add_argument("--n_layers",          type=int, default=2)
    p.add_argument("--use_curriculum",    type=str,  default='True')
    p.add_argument("--curriculum_type",   type=str,  default='inverse')
    p.add_argument("--use_curiosity",     type=str,  default='True')
    p.add_argument("--use_memory",        type=str,  default='True')
    p.add_argument("--use_lookahead",     type=str,  default='False')
    args = p.parse_args()

    # Scenarios to run (fixed set)
    scenarios = [
        ('ppo',     'True'),
        ('ppo',     'False')
    ]

    # IMPORTANT: keep total frames constant across n_envs
    baseline = args.Ntrain

    # For aggregation
    per_scenario_metrics = {}   # key: (algo_name,use_cur) -> list of (cov,tr,te_g,te_n)
    per_scenario_dirs    = {}   # key -> list of trial run dirs (for curves)

    for algo_name, use_cur in scenarios:
        metrics_list = []
        run_dirs = []

        for t in range(args.trials):
            run_args = argparse.Namespace(**vars(args))
            run_args.algo = algo_name
            run_args.use_curiosity = use_cur
            run_args.Ntrain = baseline*args.n_envs
            run_args.seed = args.seed + 1000 * t    # different seed per trial

            # Per-trial directory: save_dir/algo_X_cur_Y/n_envs_Z/trial_T
            scen = scenario_key(algo_name, use_cur)
            per_trial_dir = f"{args.save_dir.rstrip('/')}/{scen}/n_envs_{args.n_envs}/trial_{t+1}"
            os.makedirs(per_trial_dir, exist_ok=True)
            run_args.save_dir = per_trial_dir

            cov, tr, te_g, te_n = main(run_args)
            metrics_list.append((cov, tr, te_g, te_n))
            run_dirs.append(per_trial_dir)

        per_scenario_metrics[(algo_name, use_cur)] = metrics_list
        per_scenario_dirs[(algo_name, use_cur)] = run_dirs

    # ────────────────────────────────────────────────────────────────
    # Aggregated summary (mean ± std)
    # ────────────────────────────────────────────────────────────────
    print("\n" + "="*112)
    print("  Benchmark summary (after training) — aggregated over trials")
    print("="*112)
    hdr = f"{'algo':>8} | {'curiosity':>10} | {'n_envs':>6} | {'coverage':>22} | {'train_acc':>22} | {'test_greedy':>22} | {'test@10':>22}"
    print(hdr)
    print("-"*112)

    agg_rows = []
    for (algo_name, use_cur), metrics_list in per_scenario_metrics.items():
        arr = np.array(metrics_list)  # shape (trials, 4)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)

        def fmt(mu, sd):  # fixed width "m ± s"
            return f"{mu:6.3f} ± {sd:6.3f}"

        cov_mu, tr_mu, tg_mu, t10_mu = means
        cov_sd, tr_sd, tg_sd, t10_sd = stds

        print(f"{algo_name:>8} | {use_cur:>10} | {args.n_envs:6d} | "
              f"{fmt(cov_mu, cov_sd):>22} | {fmt(tr_mu, tr_sd):>22} | {fmt(tg_mu, tg_sd):>22} | {fmt(t10_mu, t10_sd):>22}")

        agg_rows.append(dict(
            algo=algo_name, curiosity=use_cur, n_envs=args.n_envs,
            coverage_mean=cov_mu, coverage_std=cov_sd,
            train_acc_mean=tr_mu, train_acc_std=tr_sd,
            test_greedy_mean=tg_mu, test_greedy_std=tg_sd,
            test_at_10_mean=t10_mu, test_at_10_std=t10_sd
        ))
    print("="*112)

    # Save aggregated CSV
    out_csv = os.path.join(args.save_dir, f"benchmark_summary_nenvs_{args.n_envs}.csv")
    os.makedirs(args.save_dir, exist_ok=True)
    pd.DataFrame(agg_rows).to_csv(out_csv, index=False)
    log(f"Saved aggregated summary → {out_csv}")

    # ────────────────────────────────────────────────────────────────
    # Combined figure: mean curves with ±1σ shading, per scenario
    # ────────────────────────────────────────────────────────────────
    summary_fig_path = os.path.join(args.save_dir, f"summary_learning_curves_nenvs_{args.n_envs}.png")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for (algo_name, use_cur), run_dirs in per_scenario_dirs.items():
        label = scenario_label(algo_name, use_cur)
        mean_df, std_df = load_and_aggregate_curves(run_dirs)
        if mean_df is None:
            log(f"No curves to aggregate for {label}")
            continue

        # Coverage
        axs[0].plot(mean_df['step'], mean_df['coverage'], label=label)
        axs[0].fill_between(
            mean_df['step'],
            mean_df['coverage'] - std_df['coverage'],
            mean_df['coverage'] + std_df['coverage'],
            alpha=0.2
        )

        # Train accuracy (greedy) + train@10 if present
        axs[1].plot(mean_df['step'], mean_df['train_acc'], label=label)
        axs[1].fill_between(
            mean_df['step'],
            mean_df['train_acc'] - std_df['train_acc'],
            mean_df['train_acc'] + std_df['train_acc'],
            alpha=0.2
        )
        if 'train_acc_topn' in mean_df.columns and 'train_acc_topn' in std_df.columns:
            axs[1].plot(mean_df['step'], mean_df['train_acc_topn'], linestyle='--', label=f"{label} (train@10)")
            axs[1].fill_between(
                mean_df['step'],
                mean_df['train_acc_topn'] - std_df['train_acc_topn'],
                mean_df['train_acc_topn'] + std_df['train_acc_topn'],
                alpha=0.2
            )

        # Test accuracy (greedy) + test@10 if present
        axs[2].plot(mean_df['step'], mean_df['test_acc'], label=f"{label} (greedy)")
        axs[2].fill_between(
            mean_df['step'],
            mean_df['test_acc'] - std_df['test_acc'],
            mean_df['test_acc'] + std_df['test_acc'],
            alpha=0.2
        )
        if 'test_acc_topn' in mean_df.columns and 'test_acc_topn' in std_df.columns:
            axs[2].plot(mean_df['step'], mean_df['test_acc_topn'], linestyle='--', label=f"{label} (@10)")
            axs[2].fill_between(
                mean_df['step'],
                mean_df['test_acc_topn'] - std_df['test_acc_topn'],
                mean_df['test_acc_topn'] + std_df['test_acc_topn'],
                alpha=0.2
            )

    axs[0].set_ylabel('Coverage');       axs[0].grid(True); axs[0].legend(ncol=2)
    axs[1].set_ylabel('Train Accuracy'); axs[1].grid(True); axs[1].legend(ncol=2)
    axs[2].set_xlabel('Steps');          axs[2].set_ylabel('Test Accuracy'); axs[2].grid(True); axs[2].legend(ncol=2)

    plt.tight_layout()
    plt.savefig(summary_fig_path, dpi=150)
    plt.show()
    log(f"Saved combined figure → {summary_fig_path}")
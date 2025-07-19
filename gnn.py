#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# debug_gnn_equations.py
#   Minimal end-to-end loop:
#     1. build env
#     2. build GNN policy (PyG)
#     3. train for N steps
#     4. evaluate + pretty print
# ────────────────────────────────────────────────────────────────
import os, random, argparse, time, copy, numpy as np
from pathlib import Path


# ╭────────────────────────────────────────────────────────────────╮
# │ 0)  third-party deps                                          │
# ╰────────────────────────────────────────────────────────────────╯
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool           # ⚠️  PyG
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

# ╭────────────────────────────────────────────────────────────────╮
# │ 1)  your equation env                                         │
# ╰────────────────────────────────────────────────────────────────╯
# NOTE: adjust the import to where *your* env lives.
from envs.single_eqn import singleEqn

from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import pandas as pd
import numpy as np

def build_tsolve_dataframe(results, algos, eqns):
    """
    results[algo][eqn] -> list of Tsolve ( -1 == unsolved )
    Returns:
        df_num  : numeric stats (means, stds, counts)
        df_disp : pretty formatted strings for human reading
    """
    rows = []
    for eqn in eqns:
        row = {"eqn": eqn}
        for algo in algos:
            ts = results[algo][eqn]
            n_trials = len(ts)
            solved = [t for t in ts if t != -1]
            n_solved = len(solved)
            if n_solved:
                m = float(np.mean(solved))
                s = float(np.std(solved))
            else:
                m = np.nan
                s = np.nan
            row[f"{algo}_mean"] = m
            row[f"{algo}_std"]  = s
            row[f"{algo}_n"]    = n_solved
            row[f"{algo}_N"]    = n_trials
        rows.append(row)

    df_num = pd.DataFrame(rows).set_index("eqn")

    # Per-eqn mean ratio (ppo / ppo-gnn) if both present
    if "ppo" in algos and "ppo-gnn" in algos:
        df_num["gnn/ppo"] = df_num["ppo-gnn_mean"] / df_num["ppo_mean"]

    # ---- overall row ----
    overall = {"eqn": "OVERALL"}
    for algo in algos:
        # gather all solved across eqns
        all_solved = []
        N_total = 0
        for eqn in eqns:
            ts = results[algo][eqn]
            N_total += len(ts)
            all_solved.extend([t for t in ts if t != -1])
        n_solved = len(all_solved)
        if n_solved:
            m = float(np.mean(all_solved))
            s = float(np.std(all_solved))
        else:
            m = np.nan
            s = np.nan
        overall[f"{algo}_mean"] = m
        overall[f"{algo}_std"]  = s
        overall[f"{algo}_n"]    = n_solved
        overall[f"{algo}_N"]    = N_total
    if "ppo" in algos and "ppo-gnn" in algos:
        overall["gnn/ppo"] = overall["ppo-gnn_mean"] / overall["ppo_mean"]
    df_num = pd.concat([df_num, pd.DataFrame([overall]).set_index("eqn")], axis=0)

    # ---- human-readable display table ----
    def _fmt(m, s, n, N):
        if not np.isfinite(m):
            return f"— (0/{N})"
        return f"{m:.1f} ± {s:.1f} (n={n}/{N})"

    df_disp = pd.DataFrame(index=df_num.index)
    if "ppo" in algos:
        df_disp["PPO"] = [
            _fmt(df_num.loc[e, "ppo_mean"],
                 df_num.loc[e, "ppo_std"],
                 int(df_num.loc[e, "ppo_n"]),
                 int(df_num.loc[e, "ppo_N"]))
            for e in df_num.index
        ]
    if "ppo-gnn" in algos:
        df_disp["PPO-GNN"] = [
            _fmt(df_num.loc[e, "ppo-gnn_mean"],
                 df_num.loc[e, "ppo-gnn_std"],
                 int(df_num.loc[e, "ppo-gnn_n"]),
                 int(df_num.loc[e, "ppo-gnn_N"]))
            for e in df_num.index
        ]
    if "ppo" in algos and "ppo-gnn" in algos:
        df_disp["<GNN>/<PPO>"] = [
            f"{df_num.loc[e, 'gnn/ppo']:.2f}"
            if np.isfinite(df_num.loc[e, "gnn/ppo"]) else "—"
            for e in df_num.index
        ]

    return df_num, df_disp


class SolveLogger(BaseCallback):
    """
    • Prints a one-liner whenever the env reports a solution.
    • Every `eval_freq` steps prints (min, mean, max) external reward
      over the most-recent window.
    """
    def __init__(self, eval_freq: int):
        super().__init__()
        self.eval_freq = eval_freq
        self._reward_window = []        # collects rewards since last print
        self.early_stopping = True
        self.Tsolve = -1

    # --------------------------------------------------------------
    def _on_step(self) -> bool:         # called after *every* env.step()
        # — Track external reward (assumes single-env DummyVecEnv) —
        self._reward_window.append(float(self.locals["rewards"][0]))

        # — Check for solves —
        for info in self.locals["infos"]:
            if info.get("is_solved", False):
                main_eqn = info.get("main_eqn", "?")
                lhs      = info.get("lhs", "?")
                rhs      = info.get("rhs", "?")
                print(f"✅ Solved {main_eqn}: {lhs} = {rhs} (t={self.num_timesteps})")
                if self.Tsolve == -1:
                    self.Tsolve = self.num_timesteps
                    if self.early_stopping:
                        return False

        # — Periodic reward stats —
        if self.num_timesteps % self.eval_freq == 0:
            if self._reward_window:     # avoid empty list first step
                w = np.array(self._reward_window)
                print(f"[t={self.num_timesteps}] "
                      f"(min, mean, max)_reward = ({w.min():+.2f}, {w.mean():+.2f}, {w.max():+.2f})")
                self._reward_window.clear()

        return True                     # keep training


def get_action_mask(env):
    """wrapper helper for sb3-contrib ActionMasker"""
    return env.action_mask


# ╭────────────────────────────────────────────────────────────────╮
# │ 2)  simple GCN feature extractor                              │
# ╰────────────────────────────────────────────────────────────────╯
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GCNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space["node_features"].shape[-1]
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.lin   = nn.Linear(128, features_dim)

    def forward(self, obs):
        x_b     = obs["node_features"].float()         # (B, N, F)
        ei_b    = obs["edge_index"].long()             # (B, 2, E)
        nmask_b = obs["node_mask"].bool()              # (B, N)
        emask_b = obs["edge_mask"].bool()              # (B, E)

        B, N, feat_dim = x_b.size()                    # ← renamed
        device  = x_b.device

        xs, edge_index, batch = [], [], []
        node_off = 0
        for i in range(B):
            valid_nodes = nmask_b[i]
            valid_edges = emask_b[i]

            x_i   = x_b[i][valid_nodes]                # (Ni, feat_dim)
            ei_i  = ei_b[i][:, valid_edges]            # (2, Ei)
            keep  = valid_nodes[ei_i[0]] & valid_nodes[ei_i[1]]
            ei_i  = ei_i[:, keep]

            ei_i += node_off
            xs.append(x_i)
            edge_index.append(ei_i)
            batch.append(torch.full((x_i.size(0),), i, dtype=torch.long))
            node_off += x_i.size(0)

        x     = torch.cat(xs,  dim=0).to(device)       # (ΣNi, feat_dim)
        ei    = torch.cat(edge_index, dim=1).to(device)
        batch = torch.cat(batch).to(device)

        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = global_mean_pool(x, batch)
        return self.lin(x)



class GNNPolicy(MaskableActorCriticPolicy):
    """
    SB3 policy that drops in the extractor above.
    """
    def __init__(self, *args, **kw):
        super().__init__(
            *args,
            features_extractor_class = GCNExtractor,
            features_extractor_kwargs= dict(features_dim=128),
            **kw
        )

# ╭────────────────────────────────────────────────────────────────╮
# │ 3)  helper: evaluation loop                                   │
# ╰────────────────────────────────────────────────────────────────╯
def eval_one_equation(model, env, n_episodes: int = 10) -> tuple[float, list[float]]:
    """
    Evaluate success % on the *current* equation configured in env,
    and collect total rewards per episode, with cleaner printing.
    """
    successes = 0
    all_episode_rewards = []
    current_equation = env.envs[0].main_eqn
    print(f"\n--- Evaluating Equation: {current_equation} ({n_episodes} episodes) ---")
    for i in range(n_episodes):
        obs = env.reset()
        current_episode_reward = 0
        done = [False]
        steps_taken = 0
        print(f"\n--- Episode {i + 1}/{n_episodes} ---")
        while not done[0]:
            act, _ = model.predict(obs, deterministic=False) # Keep deterministic=False as per user's last code
            obs, rew, done, info = env.step(act)
            steps_taken += 1
            current_episode_reward += rew[0]            
            print(f"{info[0]['lhs']} = {info[0]['rhs']} | {info[0]['action']} | {rew[0]:.2f}")
            if info[0].get("is_solved", False):
                successes += 1
                print(f"  ✅ Solved in {steps_taken} steps!")
                break
            elif done[0]: # Episode terminated for reasons other than solved (e.g., max steps, invalid eqn)
                break
        all_episode_rewards.append(current_episode_reward) # Store total reward for this episode
    
    return 100.0 * successes / n_episodes, all_episode_rewards

class Colors:
    CYAN = '\033[96m'
    ENDC = '\033[0m'

# ╭────────────────────────────────────────────────────────────────╮
# │ 4)  main                                                      │
# ╰────────────────────────────────────────────────────────────────╯
def main(args):
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

   # Print args prettily
    print(f"\n{Colors.CYAN}----")
    print("Args")
    print("----")
    for arg_name, arg_value in vars(args).items():
        print(f"  .{arg_name}: {arg_value}")
    print(f"{Colors.ENDC}\n")

    # build env (single-proc for simplicity)
    def _make_env():
        e = singleEqn(
            main_eqn=args.main_eqn, 
            state_rep="graph_integer_2d" if args.algo == 'ppo-gnn' else 'integer_1d',
            normalize_rewards=True
        )
        e = ActionMasker(e, get_action_mask)
        return e

    train_env = DummyVecEnv([_make_env])
    eval_env  = DummyVecEnv([_make_env])

    # create model
    model = MaskablePPO(
        policy          = GNNPolicy if args.algo == 'ppo-gnn' else 'MlpPolicy',
        env             = train_env,
        n_steps         = 2048,
        batch_size      = 512,
        n_epochs        = 4,
        learning_rate   = 3e-4,
        gamma           = 0.99,
        ent_coef        = 0.05,
        device          = "cuda" if torch.cuda.is_available() else "cpu",
    )

    # ---- train ----
    print(f"\nTraining for {args.timesteps:,} steps …")
    t0 = time.time()

    eval_freq = max(1000, int(0.1 * args.timesteps))   # every 10 % of total steps
    callback  = SolveLogger(eval_freq)
    model.learn(total_timesteps=args.timesteps,callback=callback)
    print(f"✅ done in {time.time()-t0:.1f}s\n")

    print("┌──────────────────────────┐")
    print("│  FINAL EVALUATION (mean) │")
    print("└──────────────────────────┘")

    
    env_eqn = train_env.envs[0].main_eqn
    #pct, _ = eval_one_equation(model, eval_env, n_episodes=args.num_eval)
    #print(f"{env_eqn}: {pct:6.2f}% success over {args.num_eval} episodes")


    # ------------------------------------------------------------------
    # DEBUG ROLLOUT INSPECTOR
    # ------------------------------------------------------------------
    def debug_rollout(model, env, max_steps=25, show_feats=4):
        """
        Run a *single* episode in `env` with `model`, printing detailed
        per-step information.  `max_steps` guards against infinite loops.
        """
        obs = env.reset()                     # vec-env → dict batch of size 1
        print("\n┌──────────────── Roll-out debug ────────────────┐")
        for t in range(max_steps):
            # --- unpack observation dict (batch axis = 0) -------------
            nfeat   = obs["node_features"][0]              # (N, F)
            ei      = obs["edge_index"][0]                 # (2, E)
            nmask   = obs["node_mask"][0].astype(bool)
            emask   = obs["edge_mask"][0].astype(bool)

            valid_nodes = nmask.sum().item()
            valid_edges = emask.sum().item()

            # --- action mask & policy action --------------------------
            a_mask = env.envs[0].action_mask               # (A,)   numpy/bool
            action, _ = model.predict(obs, deterministic=True)
            a = int(action[0])

            # --- step -------------------------------------------------
            obs, rew, done, info = env.step(action)

            print(f"t={t:02d} | nodes={valid_nodes:3d}  edges={valid_edges:3d} "
                f"| act={a:3d}  mask_ok={bool(a_mask[a])} "
                f"| r={rew[0]:+.2f}  done={done[0]}  solved={info[0].get('is_solved', False)}")

            # show the first few node-feature rows to spot encoding issues
            if t == 0:   # only print once; drop this ‘if’ to print every step
                print("   sample node features (first rows):")
                for row in nfeat[:show_feats]:
                    print("   ", row)          # row is NumPy → no .cpu() needed

            if done[0]:
                print("Episode finished.\n└──────────────────────────────────────────────┘\n")
                return
        print("Max-step limit reached.\n└──────────────────────────────────────────────┘\n")

    # debug_rollout(model, eval_env, max_steps=30)

    Tsolve = callback.Tsolve
    return Tsolve


# ╭────────────────────────────────────────────────────────────────╮
# │ 5)  CLI                                                       │
# ╰────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default='ppo-gnn')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=5*10**5)
    p.add_argument("--main_eqn", type=str, default='a/x+b')
    p.add_argument("--n-num-eval", dest='num_eval',  type=int, default=10)
    p.add_argument("--save-dir", type=str, default="data/gnn/")
    args = p.parse_args()

    experiment = True

    if not experiment:
        main(args)

    else:
        # ------------------------------------------------------------------
        # Config for sweep
        # ------------------------------------------------------------------
        algos = ['ppo', 'ppo-gnn']
        eqns  = [
            'a*x+b',
            'a/x+b',
            'c*(a*x+b)+d',
            'd/(a*x+b)+c',
            'e*(a*x+b)+(c*x+d)',
        ]
        n_trials = 2

        # results[algo][eqn] = list of Tsolve values (one per trial)
        results = {algo: {eqn: [] for eqn in eqns} for algo in algos}

        # ------------------------------------------------------------------
        # Sweep
        # ------------------------------------------------------------------
        for algo in algos:
            print("\n" + "=" * 72)
            print(f"Running {n_trials} trial(s) for algo='{algo}'")
            print("=" * 72)
            for eqn in eqns:
                clean_eqn = eqn.replace(" ", "")
                print(f"\n--- Equation: {clean_eqn} ---")
                for trial_idx in range(n_trials):
                    # Unique, deterministic seed per (algo, eqn, trial)
                    trial_seed = (
                        args.seed
                        + 1000 * algos.index(algo)
                        + 100 * eqns.index(eqn)
                        + trial_idx
                    )

                    # Copy args so we don't mutate the outer Namespace across runs
                    run_args = copy.deepcopy(args)
                    run_args.algo      = algo
                    run_args.main_eqn  = clean_eqn
                    run_args.seed      = trial_seed

                    # (Optional) shorter eval per run? You can override here:
                    # run_args.num_eval = 5

                    print(f"  Trial {trial_idx+1}/{n_trials} (seed={trial_seed}) …")
                    tsolve = main(run_args)  # <-- your training+eval function returns Tsolve
                    results[algo][eqn].append(tsolve)
                    print(f"    -> Tsolve = {tsolve}")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "#" * 70)
        print("# Summary: Tsolve statistics")
        print("# (None = never solved within budget)")
        print("#" * 70)

        for algo in algos:
            print(f"\nAlgo: {algo}")
            all_ts = []  # collect across eqns for an overall stat
            for eqn in eqns:
                ts = results[algo][eqn]
                solved_ts = [t for t in ts if t != -1]
                unsolved  = len(ts) - len(solved_ts)
                if solved_ts:
                    mean = np.mean(solved_ts)
                    std  = np.std(solved_ts)
                    print(f"  {eqn:25s}: {mean:8.1f} ± {std:6.1f} "
                        f"(n={len(solved_ts)}/{len(ts)} solved; "
                        f"{unsolved} unsolved)")
                    all_ts.extend(solved_ts)
                else:
                    print(f"  {eqn:25s}: no solves in {len(ts)} trial(s).")

            # Overall per-algo summary
            if all_ts:
                mean = np.mean(all_ts)
                std  = np.std(all_ts)
                print(f"  {'OVERALL':25s}: {mean:8.1f} ± {std:6.1f} "
                    f"(n={len(all_ts)} solves total)")
            else:
                print(f"  {'OVERALL':25s}: no solves.")


        df_num, df_disp = build_tsolve_dataframe(results, algos=('ppo','ppo-gnn'), eqns=eqns)
        os.makedirs(args.save_dir,exist_ok=True)
        df_num.to_csv(args.save_dir + "tsolve_numeric.csv")

        print("\n\n" + "#" * 70)
        print("# Summary: Tsolve statistics (pandas)")
        print("#" * 70)
        print(df_disp.to_string())







#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# debug_gnn_equations.py · PPO vs GCN / GAT / GIN / GraphSAGE
# (parallel + timestamps + colored tqdm + incremental saves)
# Includes residual + LayerNorm (configurable)
# Supports single-equation (Tsolve) and multi-equation (accuracy) modes
# ────────────────────────────────────────────────────────────────
import os, random, argparse, time, json, tempfile
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Iterable, Dict, Any
from collections import defaultdict
import torch as th
from rllte.xplore.reward import ICM

# ============ third-party deps ============
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool
)
import gymnasium as gym  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList

# ============ env ============
from envs.single_eqn import singleEqn  # adjust path if needed
from envs.multi_eqn_curriculum import multiEqn


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
        # if self.n_calls % self.log_interval == 0 and self.rewards_internal:
        #     mean_intrinsic = np.mean(self.rewards_internal[-self.log_interval:])
        #     min_intrinsic = np.min(self.rewards_internal[-self.log_interval:])
        #     max_intrinsic = np.max(self.rewards_internal[-self.log_interval:])
        #     main_eqn = self.locals["infos"][0]['main_eqn']
        #     #print(f"{main_eqn}: Step {self.num_timesteps}: "
        #     #      f"(min, mean, max)_reward_internal = ({min_intrinsic:.3f}, {mean_intrinsic:.3f}, {max_intrinsic:.3f})\n")

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


# ────────────────────────────────────────────────────────────────
# Curve saving helper
# ────────────────────────────────────────────────────────────────
def save_curves(results, out_dir: Path):
    """
    results[a] = list of run dicts (each may have 'curve': [(t,train,test), ...])
    Saves:
      curves_<algo>.csv  (wide per run index)
      curves_long.csv    (algo, run, t, train_acc, test_acc)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for algo, runs in results.items():
        per_algo = {}
        for ridx, run in enumerate(runs):
            curve = run.get("curve", [])
            if not curve:
                continue
            ts    = [c[0] for c in curve]
            cov   = [c[1] for c in curve]
            train = [c[2] for c in curve]
            test  = [c[3] for c in curve]
            if "t" not in per_algo:
                per_algo["t"] = ts
            # Align lengths cautiously (truncate to min)
            L = min(len(per_algo["t"]), len(ts))
            per_algo["t"] = per_algo["t"][:L]; ts = ts[:L]; train = train[:L]; test = test[:L]
            if ts != per_algo["t"]:
                # Mismatch—skip wide export for this algo/run
                continue
            per_algo[f"train_acc_run{ridx}"] = train
            per_algo[f"test_acc_run{ridx}"]  = test
            for t_val, tr, te in zip(ts, train, test):
                long_rows.append({
                    "algo": algo,
                    "run": ridx,
                    "t": t_val,
                    "train_acc": tr,
                    "test_acc": te
                })
        if len(per_algo) > 1:
            pd.DataFrame(per_algo).to_csv(out_dir / f"curves_{algo}.csv", index=False)
    if long_rows:
        pd.DataFrame(long_rows).to_csv(out_dir / "curves_long.csv", index=False)

# ────────────────────────────────────────────────────────────────
# Monkey-patch accuracy for multiEqn without editing env source
# ────────────────────────────────────────────────────────────────
def _attach_accuracy_if_missing(env):
    """
    Adds compute_train_accuracy / compute_test_accuracy if absent.
    Coverage metric = fraction of distinct tasks solved at least once.
    """
    if all(hasattr(env, m) for m in ("compute_train_accuracy", "compute_test_accuracy")):
        return env

    train_tasks = list(getattr(env, "train_tasks", []))
    test_tasks  = list(getattr(env, "test_tasks", []))

    env._acc_task_solved = {("train", t): False for t in train_tasks}
    env._acc_task_solved.update({("test", t): False for t in test_tasks})

    train_set = set(train_tasks)
    test_set  = set(test_tasks)

    if not hasattr(env, "_orig_step_for_acc"):
        env._orig_step_for_acc = env.step

        def _step_wrapper(action):
            obs, reward, terminated, truncated, info = env._orig_step_for_acc(action)

            current_task = None
            for key in ("current_task", "task", "eqn", "equation", "main_eqn"):
                if key in info:
                    current_task = info[key]
                    break
            if current_task is None:
                current_task = getattr(env, "curr_task", None)

            if current_task is not None:
                if current_task not in train_set and current_task not in test_set:
                    # Default new unseen tasks to train split
                    train_set.add(current_task)
                    env._acc_task_solved[("train", current_task)] = False
                split = "train" if current_task in train_set else ("test" if current_task in test_set else None)
                if split and info.get("is_solved"):
                    env._acc_task_solved[(split, current_task)] = True
            return obs, reward, terminated, truncated, info

        env.step = _step_wrapper

    def compute_train_accuracy():
        keys = [k for k in env._acc_task_solved if k[0] == "train"]
        if not keys: return 0.0
        return sum(env._acc_task_solved[k] for k in keys) / len(keys)

    def compute_test_accuracy():
        keys = [k for k in env._acc_task_solved if k[0] == "test"]
        if not keys: return 0.0
        return sum(env._acc_task_solved[k] for k in keys) / len(keys)

    env.compute_train_accuracy = compute_train_accuracy
    env.compute_test_accuracy  = compute_test_accuracy
    return env

# ────────────────────────────────────────────────────────────────
# Timestamp + logging helpers
# ────────────────────────────────────────────────────────────────
def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def tprint(msg: str):
    print(f"[{_now()}] {msg}")

def fmt_acc(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "NA"

# ────────────────────────────────────────────────────────────────
#  SINGLE MODE SUMMARY
# ────────────────────────────────────────────────────────────────
def build_tsolve_dataframe(results: dict, algos: Iterable[str], eqns: Iterable[str]):
    algos = list(algos); eqns = list(eqns)
    rows = []
    for eqn in eqns:
        row = {"eqn": eqn}
        for a in algos:
            ts = results.get(a, {}).get(eqn, [])
            solved = [t for t in ts if t != -1]
            row[f"{a}_mean"] = np.mean(solved) if solved else np.nan
            row[f"{a}_std"]  = np.std(solved) if solved else np.nan
            row[f"{a}_n"]    = len(solved)
            row[f"{a}_N"]    = len(ts)
        rows.append(row)
    df_num = pd.DataFrame(rows).set_index("eqn")

    def _fmt(r, a):
        N = r.get(f"{a}_N", 0); n = r.get(f"{a}_n", 0)
        if np.isnan(N): N = 0
        if np.isnan(n): n = 0
        N, n = int(N), int(n)
        if N == 0: return "—"
        if n == 0: return f"— (0/{N})"
        mean = r.get(f"{a}_mean", np.nan); std = r.get(f"{a}_std", np.nan)
        if np.isnan(mean) or np.isnan(std): return f"— ({n}/{N})"
        return f"{mean:.0f}±{std:.0f} (n={n}/{N})"

    def _display_name(a):
        return "-".join(part.upper() if part.isalpha() else part for part in a.split('-'))

    disp_cols = { _display_name(a): df_num.apply(lambda r: _fmt(r, a), axis=1) for a in algos }
    df_disp = pd.DataFrame(disp_cols, index=df_num.index)
    return df_num, df_disp

# ────────────────────────────────────────────────────────────────
#  MULTI MODE SUMMARY
# ────────────────────────────────────────────────────────────────
def build_multi_accuracy_dataframe(results: dict, algos: Iterable[str]):
    algos = list(algos)
    rows = []
    for a in algos:
        trials = results.get(a, [])
        train_vals = [d["train_acc"]   for d in trials if d.get("train_acc")   is not None]
        test_vals  = [d["test_acc"]    for d in trials if d.get("test_acc")    is not None]
        cov_vals   = [d["coverage"]    for d in trials if d.get("coverage")    is not None]
        rows.append({
            "algo": a,
            "train_mean": np.mean(train_vals) if train_vals else np.nan,
            "train_std":  np.std(train_vals)  if train_vals else np.nan,
            "train_n":    len(train_vals),
            "test_mean":  np.mean(test_vals)  if test_vals  else np.nan,
            "test_std":   np.std(test_vals)   if test_vals  else np.nan,
            "test_n":     len(test_vals),
            "cov_mean":   np.mean(cov_vals)   if cov_vals   else np.nan,
            "cov_std":    np.std(cov_vals)    if cov_vals   else np.nan,
            "cov_n":      len(cov_vals),
        })

    df_num = pd.DataFrame(rows).set_index("algo")

    def _fmt(mean, std, n):
        if n == 0 or np.isnan(mean): return "—"
        return f"{mean:.3f}±{std:.3f} (n={n})"

    df_disp = pd.DataFrame({
        "Train": df_num.apply(lambda r: _fmt(r.train_mean, r.train_std, r.train_n), axis=1),
        "Test" : df_num.apply(lambda r: _fmt(r.test_mean,  r.test_std,  r.test_n),  axis=1),
        "Cov"  : df_num.apply(lambda r: _fmt(r.cov_mean,   r.cov_std,   r.cov_n),   axis=1),
    }, index=df_num.index)

    return df_num, df_disp


# ────────────────────────────────────────────────────────────────
#  Graph Feature Extractors
# ────────────────────────────────────────────────────────────────
class _BatchGraph(BaseFeaturesExtractor):
    def _merge(self, obs):
        x_b, ei_b = obs["node_features"].float(), obs["edge_index"].long()
        nmask_b, emask_b = obs["node_mask"].bool(), obs["edge_mask"].bool()
        xs, eids, batch, off = [], [], [], 0
        for i in range(x_b.size(0)):
            vn, ve = nmask_b[i], emask_b[i]
            x_i = x_b[i][vn]
            ei  = ei_b[i][:, ve]
            keep = vn[ei[0]] & vn[ei[1]]
            ei  = ei[:, keep] + off
            xs.append(x_i); eids.append(ei)
            batch.append(torch.full((x_i.size(0),), i, dtype=torch.long, device=x_i.device))
            off += x_i.size(0)
        return torch.cat(xs, 0), torch.cat(eids, 1), torch.cat(batch, 0)

class GraphBaseExtractor(_BatchGraph):
    def _maybe_residual(self, prev, new):
        return new + prev if prev.shape == new.shape else new

class GCNExtractor(GraphBaseExtractor):
    def __init__(self, obs_space, features_dim=128, hidden_dims=(64,128),
                 use_residual=True, use_layernorm=True):
        super().__init__(obs_space, features_dim)
        in_dim = obs_space["node_features"].shape[-1]
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            conv = GCNConv(d_in, d_out)
            ln = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
            self.layers.append(nn.ModuleDict({"conv": conv, "ln": ln}))
        self.use_res = use_residual
        self.lin = nn.Linear(dims[-1], features_dim)
    def forward(self, obs):
        x, ei, b = self._merge(obs)
        for layer in self.layers:
            prev = x
            h = layer["conv"](x, ei); h = layer["ln"](h); h = F.relu(h)
            if self.use_res: h = self._maybe_residual(prev, h)
            x = h
        return self.lin(global_mean_pool(x, b))

class GATExtractor(GraphBaseExtractor):
    def __init__(self, obs_space, features_dim=128, hidden_dims=(64,64),
                 heads=4, dropout=0.1, use_residual=True, use_layernorm=True):
        super().__init__(obs_space, features_dim)
        in_dim = obs_space["node_features"].shape[-1]
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            conv = GATConv(d_in, d_out, heads=heads, concat=False, dropout=dropout)
            ln = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
            self.layers.append(nn.ModuleDict({"conv": conv, "ln": ln}))
        self.use_res = use_residual
        self.lin = nn.Linear(dims[-1], features_dim)
    def forward(self, obs):
        x, ei, b = self._merge(obs)
        for layer in self.layers:
            prev = x
            h = layer["conv"](x, ei); h = layer["ln"](h); h = F.elu(h)
            if self.use_res: h = self._maybe_residual(prev, h)
            x = h
        return self.lin(global_mean_pool(x, b))

class GINExtractor(GraphBaseExtractor):
    def __init__(self, obs_space, features_dim=128, hidden_dims=(64,64,64),
                 eps=0.0, train_eps=False, use_residual=True, use_layernorm=True):
        super().__init__(obs_space, features_dim)
        in_dim = obs_space["node_features"].shape[-1]
        dims = [in_dim] + list(hidden_dims)
        self.convs, self.lns = nn.ModuleList(), nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            def mlp(in_c, out_c):
                return nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c))
            conv = GINConv(mlp(d_in, d_out), eps=eps, train_eps=train_eps)
            ln = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
            self.convs.append(conv); self.lns.append(ln)
        self.use_res = use_residual
        self.lin_out = nn.Linear(dims[-1], features_dim)
    def forward(self, obs):
        x, ei, b = self._merge(obs)
        for conv, ln in zip(self.convs, self.lns):
            prev = x
            h = conv(x, ei); h = ln(h); h = F.relu(h)
            if self.use_res: h = self._maybe_residual(prev, h)
            x = h
        return self.lin_out(global_mean_pool(x, b))

class SAGEExtractor(GraphBaseExtractor):
    def __init__(self, obs_space, features_dim=128, hidden_dims=(64,64),
                 use_residual=True, use_layernorm=True):
        super().__init__(obs_space, features_dim)
        in_dim = obs_space["node_features"].shape[-1]
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            conv = SAGEConv(d_in, d_out)
            ln = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
            self.layers.append(nn.ModuleDict({"conv": conv, "ln": ln}))
        self.use_res = use_residual
        self.lin = nn.Linear(dims[-1], features_dim)
    def forward(self, obs):
        x, ei, b = self._merge(obs)
        for layer in self.layers:
            prev = x
            h = layer["conv"](x, ei); h = layer["ln"](h); h = F.relu(h)
            if self.use_res: h = self._maybe_residual(prev, h)
            x = h
        return self.lin(global_mean_pool(x, b))

# ────────────────────────────────────────────────────────────────
#  Policy Factory
# ────────────────────────────────────────────────────────────────
def make_policy(arch: str, hidden_dims, features_dim=128):  # <<< CHG
    fe_cls = {
        "gcn":  GCNExtractor,
        "gat":  GATExtractor,
        "gin":  GINExtractor,
        "sage": SAGEExtractor,
    }[arch]
    class _P(MaskableActorCriticPolicy):
        def __init__(self, *a, **kw):
            super().__init__(
                *a,
                features_extractor_class=fe_cls,
                features_extractor_kwargs=dict(
                    features_dim=features_dim,
                    hidden_dims=tuple(hidden_dims)  # <<< NEW
                ),
                **kw
            )
    return _P

# ────────────────────────────────────────────────────────────────
#  Callbacks
# ────────────────────────────────────────────────────────────────
class SolveLoggerMulti(BaseCallback):
    """
    Callback for multi–equation training.

    • Logs reward summaries periodically.
    • Tracks *coverage* = fraction of distinct **train** equations solved
      at least once during training.
    • At every ``eval_freq`` steps runs roll-outs on *all* train / test
      equations (``n_eval_rollouts`` each) and reports average success rates.
    • Optional early stop once a test-accuracy threshold is reached.
    """

    # --------------------------------------------------------------
    # constructor
    # --------------------------------------------------------------
    def __init__(
        self,
        eval_freq: int,
        log_freq: Optional[int] = None,
        *,
        n_eval_rollouts: int = 1,
        max_eval_steps: int = 100,
        early: bool = True,
        target_test_acc: float = 1.0,
        target_train_acc: Optional[float] = None,
        algo_name = None,
    ):
        super().__init__()
        self.eval_freq          = eval_freq
        self.log_freq           = log_freq or eval_freq
        self.n_eval_rollouts    = n_eval_rollouts
        self.max_eval_steps     = max_eval_steps
        self.early              = early
        self.target_test_acc    = target_test_acc
        self.target_train_acc   = target_train_acc
        self.algo_name          = algo_name

        # live state ------------------------------------------------
        self.reward_buf: list[float]                       = []
        self.equations_solved: set[str]                    = set()
        self.curve: list[tuple[int, float, float, float]]  = []   # (t, cov, train, test)

        self.T                   = -1      # timestep of early-stop success
        self.last_eval           = 0
        self.last_log            = 0
        self.last_train_acc      = None
        self.last_test_acc       = None

        # filled on _on_training_start -----------------------------
        self.train_eqns: list[str] = []
        self.test_eqns:  list[str] = []
        self.total_num_eqns: int   = 0
        self._state_rep: str       = "integer_1d"

        # evaluation-env cache -------------------------------------
        self._eval_envs: dict[str, "gym.Env"] = {}

    # --------------------------------------------------------------
    # helpers
    # --------------------------------------------------------------
    def _emit_reward_log(self) -> None:
        if not self.reward_buf:
            return
        self.reward_buf.clear()

    # lazy-create / fetch single-equation env ----------------------
    def _get_eval_env(self, eqn: str):
        if eqn not in self._eval_envs:
            from envs.single_eqn import singleEqn
            from sb3_contrib.common.wrappers import ActionMasker

            base_env = singleEqn(
                eqn,
                state_rep=self._state_rep,
                normalize_rewards=True,
            )
            self._eval_envs[eqn] = ActionMasker(base_env, lambda e: e.action_mask)

        return self._eval_envs[eqn]      # ← just return the cached masked env

    @torch.no_grad()
    def _rollout_single_equation(self, eqn: str) -> float:
        env = self._get_eval_env(eqn)
        solved = 0
        for _ in range(self.n_eval_rollouts):
            obs, _ = env.reset()

            for _ in range(self.max_eval_steps):
                # pull mask safely
                try:
                    mask = env.get_wrapper_attr('action_mask')
                except Exception:
                    mask = getattr(env.unwrapped, 'action_mask', None)

                if mask is not None:
                    mask = np.asarray(mask, dtype=bool)
                    if not mask.any():          # ← guard: all invalid
                        # skip / random fallback / count as fail
                        break

                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                    action_masks=mask
                )
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if info.get("is_solved"):
                        solved += 1
                    break
        return solved / self.n_eval_rollouts


    # compute train / test accuracy -------------------------------
    def _compute_accuracy(self) -> tuple[float, float]:
        env = self.training_env.envs[0].unwrapped
        # slow path – manual roll-outs (cached envs)
        train_acc = (
            np.mean([self._rollout_single_equation(e) for e in self.train_eqns])
            if self.train_eqns else float("nan")
        )
        test_acc = (
            np.mean([self._rollout_single_equation(e) for e in self.test_eqns])
            if self.test_eqns else float("nan")
        )
        return train_acc, test_acc

    # --------------------------------------------------------------
    # SB3 hooks
    # --------------------------------------------------------------
    def _on_training_start(self) -> None:
        env = self.training_env.envs[0].unwrapped
        self.train_eqns = (
            getattr(env, "train_eqns", None)
            or getattr(env, "train_equations", None)
            or list(getattr(env, "train_tasks", []))
        )
        self.test_eqns = (
            getattr(env, "test_eqns", None)
            or getattr(env, "test_equations", None)
            or list(getattr(env, "test_tasks", []))
        )
        self.total_num_eqns = len(self.train_eqns)
        self._state_rep = getattr(env, "state_rep", "integer_1d")

        tprint(f"📝 detected {self.total_num_eqns} train eqns, "
               f"{len(self.test_eqns)} test eqns")

    def _on_step(self) -> bool:
        self.reward_buf.append(float(self.locals["rewards"][0]))

        # newly solved equations -----------------------------------
        for info in self.locals["infos"]:
            if info.get("is_solved"):
                eqn = info.get("main_eqn")
                if eqn and eqn not in self.equations_solved:
                    lhs, rhs = info.get("lhs"), info.get("rhs")
                    #tprint(f"{self.algo_name}| ✅ solved {eqn} ==> {lhs} = {rhs}  t={self.num_timesteps}")
                    self.equations_solved.add(eqn)

        # periodic reward log --------------------------------------
        if (self.num_timesteps - self.last_log) >= self.log_freq:
            self._emit_reward_log()
            self.last_log = self.num_timesteps

        # periodic accuracy evaluation -----------------------------
        if (self.num_timesteps - self.last_eval) >= self.eval_freq:
            self.last_eval = self.num_timesteps
            train_acc, test_acc = self._compute_accuracy()
            cov = len(self.equations_solved) / self.total_num_eqns if self.total_num_eqns else float("nan")

            self.last_train_acc, self.last_test_acc = train_acc, test_acc
            self.curve.append((self.num_timesteps, cov, train_acc, test_acc))

            tprint(f"{self.algo_name}|[{self.num_timesteps}] "
                   f"cov={cov:.3f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

            # optional early stop -----------------------------------
            if self.early and not np.isnan(test_acc):
                if (test_acc >= self.target_test_acc and
                    (self.target_train_acc is None or train_acc >= self.target_train_acc)):
                    tprint(f"🎯 early-stop at t={self.num_timesteps} "
                           f"(train={fmt_acc(train_acc)}, test={fmt_acc(test_acc)})")
                    self.T = self.num_timesteps
                    return False
        return True

    # clean-up cached envs -----------------------------------------
    def _on_training_end(self) -> None:
        for env in self._eval_envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._eval_envs.clear()



# ────────────────────────────────────────────────────────────────
#  Training Run
# ────────────────────────────────────────────────────────────────
def run_training(algo: str, eqn: str, env_type: str,
                 seed: int, steps: int, use_gpu: bool, gen_tag: str,
                 n_layers: int, n_hidden: int, ent_coeff: float):  # <<< CHG
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    hidden_list = [n_hidden] * n_layers  # <<< NEW

    def make_env():
        graph_mode = any(tag in algo for tag in ("gcn","gat","gin","sage"))
        state_rep = "graph_integer_2d" if graph_mode else "integer_1d"
        if env_type == "single":
            env = singleEqn(eqn, state_rep=state_rep, normalize_rewards=True)
        elif env_type == "multi":
            env = multiEqn(generalization=gen_tag, state_rep=state_rep, normalize_rewards=True)
            _attach_accuracy_if_missing(env)
        else:
            raise ValueError(f"Unknown env_type={env_type}")
        return ActionMasker(env, lambda e: e.action_mask)

    env = DummyVecEnv([make_env])

    # policy + kwargs ---------------------------------------------
    if  algo in ["ppo",'ppo-icm']:
        pol, pkw = "MlpPolicy", dict(net_arch=hidden_list)  # <<< CHG
    elif algo == "ppo-gcn":
        pol, pkw = make_policy("gcn",  hidden_list), {}
    elif algo == "ppo-gat":
        pol, pkw = make_policy("gat",  hidden_list), {}
    elif algo == "ppo-gin":
        pol, pkw = make_policy("gin",  hidden_list), {}
    elif algo == "ppo-sage":
        pol, pkw = make_policy("sage", hidden_list), {}
    else:
        raise ValueError(algo)

    model = MaskablePPO(
        pol, env, policy_kwargs=pkw,
        n_steps=2048, batch_size=512, n_epochs=4,
        learning_rate=3e-4, gamma=0.99, ent_coef=ent_coeff,  # <<< CHG
        seed=seed, device=device, verbose=0
    )

    eval_freq = max(steps // 10, 1000)
    eval_freq = 10**5

    if env_type == "single":
        cb = SolveLogger(freq=eval_freq)  # NOTE: assumes you defined SolveLogger elsewhere
    else:
        cb = SolveLoggerMulti(
            eval_freq=eval_freq,
            log_freq=max(1, eval_freq // 2),
            early=True,
            target_test_acc=1.0,
            target_train_acc=None,
            algo_name = algo
        )

        if algo == 'ppo-icm':
            irs = ICM(env, device=device)
            callback_intrinsic_reward = IntrinsicReward(irs)
            cb = [cb, callback_intrinsic_reward]


    try:
        model.learn(total_timesteps=steps, callback=cb)
    except Exception:
        tprint("[TRAIN ERROR]")
        import traceback; traceback.print_exc()
        if env_type == "single":
            return {"Tsolve": -1}
        return {"Tsolve": -1, "train_acc": None, "test_acc": None, "curve": []}

    if env_type == "single":
        return {"Tsolve": cb.T}
    else:
        # Handle cb as list or single
        main_cb = cb[0] if isinstance(cb, list) else cb
        cov_last = main_cb.curve[-1][1] if main_cb.curve else None
        return {
            "Tsolve": main_cb.T,
            "train_acc": getattr(main_cb, "last_train_acc", None),
            "test_acc": getattr(main_cb, "last_test_acc", None),
            "coverage": cov_last,
            "curve": list(getattr(main_cb, "curve", []))
        }

def worker(job):
    (a, e, seed, steps, use_gpu, env_type, gen_tag,
     n_layers, n_hidden, ent_coeff) = job  # <<< CHG
    try:
        out = run_training(a, e, env_type, seed, steps, use_gpu, gen_tag,
                           n_layers, n_hidden, ent_coeff)  # <<< CHG
    except Exception:
        import traceback; traceback.print_exc()
        out = {"Tsolve": -1}
    return a, e, out

# ────────────────────────────────────────────────────────────────
#  Incremental save helper
# ────────────────────────────────────────────────────────────────
def save_partial(results, out_dir: Path, algos, eqns, env_type: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_json = tempfile.mkstemp(dir=out_dir, prefix="partial_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(results, f)
    os.replace(tmp_json, out_dir / "partial_results.json")

    if env_type == "single":
        df_num, _ = build_tsolve_dataframe(results, algos, eqns)
        csv_name = "partial_tsolve_numeric.csv"
    else:
        df_num, _ = build_multi_accuracy_dataframe(results, algos)
        csv_name = "partial_accuracy_numeric.csv"

    fd, tmp_csv = tempfile.mkstemp(dir=out_dir, prefix="partial_", suffix=".csv")
    with os.fdopen(fd, "w") as f:
        df_num.to_csv(f)
    os.replace(tmp_csv, out_dir / csv_name)
    tprint("Partial results saved.")

# ────────────────────────────────────────────────────────────────
#  Sweep
# ────────────────────────────────────────────────────────────────
def sweep(args):
    ALGOS = ["ppo", "ppo-gcn", "ppo-gat", "ppo-gin", "ppo-sage"][:1]
    ALGOS = ['ppo', 'ppo-icm']
    ALL_EQNS = [
        "a*x", "x+b", "a*x+b", "a/x+b",
        "c*(a*x+b)+d", "d/(a*x+b)+c",
        "e*(a*x+b)+(c*x+d)", "(ax+b)/(cx+d)+e"
    ]

    EQNS = ALL_EQNS if args.env_type == "single" else ["__multi__"]
    TRIALS = args.trials

    if args.env_type == "single":
        results: Dict[str, Any] = {a: {e: [] for e in EQNS} for a in ALGOS}
    else:
        results = {a: [] for a in ALGOS}

    jobs = []
    for a in ALGOS:
        for e in EQNS:
            for k in range(TRIALS):
                seed = args.seed + 10_000 * ALGOS.index(a) + 100 * k
                jobs.append((
                    a, e, seed, args.timesteps, args.use_gpu,
                    args.env_type, args.gen,
                    args.n_layers, args.n_hidden_dim, args.ent_coeff    # <<< NEW
                ))
    random.shuffle(jobs)

    if args.num_workers in (-1, 1):
        n_workers = 1
    elif args.num_workers == 0:
        n_workers = os.cpu_count()
    else:
        n_workers = args.num_workers
    parallel = n_workers > 1
    tprint(f"Multiprocessing: {'ON' if parallel else 'OFF'} ({n_workers if parallel else 1} proc)")

    pbar = tqdm(total=len(jobs), ncols=100, desc="Jobs", colour="cyan")
    start = time.time()
    out_dir = Path(args.save_dir)

    try:
        if parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(worker, j) for j in jobs]
                for fut in as_completed(futures):
                    a, e, out = fut.result()
                    if args.env_type == "single":
                        results[a][e].append(out["Tsolve"])
                        tprint(f"→ {a:7s} | {e:18s} | Tsolve={out['Tsolve']}")
                        pbar.set_description(f"{a}:{e} last={out['Tsolve']}")
                    else:
                        results[a].append(out)
                        tr, te = out.get("train_acc"), out.get("test_acc")
                        tprint(f"→ {a:7s} | train_acc={fmt_acc(tr)} test_acc={fmt_acc(te)} T={out['Tsolve']}")
                        pbar.set_description(f"{a} acc={fmt_acc(te)}")
                    pbar.update(1)
                    save_partial(results, out_dir, ALGOS, EQNS, args.env_type)
        else:
            for job in jobs:
                t0 = time.time()
                a, e, out = worker(job)
                if args.env_type == "single":
                    results[a][e].append(out["Tsolve"])
                    tprint(f"→ {a:7s} | {e:18s} | Tsolve={out['Tsolve']} | wall={time.time()-t0:.1f}s")
                    pbar.set_description(f"{a}:{e} last={out['Tsolve']}")
                else:
                    results[a].append(out)
                    tr, te = out.get("train_acc"), out.get("test_acc")
                    tprint(f"→ {a:7s} | train_acc={fmt_acc(tr)} test_acc={fmt_acc(te)} T={out['Tsolve']} | wall={time.time()-t0:.1f}s")
                    pbar.set_description(f"{a} acc={fmt_acc(te)}")
                pbar.update(1)
                save_partial(results, out_dir, ALGOS, EQNS, args.env_type)
    except KeyboardInterrupt:
        tprint("KeyboardInterrupt — saving final partial snapshot before exit.")
        save_partial(results, out_dir, ALGOS, EQNS, args.env_type)
        raise
    finally:
        pbar.close()

    tprint(f"Total sweep wall-time: {(time.time()-start)/60:.2f} min")

    if args.env_type == "single":
        df_num, df_disp = build_tsolve_dataframe(results, ALGOS, EQNS)
        df_num.to_csv(out_dir / "tsolve_numeric.csv")
        tprint("Summary table (Tsolve):\n" + df_disp.to_string())
        tprint(f"Saved → {out_dir/'tsolve_numeric.csv'}")
    else:
        df_num, df_disp = build_multi_accuracy_dataframe(results, ALGOS)
        df_num.to_csv(out_dir / "accuracy_numeric.csv")
        tprint("Summary table (Accuracy):\n" + df_disp.to_string())
        tprint(f"Saved → {out_dir/'accuracy_numeric.csv'}")
        save_curves(results, out_dir)
        tprint(f"Saved curves → {out_dir}")

# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument('--env_type', type=str, choices=['single','multi'], default='multi')
    pa.add_argument('--gen', type=str, default='poesia-small', help="generalization tag (multi mode)")
    pa.add_argument('--trials', type=int, default=4, help="trials per algo/eqn (single) or per algo (multi)")
    pa.add_argument("--timesteps", type=int, default=3*10**6)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--save-dir", type=str, default="data/gnn_poesia_small/")
    pa.add_argument("--num-workers", type=int, default=8,
                    help="-1 or 1=sequential · 0=all cores · N=explicit processes")
    pa.add_argument("--use-gpu", action="store_true", help="allow GPU inside workers")

    # <<< NEW args
    pa.add_argument("--n_layers", type=int, default=2,
                    help="Number of hidden layers for MLP/GNN feature extractor")
    pa.add_argument("--n_hidden_dim", type=int, default=512,
                    help="Hidden dim size for each hidden layer in MLP/GNN")
    pa.add_argument("--ent_coeff", type=float, default=0.05,
                    help="Entropy coefficient for PPO")

    args = pa.parse_args()

    args.save_dir = os.path.join('data', f'gnn_{args.gen}')

    sweep(args)

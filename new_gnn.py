#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# debug_gnn_equations.py · PPO vs GCN vs GAT
# (parallel + timestamps + colored tqdm + incremental saves)
# ────────────────────────────────────────────────────────────────
import os, random, argparse, time, json, numpy as np, pandas as pd, tempfile
from pathlib import Path
from contextlib import suppress
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# ============ third-party deps ============
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

# ============ env ============
from envs.single_eqn import singleEqn  # adjust path if needed


# ────────────────────────────────────────────────────────────────
# Timestamp + logging helpers
# ────────────────────────────────────────────────────────────────
def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def tprint(msg: str):
    print(f"[{_now()}] {msg}")


# ────────────────────────────────────────────────────────────────
#  util: formatted Tsolve dataframe  (3 fixed columns)
# ────────────────────────────────────────────────────────────────
def build_tsolve_dataframe(results: dict, algos, eqns: list[str]):
    """
    Return:
      df_num  – numeric stats (mean/std/n/N) for each algo+eqn
      df_disp – pretty strings with exactly columns: PPO, PPO-GNN, PPO-GAT
    """
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
        n, N = int(r[f"{a}_n"]), int(r[f"{a}_N"])
        if n == 0:
            return f"— (0/{N})"
        return f"{r[f'{a}_mean']:.0f}±{r[f'{a}_std']:.0f} (n={n}/{N})"

    df_disp = pd.DataFrame({
        "PPO"     : df_num.apply(lambda r: _fmt(r, 'ppo'),     axis=1),
        "PPO-GNN" : df_num.apply(lambda r: _fmt(r, 'ppo-gnn'), axis=1),
        "PPO-GAT" : df_num.apply(lambda r: _fmt(r, 'ppo-gat'), axis=1),
    }, index=df_num.index)

    return df_num, df_disp


# ────────────────────────────────────────────────────────────────
#  Graph feature extractors (GCN / GAT)
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


class GCNExtractor(_BatchGraph):
    def __init__(self, obs_space, features_dim=128):
        super().__init__(obs_space, features_dim)
        c = obs_space["node_features"].shape[-1]
        self.c1, self.c2 = GCNConv(c, 64), GCNConv(64, 128)
        self.lin = nn.Linear(128, features_dim)

    def forward(self, obs):
        x, ei, b = self._merge(obs)
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        return self.lin(global_mean_pool(x, b))


class GATExtractor(_BatchGraph):
    def __init__(self, obs_space, features_dim=128, hidden=64, heads=4, dropout=0.1):
        super().__init__(obs_space, features_dim)
        c = obs_space["node_features"].shape[-1]
        self.g1 = GATConv(c, hidden, heads=heads, concat=False, dropout=dropout)
        self.g2 = GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
        self.lin = nn.Linear(hidden, features_dim)

    def forward(self, obs):
        x, ei, b = self._merge(obs)
        x = F.elu(self.g1(x, ei))
        x = F.elu(self.g2(x, ei))
        return self.lin(global_mean_pool(x, b))


# ────────────────────────────────────────────────────────────────
#  Switchable policy
# ────────────────────────────────────────────────────────────────
def make_policy(arch: str):
    fe_cls = {"gcn": GCNExtractor, "gat": GATExtractor}[arch]
    class _P(MaskableActorCriticPolicy):
        def __init__(self, *a, **kw):
            super().__init__(
                *a,
                features_extractor_class=fe_cls,
                features_extractor_kwargs=dict(features_dim=128),
                **kw
            )
    return _P


# ────────────────────────────────────────────────────────────────
#  Callback
# ────────────────────────────────────────────────────────────────
class SolveLogger(BaseCallback):
    def __init__(self, freq=10_000, early=True):
        super().__init__()
        self.freq = freq
        self.early = early
        self.buf = []
        self.T = -1
    def _on_step(self) -> bool:
        self.buf.append(float(self.locals["rewards"][0]))
        for info in self.locals["infos"]:
            if info.get("is_solved"):
                tprint(f"✅ solved {info['main_eqn']}  t={self.num_timesteps}")
                if self.T == -1:
                    self.T = self.num_timesteps
                if self.early:
                    return False
        if self.num_timesteps % self.freq == 0 and self.buf:
            w = np.array(self.buf)
            tprint(f"[{self.num_timesteps}] (min,mean,max)_reward= "
                   f"{w.min():+.2f},{w.mean():+.2f},{w.max():+.2f}")
            self.buf.clear()
        return True


# ────────────────────────────────────────────────────────────────
#  One run
# ────────────────────────────────────────────────────────────────
def run_training(algo, eqn, seed, steps, use_gpu):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    def make_env():
        env = singleEqn(
            eqn,
            state_rep="graph_integer_2d" if 'g' in algo else 'integer_1d',
            normalize_rewards=True
        )
        return ActionMasker(env, lambda e: e.action_mask)

    env = DummyVecEnv([make_env])

    if algo == "ppo":
        pol, pkw = "MlpPolicy", {}
    elif algo == "ppo-gnn":
        pol, pkw = make_policy("gcn"), {}
    elif algo == "ppo-gat":
        pol, pkw = make_policy("gat"), {}
    else:
        raise ValueError(algo)

    model = MaskablePPO(
        pol, env, policy_kwargs=pkw,
        n_steps=2048, batch_size=512, n_epochs=4,
        learning_rate=3e-4, gamma=0.99, ent_coef=0.05,
        seed=seed, device=device, verbose=0
    )
    cb = SolveLogger(freq=max(1000, steps // 10))
    with suppress(Exception):
        model.learn(total_timesteps=steps, callback=cb)
    return cb.T


def worker(job):
    a, e, seed, steps, use_gpu = job
    try:
        Tsolve = run_training(a, e, seed, steps, use_gpu)
    except Exception:
        import traceback; traceback.print_exc()
        Tsolve = -1
    return a, e, seed, Tsolve


# ────────────────────────────────────────────────────────────────
#  Incremental save helper (atomic)
# ────────────────────────────────────────────────────────────────
def save_partial(results, out_dir, algos, eqns):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save raw results JSON atomically
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix="partial_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(results, f)
    os.replace(tmp, out_dir / "partial_results.json")

    # Save numeric CSV atomically
    df_num, _ = build_tsolve_dataframe(results, algos, eqns)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix="partial_", suffix=".csv")
    with os.fdopen(fd, "w") as f:
        df_num.to_csv(f)
    os.replace(tmp, out_dir / "partial_tsolve_numeric.csv")

    tprint("Partial results saved.")


# ────────────────────────────────────────────────────────────────
#  Sweep
# ────────────────────────────────────────────────────────────────
def sweep(args):
    ALGOS = ["ppo", "ppo-gnn", "ppo-gat"]
    EQNS = [
        "a*x", "x+b", "a*x+b", "a/x+b",
        "c*(a*x+b)+d", "d/(a*x+b)+c",
        "e*(a*x+b)+(c*x+d)", "(ax+b)/(cx+d)+e"
    ]
    TRIALS = 3
    results = {a: {e: [] for e in EQNS} for a in ALGOS}

    jobs = []
    for a in ALGOS:
        for e in EQNS:
            for k in range(TRIALS):
                seed = args.seed + 10_000 * ALGOS.index(a) + 100 * k
                jobs.append((a, e, seed, args.timesteps, args.use_gpu))

    random.shuffle(jobs)  # mix difficulty

    n_workers = (os.cpu_count() if args.num_workers in (-1, 0) else args.num_workers)
    parallel = n_workers > 1
    tprint(f"Multiprocessing: {'ON' if parallel else 'OFF'} ({n_workers if parallel else 1} proc)")

    total_jobs = len(jobs)
    pbar = tqdm(total=total_jobs, ncols=100, desc="Jobs", colour="cyan")
    start = time.time()

    out_dir = Path(args.save_dir)

    try:
        if parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(worker, j) for j in jobs]
                for fut in as_completed(futures):
                    a, e, seed, T = fut.result()
                    results[a][e].append(T)
                    tprint(f"→ {a:7s} | {e:18s} | seed={seed:<7d} | Tsolve={T}")
                    pbar.set_description(f"{a}:{e} last={T}")
                    pbar.update(1)
                    save_partial(results, out_dir, ALGOS, EQNS)
        else:
            for job in jobs:
                t0 = time.time()
                a, e, seed, T = worker(job)
                results[a][e].append(T)
                tprint(f"→ {a:7s} | {e:18s} | seed={seed:<7d} | Tsolve={T} | wall={time.time()-t0:.1f}s")
                pbar.set_description(f"{a}:{e} last={T}")
                pbar.update(1)
                save_partial(results, out_dir, ALGOS, EQNS)
    except KeyboardInterrupt:
        tprint("KeyboardInterrupt — saving final partial snapshot before exit.")
        save_partial(results, out_dir, ALGOS, EQNS)
        raise
    finally:
        pbar.close()

    tprint(f"Total sweep wall-time: {(time.time()-start)/60:.2f} min")

    df_num, df_disp = build_tsolve_dataframe(results, ALGOS, EQNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_num.to_csv(out_dir / "tsolve_numeric.csv")
    tprint("Summary table:\n" + df_disp.to_string())
    tprint(f"Saved → {out_dir/'tsolve_numeric.csv'}")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--timesteps", type=int, default=3 * 10**6)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--save-dir", type=str, default="data/gnn_runs/")
    pa.add_argument(
        "--num-workers", type=int, default=9,
        help="-1 or 1 = sequential · 0 = all cores · N = explicit processes"
    )
    pa.add_argument("--use-gpu", action="store_true", help="allow GPU inside workers")
    args = pa.parse_args()
    sweep(args)

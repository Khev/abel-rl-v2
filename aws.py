#!/usr/bin/env python3
"""
Benchmark MaskablePPO variants on multiEqn with timestamp logging + AWS/Spot resiliency.

Additions:
- Periodic checkpoints (local + optional S3).
- Heartbeat & incremental curve flushing.
- Spot interruption detection (EC2 metadata).
- Graceful SIGTERM/SIGINT handling.
- Optional resume from latest checkpoint (local or S3).
- Final tarball upload of the run folder.

Notes:
- Choose ONE multiEqn import below that exists in your repo.
"""

import os, sys, argparse, datetime, time, json, tarfile, signal, threading, math, io
import numpy as np
import pandas as pd
import torch, gymnasium as gym
from torch import as_tensor
import matplotlib.pyplot as plt
from typing import Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

# ⚠️ Keep only the one you actually have:
# from envs.multi_eqn_develop import multiEqn
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
# AWS / S3 helpers (optional)
# ────────────────────────────────────────────────────────────────
_BOTO3_AVAILABLE = False
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except Exception:
    _BOTO3_AVAILABLE = False

def log(msg: str, color: str = None) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    if color == "cyan":
        msg = f"\033[36m{msg}\033[0m"
    print(f"[{stamp}] {msg}", flush=True)

def _s3_client_or_none():
    if not _BOTO3_AVAILABLE:
        return None
    try:
        return boto3.client("s3")
    except Exception as e:
        log(f"S3 client init failed: {e}")
        return None

def s3_upload_file(local_path: str, bucket: str, key: str, max_retries: int = 3) -> bool:
    """Upload file to S3 with basic retries. Returns True on success or if disabled."""
    if not bucket or not _BOTO3_AVAILABLE or not os.path.exists(local_path):
        return False if bucket and _BOTO3_AVAILABLE else False
    s3 = _s3_client_or_none()
    if s3 is None:
        return False
    attempt = 0
    while attempt < max_retries:
        try:
            s3.upload_file(local_path, bucket, key)
            return True
        except (BotoCoreError, ClientError) as e:
            wait = 2 ** attempt
            log(f"S3 upload retry {attempt+1}/{max_retries} for {key}: {e}; sleeping {wait}s")
            time.sleep(wait)
            attempt += 1
    log(f"S3 upload failed after {max_retries} attempts: s3://{bucket}/{key}")
    return False

def s3_download_file(bucket: str, key: str, local_path: str) -> bool:
    if not bucket or not _BOTO3_AVAILABLE:
        return False
    s3 = _s3_client_or_none()
    if s3 is None:
        return False
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        log(f"S3 download failed s3://{bucket}/{key}: {e}")
        return False

def tar_and_upload_dir(dir_path: str, bucket: str, key_prefix: str, tar_name: str = "run_artifacts.tar.gz"):
    """Tar the directory and upload to S3."""
    if not bucket:
        return
    tar_path = os.path.join(dir_path, tar_name)
    try:
        with tarfile.open(tar_path, "w:gz") as tarf:
            tarf.add(dir_path, arcname=".")
        key = f"{key_prefix.rstrip('/')}/{tar_name}"
        ok = s3_upload_file(tar_path, bucket, key)
        if ok:
            log(f"Uploaded tarball to s3://{bucket}/{key}")
    except Exception as e:
        log(f"Tar/upload failed: {e}")

# ────────────────────────────────────────────────────────────────
# Spot interruption monitor (optional)
# ────────────────────────────────────────────────────────────────
class SpotInterruptionWatcher:
    """Poll EC2 metadata for 2-min interruption notice. Sets an Event when detected."""
    META_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    def __init__(self, enable: bool = True, poll_seconds: float = 5.0, timeout: float = 0.3):
        self.enable = enable
        self.poll_seconds = poll_seconds
        self.timeout = timeout
        self._stop_evt = threading.Event()
        self._hit_evt  = threading.Event()
        self._thr: Optional[threading.Thread] = None

    @property
    def triggered(self) -> bool:
        return self._hit_evt.is_set()

    def start(self):
        if not self.enable:
            return
        import urllib.request
        def loop():
            while not self._stop_evt.is_set() and not self._hit_evt.is_set():
                try:
                    with urllib.request.urlopen(self.META_URL, timeout=self.timeout) as r:
                        # If reachable (HTTP 200), a notice is imminent
                        if r.status == 200:
                            self._hit_evt.set()
                            log("⚠️ Spot interruption notice detected via instance metadata.")
                            break
                except Exception:
                    # Not on EC2 or no notice yet
                    pass
                time.sleep(self.poll_seconds)
        self._thr = threading.Thread(target=loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop_evt.set()
        if self._thr is not None:
            self._thr.join(timeout=1.0)

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
    """
    Incrementally logs curves every eval; writes a heartbeat; can stream to S3.
    """
    def __init__(self, log_interval, eval_interval, save_dir, eval_env, algo_name,
                 s3_bucket="", s3_prefix="", verbose=1):
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
        self.train_acc_topn = []
        self.test_acc      = []
        self.test_acc_topn = []

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/")
        os.makedirs(save_dir, exist_ok=True)

    def _write_curves(self):
        curves = pd.DataFrame(
            dict(step=self.logged_steps,
                 coverage=self.coverage,
                 train_acc=self.train_acc,
                 train_acc_topn=self.train_acc_topn,
                 test_acc=self.test_acc,
                 test_acc_topn=self.test_acc_topn)
        )
        csv_path = os.path.join(self.save_dir, "learning_curves.csv")
        curves.to_csv(csv_path, index=False)
        # heartbeat
        hb = dict(
            step=int(self.n_calls),
            time_utc=datetime.datetime.utcnow().isoformat() + "Z",
            pid=os.getpid(),
            algo=self.algo_name,
        )
        with open(os.path.join(self.save_dir, "heartbeat.json"), "w") as f:
            json.dump(hb, f)
        # S3 stream (best-effort)
        if self.s3_bucket:
            s3_upload_file(csv_path, self.s3_bucket, f"{self.s3_prefix}/learning_curves.csv")
            s3_upload_file(os.path.join(self.save_dir, "heartbeat.json"),
                           self.s3_bucket, f"{self.s3_prefix}/heartbeat.json")

    def _on_training_start(self):
        self.train_eqns = self.training_env.get_attr("train_eqns")[0]
        self.test_eqns  = self.training_env.get_attr("test_eqns")[0]
        self._write_curves()  # empty seed files

    def _on_step(self):
        # coverage
        for info in self.locals["infos"]:
            if info.get("is_solved"):
                if info.get("main_eqn") not in self.eqns_solved:
                    main_eqn, lhs, rhs = info.get("main_eqn"), info.get("lhs"), info.get("rhs")
                    print(f"\033[33mSolved {main_eqn} ==> {lhs} = {rhs} at Nstep = {self.n_calls} \033[0m")
                self.eqns_solved.add(info.get("main_eqn"))

        if self.eval_env and self.n_calls % self.eval_interval == 0:
            train_res        = evaluate_agent(self.model, self.eval_env, self.train_eqns)
            test_res_greedy  = evaluate_agent(self.model, self.eval_env, self.test_eqns)
            _, agg_topn_train = eval_success_at_n(self.model, self.eval_env, self.train_eqns, n_trials=10, max_steps=30)
            _, agg_topn       = eval_success_at_n(self.model, self.eval_env, self.test_eqns, n_trials=10, max_steps=30)

            self.logged_steps.append(self.n_calls)
            self.coverage.append(len(self.eqns_solved) / len(self.train_eqns))
            self.train_acc.append(np.mean(list(train_res.values())))
            self.test_acc.append(np.mean(list(test_res_greedy.values())))
            self.train_acc_topn.append(agg_topn_train["success_rate"])
            self.test_acc_topn.append(agg_topn["success_rate"])

            log(f"[{self.algo_name}] t={self.n_calls:6d}| "
                f"cov {self.coverage[-1]:.2f} | "
                f"train {self.train_acc[-1]:.2f} | "
                f"train@10 {self.train_acc_topn[-1]:.2f} | "
                f"test(greedy) {self.test_acc[-1]:.2f} | "
                f"test@10 {self.test_acc_topn[-1]:.2f}")

            # flush progress + heartbeat (and push to S3 if configured)
            self._write_curves()
        return True

    def _on_training_end(self):
        self._write_curves()
        log(f"Saved learning curves → {os.path.join(self.save_dir, 'learning_curves.csv')}")

class CheckpointAndSpotCallback(BaseCallback):
    """
    Saves checkpoints every N steps and on shutdown/spot notice.
    Also pushes checkpoints to S3 (best-effort).
    """
    def __init__(self, save_dir, checkpoint_interval_steps: int,
                 spot_watcher: Optional[SpotInterruptionWatcher] = None,
                 s3_bucket: str = "", s3_prefix: str = ""):
        super().__init__()
        self.save_dir = save_dir
        self.interval = max(1, int(checkpoint_interval_steps))
        self.spot = spot_watcher
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/")
        self._last_saved = 0
        os.makedirs(save_dir, exist_ok=True)
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._sig_handler)
        signal.signal(signal.SIGINT,  self._sig_handler)
        self._stop_flag = threading.Event()

    def _sig_handler(self, signum, frame):
        log(f"Received signal {signum}, requesting graceful stop…")
        self._stop_flag.set()

    def _save_ckpt(self, step: int):
        # Save SB3 model
        ckpt_name = f"ckpt_step_{step}.zip"
        ckpt_path = os.path.join(self.save_dir, ckpt_name)
        latest_path = os.path.join(self.save_dir, "latest_model.zip")
        try:
            self.model.save(ckpt_path)
            self.model.save(latest_path)
            log(f"Checkpoint saved at step={step}: {ckpt_name}")
            if self.s3_bucket:
                s3_upload_file(ckpt_path, self.s3_bucket, f"{self.s3_prefix}/{ckpt_name}")
                s3_upload_file(latest_path, self.s3_bucket, f"{self.s3_prefix}/latest_model.zip")
        except Exception as e:
            log(f"Checkpoint save failed: {e}")

    def _on_step(self) -> bool:
        step = int(self.n_calls)
        # periodic checkpoint
        if step - self._last_saved >= self.interval:
            self._save_ckpt(step)
            self._last_saved = step
        # watch for spot/shutdown
        if (self.spot and self.spot.triggered) or self._stop_flag.is_set():
            log("Interruption detected — saving emergency checkpoint and stopping training.")
            self._save_ckpt(step)
            # Also push a quick tarball of the dir
            if self.s3_bucket:
                tar_and_upload_dir(self.save_dir, self.s3_bucket, self.s3_prefix, tar_name=f"emergency_{step}.tar.gz")
            return False  # abort training loop
        return True

    def _on_training_end(self) -> None:
        # final save
        self._save_ckpt(int(self.n_calls))

# ────────────────────────────────────────────────────────────────
# env factory
# ────────────────────────────────────────────────────────────────
def make_env(use_lookahead, use_memory, use_curriculum, gen, rank, algo, seed=0):
    def _init():
        if algo == 'ppo-gnn':
            state_rep_temp = 'graph_integer_2d'
        else:
            state_rep_temp = 'integer_1d'
        env = multiEqn(
            generalization=gen, state_rep=state_rep_temp,
            normalize_rewards=True, use_curriculum=use_curriculum,
            use_memory=use_memory, use_lookahead=use_lookahead
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init

# ────────────────────────────────────────────────────────────────
# single training routine (one run)
# ────────────────────────────────────────────────────────────────
def maybe_resume(model_ctor, env, load_path_local: str, s3_bucket: str, s3_key: str):
    """
    Try to resume from a checkpoint.
    Returns (model, resumed_bool). If resume fails, returns a fresh model.
    """
    # Try local first
    if load_path_local and os.path.exists(load_path_local):
        try:
            model = model_ctor.load(load_path_local, env=env)
            log(f"Resumed from local checkpoint: {load_path_local}")
            return model, True
        except Exception as e:
            log(f"Local resume failed ({load_path_local}): {e}")
    # Try S3 if requested
    if s3_bucket and s3_key:
        tmp = load_path_local or os.path.join("/tmp", "latest_model.zip")
        if s3_download_file(s3_bucket, s3_key, tmp):
            try:
                model = model_ctor.load(tmp, env=env)
                log(f"Resumed from S3 checkpoint: s3://{s3_bucket}/{s3_key}")
                return model, True
            except Exception as e:
                log(f"S3 resume failed: {e}")
    # Fresh model
    return None, False

def main(args):
    print('\n')
    params = vars(args).copy()
    print_parameters(params)

    n_envs   = args.n_envs
    seed     = args.seed
    gen      = args.gen
    Ntrain   = args.Ntrain
    save_dir = args.save_dir
    algo     = args.algo
    use_curriculum = args.use_curriculum == 'True'
    use_memory = True if args.use_memory == 'True' else False
    use_lookahead = True if args.use_lookahead == 'True' else False
    net_arch = [args.hidden_dim] * args.n_layers

    vec_env  = SubprocVecEnv([make_env(use_lookahead, use_memory, use_curriculum, gen, i, algo, seed) for i in range(n_envs)], start_method="spawn")
    eval_env = DummyVecEnv([make_env(use_lookahead, use_memory, use_curriculum, gen, 999, algo, seed)])

    # Spot watcher
    spot = SpotInterruptionWatcher(enable=args.enable_spot_monitor)
    spot.start()

    # Model creation / resume
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix.rstrip("/")
    latest_local = os.path.join(save_dir, "latest_model.zip")
    latest_s3_key = f"{s3_prefix}/latest_model.zip" if (s3_bucket and s3_prefix) else ""

    if algo == 'ppo-gnn':
        ctor = MaskablePPO  # get_agent returns a configured MaskablePPO under the hood
        fresh_model = get_agent('ppo-gnn', vec_env)
        model, resumed = (fresh_model, False)
        if args.resume:
            # Try to resume using the same ctor as SB3 (MaskablePPO)
            m, ok = maybe_resume(ctor, vec_env, latest_local, s3_bucket, latest_s3_key if args.resume_from_s3 else "")
            if ok:
                model = m
                resumed = True
    else:
        ctor = MaskablePPO
        base_kwargs = dict(
            policy="MlpPolicy", env=vec_env,
            policy_kwargs=dict(net_arch=net_arch),
            n_steps=2048, batch_size=1024, n_epochs=4,
            learning_rate=3e-4, ent_coef=0.01, gamma=0.99,
            tensorboard_log=f".tensorboard/tensorboard_masked_n{n_envs}",
            seed=seed
        )
        model = ctor(**base_kwargs)
        resumed = False
        if args.resume:
            m, ok = maybe_resume(ctor, vec_env, latest_local, s3_bucket, latest_s3_key if args.resume_from_s3 else "")
            if ok:
                model = m
                resumed = True

    callback_logger = TrainingLogger(
        log_interval=10**5,
        eval_interval=10**5,
        save_dir=save_dir,
        eval_env=eval_env,
        algo_name=f"{algo}-nenvs-{n_envs}",
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix
    )

    ckpt_cb = CheckpointAndSpotCallback(
        save_dir=save_dir,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        spot_watcher=spot,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix
    )

    callbacks = [callback_logger, ckpt_cb]

    if args.use_curiosity == 'True':
        device = 'cpu'
        irs = ICM(vec_env, device=device)
        callbacks.append(RLeXploreWithOnPolicyRL(irs))

    log(f"Starting training … (resumed={resumed})")
    try:
        model.learn(total_timesteps=Ntrain, callback=callbacks)
    finally:
        # Always attempt a final tar + upload (even if interrupted)
        if s3_bucket and s3_prefix:
            tar_and_upload_dir(save_dir, s3_bucket, s3_prefix, tar_name="final_artifacts.tar.gz")
        spot.stop()

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
    p.add_argument("--n_envs",            type=int, default=8,     help="number of parallel environments")
    p.add_argument("--trials",            type=int, default=3,     help="number of trials per scenario (different seeds)")
    p.add_argument("--seed",              type=int, default=14850, help="base seed; per-trial seed = base + 1000*trial_idx")
    p.add_argument("--gen",               type=str, default="abel-small")
    p.add_argument("--save_dir",          type=str, default="data/vectorized_env/")
    p.add_argument("--hidden_dim",        type=int, default=1024)
    p.add_argument("--n_layers",          type=int, default=2)
    p.add_argument("--use_curriculum",    type=str,  default='True')
    p.add_argument("--use_curiosity",     type=str,  default='True')
    p.add_argument("--use_memory",        type=str,  default='True')
    p.add_argument("--use_lookahead",     type=str,  default='False')

    # NEW: AWS / resiliency options
    p.add_argument("--s3_bucket",         type=str,  default="", help="If set, upload artifacts to this S3 bucket")
    p.add_argument("--s3_prefix",         type=str,  default="", help="S3 key prefix under which to upload this run")
    p.add_argument("--checkpoint_interval_steps", type=int, default=200_000, help="Checkpoint save interval")
    p.add_argument("--enable_spot_monitor", type=lambda s: s.lower()=='true', default=True, help="Poll EC2 metadata for Spot 2-min notice")
    p.add_argument("--resume",            type=lambda s: s.lower()=='true', default=True, help="Resume from latest_model.zip if present")
    p.add_argument("--resume_from_s3",    type=lambda s: s.lower()=='true', default=False, help="If no local checkpoint, try S3 latest_model.zip")

    args = p.parse_args()

    # Scenarios to run (fixed set)
    scenarios = [
        ('ppo',     'True'),
        ('ppo',     'False')
        # Add ('ppo-gnn','False') if/when ready on this script
    ]

    baseline = args.Ntrain  # IMPORTANT: keep total frames constant across n_envs

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

            scen = scenario_key(algo_name, use_cur)
            per_trial_dir = f"{args.save_dir.rstrip('/')}/{scen}/n_envs_{args.n_envs}/trial_{t+1}"
            os.makedirs(per_trial_dir, exist_ok=True)
            run_args.save_dir = per_trial_dir

            # Derive per-trial S3 prefix (if bucket configured)
            if args.s3_bucket:
                run_args.s3_prefix = f"{args.s3_prefix.rstrip('/')}/{scen}/n_envs_{args.n_envs}/trial_{t+1}"
            else:
                run_args.s3_prefix = args.s3_prefix

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

    # Optionally push aggregated CSV to S3 (top-level summary)
    if args.s3_bucket:
        s3_key = f"{args.s3_prefix.rstrip('/')}/benchmark_summary_nenvs_{args.n_envs}.csv"
        s3_upload_file(out_csv, args.s3_bucket, s3_key)

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

    if args.s3_bucket:
        s3_key = f"{args.s3_prefix.rstrip('/')}/summary_learning_curves_nenvs_{args.n_envs}.png"
        s3_upload_file(summary_fig_path, args.s3_bucket, s3_key)

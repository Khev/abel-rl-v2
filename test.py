#!/usr/bin/env python
"""
benchmark.py – lightweight benchmark driver for train.py.
Runs every (algo, curiosity) combination for several seeds
and aggregates basic success metrics.

Robust-first design
-------------------
* Uses Python’s high-level **concurrent.futures** API instead of
  low-level multiprocessing.Pool – simpler and handles crashes better.
* Falls back to **single-process** execution when --procs 1 (no fork issues).
* Each worker gets its **own** argparse.Namespace; never shares objects.
* Child exceptions are caught and reported, never hang the driver.
* Works on Linux/Mac/Windows (ProcessPool uses “spawn” automatically).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from train import main  # your training entry point


# -----------------------------------------------------------------------------#
# helpers
# -----------------------------------------------------------------------------#
def build_trial_args(base: argparse.Namespace,
                     algo: str,
                     curiosity: str,
                     seed: int) -> argparse.Namespace:
    """Return an independent Namespace for this trial."""
    trial = argparse.Namespace(**vars(base))  # shallow copy all fields
    trial.algo = algo
    trial.curiosity = curiosity
    trial.seed = seed
    return trial


def run_trial(trial_args: argparse.Namespace) -> tuple[int, int]:
    """Safe wrapper: run train.main(), catch any crash, return (-1,-1)."""
    try:
        return main(trial_args)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {trial_args.algo}/{trial_args.curiosity}/seed{trial_args.seed}: {exc}")
        return -1, -1


# -----------------------------------------------------------------------------#
# driver
# -----------------------------------------------------------------------------#
def main_benchmark():
    cli = argparse.ArgumentParser()
    cli.add_argument("--env-id",    default="single")
    cli.add_argument("--Ntrain",    type=int, default=10**5)
    cli.add_argument("--cuda",      action="store_true")
    cli.add_argument("--eval-freq", type=int, default=10_000)
    cli.add_argument("--n-trials",  type=int, default=4)
    cli.add_argument("--procs",     type=int, default=8, help="parallel workers (1 = no multiprocessing)")
    args = cli.parse_args()

    algos       = ["ppo", "dqn", "dqn-per"]
    curiosities = [None, "ICM", "E3B"]

    Path("runs").mkdir(exist_ok=True)

    # build a flat list of all (algo, cur, seed) combos
    all_trials: list[tuple[str, str, int]] = list(
        itertools.product(algos, curiosities, range(1, args.n_trials + 1))
    )

    # --------------------------------------------------#
    # schedule work
    # --------------------------------------------------#
    results: dict[tuple[str, str], list[tuple[int, int]]] = {k: [] for k in itertools.product(algos, curiosities)}

    if args.procs == 1:
        # single-process fallback (robustest, easiest to debug)
        for algo, cur, seed in all_trials:
            trial_args = build_trial_args(args, algo, cur, seed)
            results[(algo, cur)].append(run_trial(trial_args))
    else:
        # parallel execution
        with cf.ProcessPoolExecutor(max_workers=args.procs) as executor:
            futures = {}
            for algo, cur, seed in all_trials:
                trial_args = build_trial_args(args, algo, cur, seed)
                fut = executor.submit(run_trial, trial_args)
                futures[fut] = (algo, cur)
            for fut in cf.as_completed(futures):
                algo, cur = futures[fut]
                results[(algo, cur)].append(fut.result())

    # --------------------------------------------------#
    # aggregate
    # --------------------------------------------------#
    rows = []
    for (algo, cur), trials in results.items():
        tsolves     = [t[0] for t in trials if t[0] != -1]
        tconverges  = [t[1] for t in trials if t[1] != -1]
        rows.append(dict(
            algo           = algo,
            curiosity      = cur or "none",
            success_rate   = len(tsolves) / args.n_trials,
            mean_tsolve    = np.mean(tsolves) if tsolves else -1,
            std_tsolve     = np.std(tsolves)  if tsolves else 0,
            mean_tconverge = np.mean(tconverges) if tconverges else -1,
            std_tconverge  = np.std(tconverges) if tconverges else 0,
        ))

    df = (pd.DataFrame(rows)
            .sort_values("mean_tsolve")
            .reset_index(drop=True))

    print(f"\nBenchmark results  |  env='{args.env_id}'  |  trials={args.n_trials}")
    print("-" * 80)
    print(df.to_string(index=False, float_format='%.2f'))


if __name__ == "__main__":
    main_benchmark()
